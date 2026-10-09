"""Run a diarization model over a dataset and score the result.

A run directory holds everything needed to audit a score:

``run.json``
    Which model config, dataset and mix policy produced the run.
``hypotheses.jsonl``
    One line per diarized conversation: the speaker segments the model
    returned plus timing and provenance. Appended as the run progresses;
    safe to resume.
``failures.jsonl``
    One line per failed attempt with a redacted error message.
``scores.jsonl``
    One row per conversation with DER (no collar and with collar), JER,
    speaker counts and the reference / hypothesis timelines that were compared.
``summary.json``
    Pooled corpus figures for the leaderboard.
``rttm/<conversation>.rttm``
    The scored hypothesis timeline in RTTM form, for other tools.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from converse_diarization import __version__
from converse_diarization.config import ModelConfig
from converse_diarization.dataset import Conversation, Dataset, load_dataset, read_jsonl
from converse_diarization.env import redact
from converse_diarization.metrics import (
    DEFAULT_COLLAR_S,
    MERGE_GAP_S,
    DerCounts,
    Segment,
    der,
    jer,
    overlap_ratio,
    overlap_tier,
    percent,
    speaker_count,
    speaking_time,
    write_rttm,
)
from converse_diarization.mix import MixError, ensure_mix
from converse_diarization.mix import policy as mix_policy
from converse_diarization.providers import Provider, ProviderError, create_provider
from converse_diarization.reference import hypothesis_timeline, reference_timeline

RUN_FILE = "run.json"
HYPOTHESES_FILE = "hypotheses.jsonl"
FAILURES_FILE = "failures.jsonl"
SCORES_FILE = "scores.jsonl"
SUMMARY_FILE = "summary.json"
RTTM_DIR = "rttm"
LOCK_FILE = ".lock"
RUN_SCHEMA = 1

Logger = Callable[[str], None]


class RunError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(redact(message), file=sys.stderr, flush=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RunOptions:
    resume: bool = True
    conversation_ids: tuple[str, ...] = ()
    max_consecutive_failures: int = 3
    collar_s: float = DEFAULT_COLLAR_S


def select_conversations(dataset: Dataset, options: RunOptions) -> list[Conversation]:
    wanted = set(options.conversation_ids)
    unknown = wanted - {conversation.id for conversation in dataset.conversations}
    if unknown:
        raise RunError(f"unknown conversation id(s): {', '.join(sorted(unknown))}")
    selected = [c for c in dataset.conversations if not wanted or c.id in wanted]
    if not selected:
        raise RunError("the selection matched no conversations")
    return selected


def _run_record(dataset: Dataset, config: ModelConfig) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "created_at": _now(),
        "harness": {"name": "converse-diarization", "version": __version__},
        "model": config.to_dict(),
        "model_digest": config.digest(),
        "dataset": {"root": str(dataset.root.resolve()), "digest": dataset.digest(), "source": dataset.source},
        "mix": mix_policy(),
    }


@contextmanager
def run_lock(output: Path):
    """Hold an exclusive lock on ``output`` so two runs never append to the same hypotheses file.

    The lock is advisory (``flock``) and released automatically when the process exits, so a
    crashed run never leaves a stale lock behind.
    """
    output.mkdir(parents=True, exist_ok=True)
    path = output / LOCK_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = ""
            try:
                holder = os.read(fd, 64).decode("ascii", "replace").strip()
            except OSError:
                pass
            who = f" (pid {holder})" if holder else ""
            raise RunError(
                f"another converse-diarization run is already writing to {output}{who}; wait for it or use --output"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)


def _check_existing(output: Path, record: dict[str, Any], options: RunOptions) -> dict[str, Any]:
    run_path = output / RUN_FILE
    if not run_path.exists() or not options.resume:
        return record
    existing = json.loads(run_path.read_text(encoding="utf-8"))
    mismatches = []
    if existing.get("model_digest") != record["model_digest"]:
        mismatches.append("model config")
    if existing.get("dataset", {}).get("digest") != record["dataset"]["digest"]:
        mismatches.append("dataset")
    if mismatches:
        raise RunError(
            f"{output} already holds a run for a different {' and '.join(mismatches)}; "
            "use another --output or --fresh to start over"
        )
    existing["resumed_at"] = _now()
    return existing


def _completed(output: Path, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = output / HYPOTHESES_FILE
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("model_digest") != record["model_digest"]:
            continue
        done[row["conversation"]] = row
    return done


def diarize_dataset(
    dataset: Dataset,
    config: ModelConfig,
    output: Path,
    *,
    options: RunOptions | None = None,
    provider: Provider | None = None,
    log: Logger = _log,
) -> dict[str, Any]:
    """Diarize every selected conversation, resuming where a previous run stopped.

    Returns the counts of diarized, skipped and failed conversations.
    Raises :class:`RunError` if another run already holds ``output``.
    """
    options = options or RunOptions()
    with run_lock(output):
        return _diarize_locked(dataset, config, output, options=options, provider=provider, log=log)


def _diarize_locked(
    dataset: Dataset,
    config: ModelConfig,
    output: Path,
    *,
    options: RunOptions,
    provider: Provider | None,
    log: Logger,
) -> dict[str, Any]:
    record = _check_existing(output, _run_record(dataset, config), options)
    if not options.resume:
        for name in (HYPOTHESES_FILE, FAILURES_FILE, SCORES_FILE, SUMMARY_FILE):
            (output / name).unlink(missing_ok=True)
    _write_json(output / RUN_FILE, record)

    own_provider = provider is None
    provider = provider or create_provider(config)
    problems = provider.check()
    if problems:
        raise RunError(f"{config.name} is not ready to run:\n  - " + "\n  - ".join(problems))

    selection = select_conversations(dataset, options)
    done = _completed(output, record)
    counts = {"diarized": 0, "skipped": 0, "failed": 0, "selected": len(selection)}
    consecutive_failures = 0
    try:
        for index, conversation in enumerate(selection, start=1):
            label = f"[{index}/{len(selection)}] {conversation.id}"
            try:
                mix = ensure_mix(dataset, conversation)
            except MixError as exc:
                raise RunError(str(exc)) from exc
            previous = done.get(conversation.id)
            if previous and previous.get("mix_sha256") == mix.sha256:
                counts["skipped"] += 1
                log(f"{label} already diarized, skipping")
                continue
            log(f"{label} diarizing {mix.path.name} ({mix.duration_s:.0f}s) with {config.name}")
            started = time.perf_counter()
            try:
                hypothesis = provider.diarize(mix.path, language=conversation.language, conversation_id=conversation.id)
            except ProviderError as exc:
                failure = {"retryable": exc.retryable, "status": exc.status, "error": redact(str(exc))}
            except Exception as exc:  # noqa: BLE001 - we want every failure on disk
                failure = {"retryable": False, "status": None, "error": redact(f"{type(exc).__name__}: {exc}")}
            else:
                failure = None
                latency = time.perf_counter() - started
                _append_jsonl(
                    output / HYPOTHESES_FILE,
                    {
                        "conversation": conversation.id,
                        "model": config.name,
                        "model_digest": record["model_digest"],
                        "mix_sha256": mix.sha256,
                        "recorded_at": _now(),
                        "latency_s": round(latency, 3),
                        "granularity": hypothesis.granularity,
                        "duration_s": hypothesis.duration_s,
                        "segments": list(hypothesis.segments),
                        "raw": hypothesis.raw,
                    },
                )
                counts["diarized"] += 1
                consecutive_failures = 0
                log(
                    f"{label} done in {latency:.1f}s, {len(hypothesis.segments)} {hypothesis.granularity}s, "
                    f"{len(hypothesis.speakers)} speaker(s)"
                )
            if failure is not None:
                counts["failed"] += 1
                consecutive_failures += 1
                _append_jsonl(
                    output / FAILURES_FILE, {"conversation": conversation.id, "recorded_at": _now(), **failure}
                )
                log(f"{label} failed: {failure['error']}")
                if consecutive_failures >= options.max_consecutive_failures:
                    log(f"stopping after {consecutive_failures} consecutive failures")
                    break
    finally:
        if own_provider:
            provider.close()
    return counts


def _latest_hypotheses(output: Path, model_digest: str | None) -> dict[str, dict[str, Any]]:
    path = output / HYPOTHESES_FILE
    if not path.exists():
        raise RunError(f"{path} does not exist; run the model first")
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if model_digest and row.get("model_digest") not in (None, model_digest):
            continue
        latest[row["conversation"]] = row
    return latest


def _segments_json(segments: tuple[Segment, ...]) -> list[dict[str, float | str]]:
    return [{"speaker": s.speaker, "start_s": round(s.start_s, 3), "end_s": round(s.end_s, 3)} for s in segments]


def score_conversation(
    conversation: Conversation,
    hypothesis: dict[str, Any],
    *,
    collar_s: float = DEFAULT_COLLAR_S,
) -> dict[str, Any]:
    """Score one conversation's hypothesis row against the dataset reference."""
    duration = conversation.duration_s
    reference = reference_timeline(conversation)
    predicted = hypothesis_timeline(hypothesis.get("segments", []), hypothesis.get("granularity", "segment"), duration)
    plain = der(reference, predicted, collar_s=0.0, duration_s=duration)
    collared = der(reference, predicted, collar_s=collar_s, duration_s=duration)
    jaccard = jer(reference, predicted, duration_s=duration)
    count = speaker_count(reference, predicted, duration_s=duration)
    ratio = overlap_ratio(reference)
    return {
        "conversation": conversation.id,
        "language": conversation.language,
        "duration_s": round(duration, 3),
        "overlap_ratio": round(ratio, 4),
        "tier": overlap_tier(ratio),
        "latency_s": hypothesis.get("latency_s"),
        "granularity": hypothesis.get("granularity", "segment"),
        "der": plain.to_dict(),
        "der_collar": collared.to_dict(),
        "jer": jaccard.to_dict(),
        "speaker_count": count.to_dict(),
        "speaking_time": {
            "reference": {k: round(v, 3) for k, v in speaking_time(reference).items()},
            "hypothesis": {k: round(v, 3) for k, v in speaking_time(predicted).items()},
        },
        "reference_segments": _segments_json(reference),
        "hypothesis_segments": _segments_json(predicted),
    }


def _counts_from(row: dict[str, Any]) -> DerCounts:
    return DerCounts(
        missed_s=row["missed_s"],
        false_alarm_s=row["false_alarm_s"],
        confusion_s=row["confusion_s"],
        reference_s=row["reference_s"],
    )


def score_run(output: Path, *, dataset: Dataset | None = None, collar_s: float | None = None) -> dict[str, Any]:
    """Score the hypotheses in ``output`` against the dataset references and write scores + summary."""
    run_path = output / RUN_FILE
    if not run_path.exists():
        raise RunError(f"{run_path} does not exist")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    dataset = dataset or load_dataset(Path(record["dataset"]["root"]))
    if dataset.digest() != record["dataset"]["digest"]:
        raise RunError("the dataset no longer matches the one this run was diarized from")
    collar = DEFAULT_COLLAR_S if collar_s is None else float(collar_s)
    hypotheses = _latest_hypotheses(output, record.get("model_digest"))
    failures = read_jsonl(output / FAILURES_FILE) if (output / FAILURES_FILE).exists() else []

    scores_path = output / SCORES_FILE
    scores_path.unlink(missing_ok=True)
    rttm_dir = output / RTTM_DIR
    rttm_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    pooled = DerCounts()
    pooled_collar = DerCounts()
    tiers: dict[str, dict[str, Any]] = {}
    jer_values: list[float] = []
    correct = 0
    absolute_errors = 0
    missing = 0
    for conversation in dataset.conversations:
        hypothesis = hypotheses.get(conversation.id)
        if hypothesis is None:
            missing += 1
            continue
        row = score_conversation(conversation, hypothesis, collar_s=collar)
        _append_jsonl(scores_path, row)
        (rttm_dir / f"{conversation.id}.rttm").write_text(
            write_rttm(
                hypothesis_timeline(hypothesis.get("segments", []), row["granularity"], conversation.duration_s),
                conversation.id,
            ),
            encoding="utf-8",
        )
        rows.append(row)
        pooled += _counts_from(row["der"])
        pooled_collar += _counts_from(row["der_collar"])
        tier = tiers.setdefault(row["tier"], {"conversations": 0, "der": DerCounts(), "der_collar": DerCounts()})
        tier["conversations"] += 1
        tier["der"] += _counts_from(row["der"])
        tier["der_collar"] += _counts_from(row["der_collar"])
        if row["jer"]["rate"] is not None:
            jer_values.append(row["jer"]["rate"])
        correct += int(row["speaker_count"]["correct"])
        absolute_errors += int(row["speaker_count"]["absolute_error"])

    scored = len(rows)
    summary = {
        "schema": RUN_SCHEMA,
        "scored_at": _now(),
        "model": record["model"],
        "model_digest": record.get("model_digest"),
        "dataset": record["dataset"],
        "merge_gap_s": MERGE_GAP_S,
        "collar_s": collar,
        "der": pooled.to_dict(),
        "der_collar": pooled_collar.to_dict(),
        "jer": {"rate": (sum(jer_values) / len(jer_values)) if jer_values else None, "conversations": len(jer_values)},
        "speaker_count": {
            "accuracy": (correct / scored) if scored else None,
            "correct": correct,
            "total": scored,
            "mean_absolute_error": (absolute_errors / scored) if scored else None,
        },
        "tiers": {
            name: {
                "conversations": t["conversations"],
                "der": t["der"].to_dict(),
                "der_collar": t["der_collar"].to_dict(),
            }
            for name, t in sorted(tiers.items(), key=lambda item: ("low", "medium", "high").index(item[0]))
        },
        "conversations": {"scored": scored, "missing": missing, "total": len(dataset.conversations)},
        "failures": len(failures),
        "conversation_scores": [
            {k: v for k, v in row.items() if k not in {"reference_segments", "hypothesis_segments"}} for row in rows
        ],
    }
    _write_json(output / SUMMARY_FILE, summary)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Human readable summary for the terminal."""
    model = summary["model"]
    d, c = summary["der"], summary["der_collar"]
    lines = [
        f"{model.get('display_name') or model['name']} ({model['provider']}:{model['model']})",
        f"  DER            {percent(d['rate'])}  (missed {d['missed_s']:.1f}s, false alarm {d['false_alarm_s']:.1f}s, "
        f"confusion {d['confusion_s']:.1f}s / {d['reference_s']:.1f}s of reference speech)",
        f"  DER ±{summary['collar_s']:.2f}s    {percent(c['rate'])}",
        f"  JER            {percent(summary['jer']['rate'])}",
        f"  speaker count  {summary['speaker_count']['correct']}/{summary['speaker_count']['total']} correct",
        f"  conversations  {summary['conversations']['scored']}/{summary['conversations']['total']} scored, "
        f"{summary['failures']} failed attempt(s)",
    ]
    for row in summary["conversation_scores"]:
        lines.append(
            f"  {row['conversation']}  overlap {row['overlap_ratio'] * 100:.1f}% ({row['tier']})  "
            f"DER {percent(row['der']['rate'])}  JER {percent(row['jer']['rate'])}  "
            f"speakers {row['speaker_count']['reference']}/{row['speaker_count']['hypothesis']}"
        )
    return "\n".join(lines)


def load_summary(output: Path) -> dict[str, Any]:
    path = output / SUMMARY_FILE
    if not path.exists():
        raise RunError(f"{path} does not exist; run `converse-diarization score --run {output}` first")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "FAILURES_FILE",
    "HYPOTHESES_FILE",
    "RTTM_DIR",
    "RUN_FILE",
    "SCORES_FILE",
    "SUMMARY_FILE",
    "RunError",
    "RunOptions",
    "diarize_dataset",
    "format_summary",
    "load_summary",
    "run_lock",
    "score_conversation",
    "score_run",
    "select_conversations",
]
