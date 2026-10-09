"""Build the Converse-Diarization leaderboard page from one or more scored runs.

The page is a single self-contained HTML file. It ranks the harness runs by
diarization error rate, breaks every score down into missed speech, false
alarms and speaker confusion, and draws the reference and predicted speaker
timelines of any conversation side by side.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from converse_diarization.dataset import read_jsonl

SUMMARY_FILE = "summary.json"
SCORES_FILE = "scores.jsonl"
PAYLOAD_MARKER = "__PAYLOAD__"
BENCHMARK = "Converse-Diarization"
FAMILY = "Converse"


class ReportError(RuntimeError):
    """A run directory cannot be turned into a report."""


def discover_runs(runs_dir: Path) -> list[Path]:
    """Every direct child of ``runs_dir`` that holds a ``summary.json``, sorted by name."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    found = [child for child in runs_dir.iterdir() if child.is_dir() and (child / SUMMARY_FILE).is_file()]
    return sorted(found, key=lambda child: child.name)


def load_summary(run_dir: Path) -> dict[str, Any]:
    """Read ``summary.json`` from a run directory."""
    path = Path(run_dir) / SUMMARY_FILE
    if not path.is_file():
        raise ReportError(f"{run_dir} has no {SUMMARY_FILE}; score the run first")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(summary, dict) or "model" not in summary:
        raise ReportError(f"{path}: not a run summary")
    return summary


def _pct(rate: float | None) -> float | None:
    """A fraction as a percentage with two decimals, or ``None`` when there is no rate."""
    return None if rate is None else round(float(rate) * 100, 2)


def _share(seconds: float | None, reference_s: float | None) -> float | None:
    """Seconds as a percentage of the reference speaker time."""
    if seconds is None or not reference_s:
        return None
    return round(float(seconds) / float(reference_s) * 100, 2)


def _counts(block: dict[str, Any] | None) -> dict[str, Any]:
    block = block or {}
    return {
        "rate": _pct(block.get("rate")),
        "missed_s": block.get("missed_s"),
        "false_alarm_s": block.get("false_alarm_s"),
        "confusion_s": block.get("confusion_s"),
        "reference_s": block.get("reference_s"),
        "errors_s": block.get("errors_s"),
    }


def _breakdown(block: dict[str, Any] | None) -> dict[str, float | None]:
    block = block or {}
    reference_s = block.get("reference_s")
    return {
        "missed": _share(block.get("missed_s"), reference_s),
        "false_alarm": _share(block.get("false_alarm_s"), reference_s),
        "confusion": _share(block.get("confusion_s"), reference_s),
    }


def _segments(items: Any) -> list[dict[str, Any]]:
    """Timeline regions as ``{speaker, start_s, end_s}`` sorted by start."""
    regions = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        regions.append(
            {"speaker": str(item["speaker"]), "start_s": float(item["start_s"]), "end_s": float(item["end_s"])}
        )
    return sorted(regions, key=lambda region: (region["start_s"], region["speaker"]))


def _conversation(row: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    der = row.get("der") or {}
    der_collar = row.get("der_collar") or {}
    jer = row.get("jer") or {}
    speakers = row.get("speaker_count") or {}
    speaking_time = row.get("speaking_time") or {}
    return {
        "id": row["conversation"],
        "language": row.get("language"),
        "duration_s": row.get("duration_s"),
        "overlap_ratio": _pct(row.get("overlap_ratio")),
        "tier": row.get("tier"),
        "latency_s": row.get("latency_s"),
        "granularity": row.get("granularity"),
        "der": _pct(der.get("rate")),
        "der_collar": _pct(der_collar.get("rate")),
        "jer": _pct(jer.get("rate")),
        "jer_per_speaker": {speaker: _pct(rate) for speaker, rate in (jer.get("per_speaker") or {}).items()},
        "counts": {"der": _counts(der), "der_collar": _counts(der_collar)},
        "speakers": {
            "reference": speakers.get("reference"),
            "hypothesis": speakers.get("hypothesis"),
            "correct": speakers.get("correct"),
            "absolute_error": speakers.get("absolute_error"),
        },
        "assignment": [list(pair) for pair in der.get("assignment") or []],
        "speaking_time": {
            "reference": dict(speaking_time.get("reference") or {}),
            "hypothesis": dict(speaking_time.get("hypothesis") or {}),
        },
        "reference_segments": _segments(timeline.get("reference_segments")),
        "hypothesis_segments": _segments(timeline.get("hypothesis_segments")),
    }


def run_payload(run_dir: Path) -> dict[str, Any]:
    """Collect everything the page needs about one run directory."""
    run_dir = Path(run_dir)
    summary = load_summary(run_dir)
    scores_path = run_dir / SCORES_FILE
    scored_rows = read_jsonl(scores_path) if scores_path.is_file() else []
    timelines = {row["conversation"]: row for row in scored_rows if "conversation" in row}
    rows = summary.get("conversation_scores")
    if not isinstance(rows, list):
        rows = scored_rows
    conversations = [_conversation(row, timelines.get(row["conversation"], row)) for row in rows]

    model = summary["model"]
    der = summary.get("der") or {}
    der_collar = summary.get("der_collar") or {}
    jer = summary.get("jer") or {}
    speaker_count = summary.get("speaker_count") or {}
    totals = summary.get("conversations") or {}
    scored = int(totals.get("scored", len(conversations)))
    total = int(totals.get("total", scored))
    tiers = {}
    for tier, block in (summary.get("tiers") or {}).items():
        tier_der = block.get("der") or {}
        tiers[tier] = {
            "conversations": block.get("conversations"),
            "der": _pct(tier_der.get("rate")),
            "breakdown": _breakdown(tier_der),
            "reference_s": tier_der.get("reference_s"),
        }
    return {
        "id": model["name"],
        "name": model["name"],
        "display_name": model.get("display_name") or model["name"],
        "vendor": model.get("vendor") or model.get("provider") or "",
        "provider": model.get("provider") or "",
        "model": model.get("model") or "",
        "href": model.get("href") or "",
        "notes": model.get("notes") or "",
        "model_digest": summary.get("model_digest") or "",
        "scored_at": summary.get("scored_at") or "",
        "collar_s": summary.get("collar_s"),
        "merge_gap_s": summary.get("merge_gap_s"),
        "der": _pct(der.get("rate")),
        "der_collar": _pct(der_collar.get("rate")),
        "jer": _pct(jer.get("rate")),
        "breakdown": _breakdown(der),
        "breakdown_collar": _breakdown(der_collar),
        "reference_s": der.get("reference_s"),
        "errors_s": der.get("errors_s"),
        "speaker_count": {
            "accuracy": _pct(speaker_count.get("accuracy")),
            "correct": speaker_count.get("correct"),
            "total": speaker_count.get("total"),
            "mean_absolute_error": speaker_count.get("mean_absolute_error"),
        },
        "tiers": tiers,
        "conversations_scored": scored,
        "conversations_total": total,
        "conversations_missing": int(totals.get("missing", max(total - scored, 0))),
        "failures": int(summary.get("failures") or 0),
        "complete": scored >= total and der.get("rate") is not None,
        "conversations": conversations,
    }


def _order(run: dict[str, Any]) -> tuple[int, float, str]:
    der = run["der"]
    return (0 if run["complete"] else 1, float("inf") if der is None else der, run["name"])


def build_payload(run_dirs: list[Path], *, sample_label: str = "This sample") -> dict[str, Any]:
    """The JSON document the page is rendered from: runs ranked by DER, incomplete runs last."""
    runs = sorted((run_payload(Path(run_dir)) for run_dir in run_dirs), key=_order)
    seen: set[str] = set()
    for run in runs:
        base = run["id"]
        suffix = 2
        while run["id"] in seen:
            run["id"] = f"{base}-{suffix}"
            suffix += 1
        seen.add(run["id"])
    dataset: dict[str, Any] = {"digest": "", "conversations": 0}
    if run_dirs:
        first = load_summary(Path(run_dirs[0]))
        dataset = {
            "digest": (first.get("dataset") or {}).get("digest") or "",
            "conversations": int((first.get("conversations") or {}).get("total") or 0),
        }
    return {
        "benchmark": BENCHMARK,
        "family": FAMILY,
        "generated": datetime.now(UTC).strftime("%b %-d, %Y"),
        "sampleLabel": sample_label,
        "runs": runs,
        "dataset": dataset,
    }


def render_html(payload: dict[str, Any]) -> str:
    """Inline ``payload`` into the page template."""
    template = resources.files("converse_diarization").joinpath("templates", "leaderboard.html")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return template.read_text(encoding="utf-8").replace(PAYLOAD_MARKER, data)


def write_report(run_dirs: list[Path], output: Path, *, sample_label: str = "This sample") -> Path:
    """Render the page for ``run_dirs`` and write it to ``output``."""
    output = Path(output)
    html = render_html(build_payload(run_dirs, sample_label=sample_label))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
