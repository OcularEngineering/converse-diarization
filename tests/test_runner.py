from __future__ import annotations

import json

import pytest

from converse_diarization.config import parse_model_config
from converse_diarization.dataset import read_jsonl
from converse_diarization.mix import ensure_mix
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError
from converse_diarization.runner import (
    FAILURES_FILE,
    HYPOTHESES_FILE,
    RTTM_DIR,
    SCORES_FILE,
    SUMMARY_FILE,
    RunError,
    RunOptions,
    diarize_dataset,
    format_summary,
    run_lock,
    score_run,
)
from tests.fixtures import make_dataset

CONFIG = parse_model_config(
    """
name = "fake-diarizer"
display_name = "Fake Diarizer"
vendor = "Tests"
provider = "fake"
model = "fake-1"
"""
)

# Reference timelines produced by tests.fixtures.make_dataset (after the 0.2 s merge):
#   s1  ch1: [0.0, 1.5] [2.5, 3.0]      ch2: [1.0, 1.5] [2.5, 2.8] [3.5, 3.9]   -> 3.2 s of speaker time
#   s2  ch1: [0.0, 2.0]                 ch2: [2.5, 3.9]                         -> 3.4 s
ANSWERS = {
    # s1: everything right except ch2's last region [3.5, 3.9] is missed -> 0.4 s missed / 3.2 s = 12.5 %
    "s1": [
        {"speaker": "A", "start_s": 0.0, "end_s": 1.5},
        {"speaker": "A", "start_s": 2.5, "end_s": 3.0},
        {"speaker": "B", "start_s": 1.0, "end_s": 1.5},
        {"speaker": "B", "start_s": 2.5, "end_s": 2.8},
    ],
    # s2: perfect, with the labels the other way round -> DER 0 after assignment
    "s2": [
        {"speaker": "B", "start_s": 0.0, "end_s": 2.0},
        {"speaker": "A", "start_s": 2.5, "end_s": 3.9},
    ],
}


def _short(conversation_id: str) -> str:
    return conversation_id.rsplit("-", 1)[-1]


class FakeProvider(Provider):
    key = "fake"

    def __init__(self, config, *, fail=(), answers=None):
        super().__init__(config)
        self.fail = set(fail)
        self.answers = answers or ANSWERS
        self.calls: list[str] = []

    def check(self):
        return []

    def diarize(self, audio, *, language, conversation_id):
        assert audio.is_file() and audio.suffix == ".wav"
        short = _short(conversation_id)
        self.calls.append(short)
        if short in self.fail:
            raise ProviderError("boom", retryable=True, status=503)
        return Hypothesis(segments=tuple(self.answers[short]), granularity="segment", duration_s=4.5)


def test_diarize_scores_and_resumes(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    provider = FakeProvider(CONFIG)
    counts = diarize_dataset(dataset, CONFIG, output, provider=provider)
    assert counts == {"diarized": 2, "skipped": 0, "failed": 0, "selected": 2}
    rows = read_jsonl(output / HYPOTHESES_FILE)
    assert len(rows) == 2 and rows[0]["granularity"] == "segment" and rows[0]["mix_sha256"]

    summary = score_run(output, dataset=dataset)
    by_id = {_short(row["conversation"]): row for row in summary["conversation_scores"]}
    assert by_id["s2"]["der"]["rate"] == pytest.approx(0.0, abs=1e-9)
    assert by_id["s2"]["der"]["assignment"] == [["ch1", "B"], ["ch2", "A"]]
    assert by_id["s1"]["der"]["missed_s"] == pytest.approx(0.4, abs=1e-6)
    assert by_id["s1"]["der"]["rate"] == pytest.approx(0.4 / 3.2, abs=1e-6)
    assert by_id["s1"]["tier"] == "high" and by_id["s2"]["tier"] == "low"
    assert by_id["s1"]["overlap_ratio"] == pytest.approx(0.8 / 2.4, abs=1e-3)
    # pooled: 0.4 s of error over 6.6 s of reference speaker time
    assert summary["der"]["rate"] == pytest.approx(0.4 / 6.6, abs=1e-6)
    assert summary["speaker_count"] == {"accuracy": 1.0, "correct": 2, "total": 2, "mean_absolute_error": 0.0}
    assert set(summary["tiers"]) == {"low", "high"}
    assert summary["conversations"] == {"scored": 2, "missing": 0, "total": 2}

    scores = read_jsonl(output / SCORES_FILE)
    assert len(scores) == 2 and scores[0]["reference_segments"] and scores[0]["hypothesis_segments"]
    rttm = (output / RTTM_DIR / f"{dataset.conversations[0].id}.rttm").read_text(encoding="utf-8")
    assert rttm.startswith("SPEAKER ") and dataset.conversations[0].id in rttm
    text = format_summary(summary)
    assert "DER" in text and "JER" in text and "speaker count  2/2" in text

    # Resume: nothing is re-sent while the mixes are unchanged.
    again = diarize_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG))
    assert again["skipped"] == 2 and again["diarized"] == 0


def test_failures_are_recorded_and_scoring_marks_missing(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    counts = diarize_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG, fail={"s1"}))
    assert counts["failed"] == 1 and counts["diarized"] == 1
    failure = read_jsonl(output / FAILURES_FILE)[0]
    assert failure["retryable"] is True and failure["status"] == 503 and "boom" in failure["error"]
    summary = score_run(output, dataset=dataset)
    assert summary["conversations"] == {"scored": 1, "missing": 1, "total": 2} and summary["failures"] == 1
    assert json.loads((output / SUMMARY_FILE).read_text(encoding="utf-8"))["der"]["rate"] == pytest.approx(0.0)


def test_consecutive_failures_stop_the_run(tmp_path):
    dataset = make_dataset(tmp_path)
    provider = FakeProvider(CONFIG, fail={"s1", "s2"})
    counts = diarize_dataset(
        dataset, CONFIG, tmp_path / "run", provider=provider, options=RunOptions(max_consecutive_failures=1)
    )
    assert counts["failed"] == 1 and provider.calls == ["s1"]


def test_run_directory_refuses_a_different_model_unless_fresh(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    diarize_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG))
    other = parse_model_config('name = "other"\nprovider = "fake"\nmodel = "fake-2"\n')
    with pytest.raises(RunError, match="different model config"):
        diarize_dataset(dataset, other, output, provider=FakeProvider(other))
    counts = diarize_dataset(dataset, other, output, provider=FakeProvider(other), options=RunOptions(resume=False))
    assert counts["diarized"] == 2


def test_selection_and_unknown_conversation(tmp_path):
    dataset = make_dataset(tmp_path)
    provider = FakeProvider(CONFIG)
    only = dataset.conversations[1].id
    counts = diarize_dataset(
        dataset, CONFIG, tmp_path / "run", provider=provider, options=RunOptions(conversation_ids=(only,))
    )
    assert counts["selected"] == 1 and provider.calls == [_short(only)]
    with pytest.raises(RunError, match="unknown conversation"):
        diarize_dataset(
            dataset, CONFIG, tmp_path / "run2", provider=provider, options=RunOptions(conversation_ids=("x",))
        )


def test_a_locked_run_directory_is_refused(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    with run_lock(output):
        with pytest.raises(RunError, match="already writing"):
            diarize_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG))


def test_changed_mix_is_rediarized(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "run"
    diarize_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG))
    rows = read_jsonl(output / HYPOTHESES_FILE)
    # Pretend the first conversation was diarized from some other mix.
    rows[0]["mix_sha256"] = "0" * 64
    (output / HYPOTHESES_FILE).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    provider = FakeProvider(CONFIG)
    counts = diarize_dataset(dataset, CONFIG, output, provider=provider)
    assert counts["diarized"] == 1 and counts["skipped"] == 1
    assert ensure_mix(dataset, dataset.conversations[0]).sha256 != "0" * 64
