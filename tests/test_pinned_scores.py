"""Pinned scores: the published board depends on these exact numbers.

Converse-Diarization publishes a board version per scoring methodology. Anything that changes what
``der``, ``jer``, ``speaker_count`` or the segment rules return for the timelines below (the merge
gap, clipping, the collar zones, the pairing, overlap handling) changes every published number and
is a **new board version**, not a fix to the current one.

If this test fails on purpose: bump the board version in
``src/converse_diarization/boards/converse-diarization.json``, bump the package version, re-pin the
numbers here in the same commit, and say so in the changelog. The site syncs refuse an export whose
harness version differs from the committed board unless told it is a new version, so the pin here is
the first of two gates.
"""

from __future__ import annotations

import pytest

from converse_diarization.metrics import (
    DEFAULT_COLLAR_S,
    MERGE_GAP_S,
    der,
    jer,
    merge_segments,
    overlap_ratio,
    overlap_tier,
    speaker_count,
)

approx = pytest.approx

# Two reference speakers over 20 s with three overlaps (3.9-4.2, 11.3-11.9, 16.2-16.4), a short
# backchannel and a trailing turn; a hypothesis that is slightly early or late at every boundary,
# misses the backchannel, and invents a third speaker for 0.4 s.
REFERENCE = [
    ("ch1", 0.0, 4.2),
    ("ch2", 3.9, 9.5),
    ("ch1", 9.7, 12.0),
    ("ch2", 11.3, 11.9),
    ("ch1", 13.0, 16.4),
    ("ch2", 16.2, 20.0),
]
HYPOTHESIS = [
    ("A", 0.1, 4.5),
    ("B", 4.1, 9.2),
    ("A", 9.9, 11.8),
    ("A", 12.9, 16.6),
    ("B", 16.5, 19.6),
    ("C", 17.0, 17.4),
]
DURATION_S = 20.0

# Word-level output: gaps of 0.1 s merge, 0.35 s does not.
WORDS = [("A", 0.1, 0.5), ("A", 0.6, 1.0), ("A", 1.35, 4.5)]


def test_constants_are_pinned():
    assert MERGE_GAP_S == 0.2
    assert DEFAULT_COLLAR_S == 0.25


def test_merge_rule_is_pinned():
    merged = [(s.speaker, s.start_s, s.end_s) for s in merge_segments(WORDS)]
    assert merged == [("A", 0.1, 1.0), ("A", 1.35, 4.5)]


def test_der_is_pinned():
    plain = der(REFERENCE, HYPOTHESIS, collar_s=0.0, duration_s=DURATION_S).to_dict()
    assert plain["missed_s"] == approx(2.2, abs=1e-9)
    assert plain["false_alarm_s"] == approx(0.9, abs=1e-9)
    assert plain["confusion_s"] == approx(0.1, abs=1e-9)
    assert plain["reference_s"] == approx(19.9, abs=1e-9)
    assert plain["errors_s"] == approx(3.2, abs=1e-9)
    assert plain["rate"] == approx(0.1608040201005025, abs=1e-12)
    assert plain["assignment"] == [["ch1", "A"], ["ch2", "B"]]

    collared = der(REFERENCE, HYPOTHESIS, collar_s=DEFAULT_COLLAR_S, duration_s=DURATION_S).to_dict()
    assert collared["missed_s"] == approx(0.3, abs=1e-9)
    assert collared["false_alarm_s"] == approx(0.45, abs=1e-9)
    assert collared["confusion_s"] == approx(0.0, abs=1e-9)
    assert collared["reference_s"] == approx(15.3, abs=1e-9)
    assert collared["errors_s"] == approx(0.75, abs=1e-9)
    assert collared["rate"] == approx(0.04901960784313708, abs=1e-12)


def test_jer_is_pinned():
    score = jer(REFERENCE, HYPOTHESIS, duration_s=DURATION_S).to_dict()
    assert score["reference_speakers"] == 2
    assert score["per_speaker"] == {"ch1": approx(0.104762, abs=1e-6), "ch2": approx(0.18, abs=1e-6)}
    assert score["error_sum"] == approx(0.284762, abs=1e-6)
    assert score["rate"] == approx(0.1423809523809525, abs=1e-9)
    assert score["assignment"] == [["ch1", "A"], ["ch2", "B"]]


def test_speaker_count_and_tier_are_pinned():
    assert speaker_count(REFERENCE, HYPOTHESIS, duration_s=DURATION_S).to_dict() == {
        "reference": 2,
        "hypothesis": 3,
        "correct": False,
        "absolute_error": 1,
    }
    ratio = overlap_ratio(REFERENCE)
    assert ratio == approx(0.05851063829787229, abs=1e-12)
    assert overlap_tier(ratio) == "low"
