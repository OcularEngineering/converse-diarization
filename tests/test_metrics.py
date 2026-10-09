from __future__ import annotations

import math

import pytest

from converse_diarization.metrics import (
    DEFAULT_COLLAR_S,
    MERGE_GAP_S,
    DerCounts,
    Segment,
    canonical,
    clip_segments,
    der,
    jer,
    merge_segments,
    overlap_ratio,
    overlap_tier,
    percent,
    read_rttm,
    speaker_count,
    speaking_time,
    to_segments,
    write_rttm,
)

approx = pytest.approx


# ---------------------------------------------------------------------------
# Segment / to_segments / clip / merge / canonical


def test_constants():
    assert MERGE_GAP_S == 0.2
    assert DEFAULT_COLLAR_S == 0.25


def test_segment_validation():
    segment = Segment("ch1", 1, 2)
    assert isinstance(segment.start_s, float) and segment.duration_s == 1.0
    assert segment.to_dict() == {"speaker": "ch1", "start_s": 1.0, "end_s": 2.0}
    Segment("ch1", 3.0, 3.0)  # zero length is allowed
    with pytest.raises(ValueError, match="before start_s"):
        Segment("ch1", 2.0, 1.0)
    with pytest.raises(ValueError, match=">= 0"):
        Segment("ch1", -0.1, 1.0)
    with pytest.raises(ValueError, match="finite"):
        Segment("ch1", 0.0, math.nan)
    with pytest.raises(ValueError, match="finite"):
        Segment("ch1", 0.0, math.inf)
    with pytest.raises(ValueError, match="number"):
        Segment("ch1", "0", 1.0)
    with pytest.raises(ValueError, match="speaker"):
        Segment("", 0.0, 1.0)


def test_to_segments_accepts_three_shapes():
    items = [
        Segment("a", 0.0, 1.0),
        ("b", 1, 2),
        ["c", 2.0, 3.0],
        {"speaker": "d", "start_s": 3.0, "end_s": 4.0, "text": "ignored"},
    ]
    assert to_segments(items) == (Segment("a", 0, 1), Segment("b", 1, 2), Segment("c", 2, 3), Segment("d", 3, 4))
    assert to_segments(()) == ()
    assert to_segments(None) == ()
    assert to_segments(segment for segment in items[:1]) == (Segment("a", 0, 1),)


def test_to_segments_rejects_bad_items():
    with pytest.raises(ValueError, match="missing key"):
        to_segments([{"speaker": "a", "start_s": 0.0}])
    with pytest.raises(ValueError, match="3 values|\\(speaker, start_s, end_s\\)"):
        to_segments([("a", 0.0)])
    with pytest.raises(ValueError, match="unsupported type"):
        to_segments([42])
    with pytest.raises(ValueError, match="iterable"):
        to_segments("SPEAKER a 0 1")
    with pytest.raises(ValueError, match="iterable"):
        to_segments(Segment("a", 0, 1))


def test_clip_segments():
    segments = [("a", 0.0, 5.0), ("a", 8.0, 12.0), ("b", 10.0, 11.0), ("b", 2.0, 2.0)]
    assert clip_segments(segments, 10.0) == (Segment("a", 0, 5), Segment("a", 8, 10))
    # None keeps everything but zero-length intervals
    assert clip_segments(segments, None) == (Segment("a", 0, 5), Segment("a", 8, 12), Segment("b", 10, 11))
    with pytest.raises(ValueError, match="> 0"):
        clip_segments(segments, 0.0)
    with pytest.raises(ValueError, match="finite"):
        clip_segments(segments, math.inf)


def test_merge_segments_gap_rule():
    merged = merge_segments([("a", 0.0, 1.0), ("a", 1.2, 2.0)])
    assert merged == (Segment("a", 0.0, 2.0),)
    not_merged = merge_segments([("a", 0.0, 1.0), ("a", 1.201, 2.0)])
    assert not_merged == (Segment("a", 0.0, 1.0), Segment("a", 1.201, 2.0))


def test_merge_segments_is_robust_to_float_noise():
    # 1.1 - 0.9 == 0.20000000000000007 in binary floating point; it must still count as a 0.2 s gap.
    assert merge_segments([("a", 0.5, 0.9), ("a", 1.1, 1.5)]) == (Segment("a", 0.5, 1.5),)


def test_merge_segments_per_speaker_sorted_and_unordered_input():
    merged = merge_segments([("b", 5.0, 6.0), ("a", 3.0, 4.0), ("a", 0.0, 1.0), ("a", 1.1, 2.0), ("b", 6.1, 7.0)])
    assert merged == (Segment("a", 0.0, 2.0), Segment("a", 3.0, 4.0), Segment("b", 5.0, 7.0))
    # different speakers never merge even when touching
    assert merge_segments([("a", 0.0, 1.0), ("b", 1.0, 2.0)]) == (Segment("a", 0, 1), Segment("b", 1, 2))
    # overlapping intervals collapse and the longest end wins
    assert merge_segments([("a", 0.0, 5.0), ("a", 1.0, 2.0), ("a", 4.0, 9.0)]) == (Segment("a", 0, 9),)
    with pytest.raises(ValueError, match="gap_s"):
        merge_segments([("a", 0.0, 1.0)], gap_s=-1)


def test_canonical_clips_and_unions_without_gap_merging():
    segments = [("a", 0.0, 1.0), ("a", 1.0, 2.0), ("a", 2.1, 3.0), ("a", 9.0, 15.0), ("b", 20.0, 30.0)]
    assert canonical(segments, 10.0) == (Segment("a", 0.0, 2.0), Segment("a", 2.1, 3.0), Segment("a", 9.0, 10.0))
    assert canonical(segments) == (
        Segment("a", 0.0, 2.0),
        Segment("a", 2.1, 3.0),
        Segment("a", 9.0, 15.0),
        Segment("b", 20.0, 30.0),
    )


# ---------------------------------------------------------------------------
# DER


def test_der_perfect_match_is_zero():
    reference = [("ch1", 0.0, 5.0), ("ch2", 5.0, 10.0)]
    score = der(reference, reference)
    assert score.rate == 0.0
    assert score.counts.reference_s == approx(10.0)
    assert score.counts.errors_s == 0.0
    assert score.assignment == (("ch1", "ch1"), ("ch2", "ch2"))
    assert score.collar_s == 0.0


def test_der_missed_speech():
    score = der([("ch1", 0.0, 10.0)], [("A", 0.0, 9.0)])
    assert score.counts.missed_s == approx(1.0)
    assert score.counts.false_alarm_s == 0.0
    assert score.counts.confusion_s == 0.0
    assert score.rate == approx(0.1)


def test_der_false_alarm():
    score = der([("ch1", 0.0, 10.0)], [("A", 0.0, 12.0)])
    assert score.counts.false_alarm_s == approx(2.0)
    assert score.counts.missed_s == 0.0
    assert score.rate == approx(0.2)


def test_der_swapped_labels_resolved_by_assignment():
    reference = [("ch1", 0.0, 5.0), ("ch2", 5.0, 10.0)]
    hypothesis = [("B", 0.0, 5.0), ("A", 5.0, 10.0)]
    score = der(reference, hypothesis)
    assert score.rate == 0.0
    assert score.assignment == (("ch1", "B"), ("ch2", "A"))


def test_der_speaker_confusion():
    reference = [("ch1", 0.0, 5.0), ("ch2", 5.0, 10.0)]
    hypothesis = [("A", 0.0, 6.0), ("B", 6.0, 10.0)]
    score = der(reference, hypothesis)
    # [5, 6]: reference says ch2, hypothesis says A (mapped to ch1) -> one second of confusion
    assert score.counts.confusion_s == approx(1.0)
    assert score.counts.missed_s == 0.0
    assert score.counts.false_alarm_s == 0.0
    assert score.rate == approx(0.1)
    assert score.assignment == (("ch1", "A"), ("ch2", "B"))


def test_der_overlap_counts_once_per_speaker():
    reference = [("ch1", 0.0, 10.0), ("ch2", 0.0, 10.0)]
    both = der(reference, [("A", 0.0, 10.0), ("B", 0.0, 10.0)])
    assert both.counts.reference_s == approx(20.0)
    assert both.rate == 0.0

    one = der(reference, [("A", 0.0, 10.0)])
    assert one.counts.reference_s == approx(20.0)
    assert one.counts.missed_s == approx(10.0)
    assert one.counts.confusion_s == 0.0
    assert one.rate == approx(0.5)


def test_der_partial_overlap_hand_computed():
    # ch1 0-6, ch2 4-10 (2 s of overlap, 12 s reference). Hypothesis: A 0-5, B 5-10 (no overlap).
    # [0,4]: ch1/A ok. [4,5]: ref {ch1,ch2}, hyp {A}: one missed, A->ch1 ok. [5,6]: ref {ch1,ch2}, hyp {B}:
    # one missed, B->ch2 ok. [6,10]: ch2/B ok. Total missed 2 s, DER = 2/12.
    score = der([("ch1", 0.0, 6.0), ("ch2", 4.0, 10.0)], [("A", 0.0, 5.0), ("B", 5.0, 10.0)])
    assert score.counts.reference_s == approx(12.0)
    assert score.counts.missed_s == approx(2.0)
    assert score.counts.confusion_s == approx(0.0)
    assert score.counts.false_alarm_s == 0.0
    assert score.rate == approx(2 / 12)


def test_der_collar_removes_boundary_errors_and_reference_time():
    reference = [("ch1", 0.0, 10.0)]
    hypothesis = [("A", 0.1, 9.9)]
    plain = der(reference, hypothesis)
    assert plain.counts.missed_s == approx(0.2)
    assert plain.rate == approx(0.02)

    collared = der(reference, hypothesis, collar_s=0.25)
    assert collared.collar_s == 0.25
    assert collared.counts.missed_s == approx(0.0)
    assert collared.counts.reference_s == approx(9.5)  # 10 s minus two 0.25 s half-collars inside the region
    assert collared.rate == 0.0
    assert collared.to_dict()["collar_s"] == 0.25


def test_der_collar_only_follows_reference_boundaries():
    # A false alarm far away from every reference boundary survives the collar; one near a boundary does not.
    reference = [("ch1", 0.0, 10.0)]
    hypothesis = [("A", 0.0, 10.0), ("A", 20.0, 21.0), ("A", 10.0, 10.2)]
    score = der(reference, hypothesis, collar_s=0.25)
    assert score.counts.false_alarm_s == approx(1.0)
    assert score.counts.reference_s == approx(9.5)  # collars at 0 and 10 each remove 0.25 s of reference
    # a hypothesis boundary alone does not create a collar: a 0.3 s false alarm starting at 5 s is fully scored
    inside = der(reference, [("A", 0.0, 10.0), ("B", 5.0, 5.3)], collar_s=0.25)
    assert inside.counts.false_alarm_s == approx(0.3)


def test_der_collar_validation():
    with pytest.raises(ValueError, match="collar_s"):
        der([("ch1", 0.0, 1.0)], [], collar_s=-0.1)
    with pytest.raises(ValueError, match="collar_s"):
        der([("ch1", 0.0, 1.0)], [], collar_s=math.nan)


def test_der_clips_both_sides_to_duration():
    score = der([("ch1", 0.0, 12.0)], [("A", 0.0, 15.0)], duration_s=10.0)
    assert score.counts.reference_s == approx(10.0)
    assert score.rate == 0.0
    assert score.reference == (Segment("ch1", 0.0, 10.0),)
    assert score.hypothesis == (Segment("A", 0.0, 10.0),)


def test_der_is_invariant_to_label_permutation():
    reference = [("ch1", 0.0, 5.0), ("ch2", 4.0, 10.0)]
    hypothesis = [("A", 0.0, 4.5), ("B", 4.5, 9.0), ("C", 9.0, 11.0)]
    base = der(reference, hypothesis)
    renamed = [(f"spk{ord(label)}", start, end) for label, start, end in hypothesis]
    other = der(reference, renamed)
    assert other.counts == base.counts
    assert [ref for ref, _ in other.assignment] == [ref for ref, _ in base.assignment]
    # reference relabelling does not change the counts either
    flipped = der([("ch2", 0.0, 5.0), ("ch1", 4.0, 10.0)], hypothesis)
    assert flipped.counts == base.counts


def test_der_exact_intervals_without_frame_quantization():
    reference = [("ch1", 0.3333, 7.7777), ("ch2", 7.7777, 12.3456)]
    assert der(reference, [("x", 0.3333, 7.7777), ("y", 7.7777, 12.3456)]).rate == 0.0
    shifted = der(reference, [("x", 0.3333, 7.7777), ("y", 7.7777, 12.3457)])
    assert shifted.counts.false_alarm_s == approx(0.0001)


def test_der_empty_reference_and_empty_hypothesis():
    only_hyp = der([], [("A", 0.0, 3.0)])
    assert only_hyp.rate is None
    assert only_hyp.counts.false_alarm_s == approx(3.0)
    assert only_hyp.assignment == ()
    assert percent(only_hyp.rate) == "n/a"

    only_ref = der([("ch1", 0.0, 3.0)], [])
    assert only_ref.counts.missed_s == approx(3.0)
    assert only_ref.rate == approx(1.0)
    assert only_ref.assignment == ()

    nothing = der([], [])
    assert nothing.rate is None and nothing.counts.errors_s == 0.0


def test_der_extra_hypothesis_speaker_is_false_alarm_and_not_assigned():
    score = der([("ch1", 0.0, 10.0)], [("A", 0.0, 10.0), ("B", 0.0, 10.0)])
    assert score.counts.false_alarm_s == approx(10.0)
    assert score.rate == approx(1.0)
    assert score.assignment == (("ch1", "A"),)


def test_der_unions_self_overlapping_hypothesis_segments():
    reference = [("ch1", 0.0, 10.0)]
    assert der(reference, [("A", 0.0, 5.0), ("A", 3.0, 10.0)]).rate == 0.0
    assert der(reference, [("A", 0.0, 5.0), ("A", 5.0, 10.0)]).rate == 0.0


def test_der_to_dict_shape():
    payload = der([("ch1", 0.0, 10.0)], [("A", 0.0, 9.0)], collar_s=0.0).to_dict()
    assert set(payload) == {
        "rate",
        "missed_s",
        "false_alarm_s",
        "confusion_s",
        "reference_s",
        "errors_s",
        "collar_s",
        "assignment",
    }
    assert payload["assignment"] == [["ch1", "A"]]
    assert payload["rate"] == approx(0.1)
    assert payload["missed_s"] == 1.0


# ---------------------------------------------------------------------------
# DerCounts


def test_der_counts_addition_and_pooling():
    first = DerCounts(missed_s=1.0, false_alarm_s=0.5, confusion_s=0.5, reference_s=10.0)  # 20 %
    second = DerCounts(missed_s=0.0, false_alarm_s=0.0, confusion_s=2.0, reference_s=30.0)  # 6.67 %
    pooled = first + second
    assert pooled == DerCounts(missed_s=1.0, false_alarm_s=0.5, confusion_s=2.5, reference_s=40.0)
    assert pooled.rate == approx(4.0 / 40.0)  # pooled, not the mean of 0.2 and 0.0667
    assert first.rate == approx(0.2)

    total = DerCounts()
    total += first
    total += second
    assert total == pooled
    assert sum([first, second], DerCounts()) == pooled


def test_der_counts_rate_none_and_to_dict_rounding():
    assert DerCounts().rate is None
    counts = DerCounts(missed_s=1.23456, false_alarm_s=0.0004, confusion_s=0.0, reference_s=3.0)
    payload = counts.to_dict()
    assert payload == {
        "rate": counts.rate,
        "missed_s": 1.235,
        "false_alarm_s": 0.0,
        "confusion_s": 0.0,
        "reference_s": 3.0,
        "errors_s": 1.235,
    }
    assert payload["rate"] == approx(1.23496 / 3.0)
    assert DerCounts().to_dict()["rate"] is None
    with pytest.raises(TypeError):
        DerCounts() + 1  # type: ignore[operator]


# ---------------------------------------------------------------------------
# JER


def test_jer_perfect_match():
    reference = [("ch1", 0.0, 5.0), ("ch2", 5.0, 10.0)]
    score = jer(reference, [("B", 0.0, 5.0), ("A", 5.0, 10.0)])
    assert score.rate == 0.0
    assert score.error_sum == 0.0
    assert score.reference_speakers == 2
    assert score.per_speaker == {"ch1": 0.0, "ch2": 0.0}
    assert score.assignment == (("ch1", "B"), ("ch2", "A"))


def test_jer_half_covered_speaker():
    score = jer([("ch1", 0.0, 10.0)], [("A", 0.0, 5.0)])
    assert score.per_speaker["ch1"] == approx(0.5)
    assert score.rate == approx(0.5)
    # over-long hypothesis: intersection 10, union 20
    assert jer([("ch1", 0.0, 10.0)], [("A", 0.0, 20.0)]).rate == approx(0.5)


def test_jer_unmatched_reference_speaker_scores_one():
    score = jer([("ch1", 0.0, 10.0), ("ch2", 10.0, 20.0)], [("A", 0.0, 10.0)])
    assert score.per_speaker == {"ch1": 0.0, "ch2": 1.0}
    assert score.error_sum == approx(1.0)
    assert score.rate == approx(0.5)
    assert score.assignment == (("ch1", "A"),)


def test_jer_minimizes_total_jaccard_error():
    # A covers ch1 fully and a bit of ch2; B covers ch2 only partially. The best pairing is ch1-A, ch2-B.
    reference = [("ch1", 0.0, 10.0), ("ch2", 10.0, 20.0)]
    hypothesis = [("A", 0.0, 12.0), ("B", 12.0, 20.0)]
    score = jer(reference, hypothesis)
    assert score.assignment == (("ch1", "A"), ("ch2", "B"))
    assert score.per_speaker["ch1"] == approx(1 - 10 / 12)
    assert score.per_speaker["ch2"] == approx(1 - 8 / 10)
    assert score.rate == approx(((1 - 10 / 12) + (1 - 8 / 10)) / 2)


def test_jer_edge_cases_and_to_dict():
    none = jer([], [("A", 0.0, 1.0)])
    assert none.rate is None and none.reference_speakers == 0 and none.per_speaker == {}
    missing = jer([("ch1", 0.0, 1.0)], [])
    assert missing.rate == 1.0 and missing.assignment == ()
    payload = jer([("ch1", 0.0, 10.0)], [("A", 0.0, 5.0)], duration_s=8.0).to_dict()
    assert set(payload) == {"rate", "error_sum", "reference_speakers", "per_speaker", "assignment"}
    assert payload["rate"] == approx(1 - 5 / 8)  # reference clipped to 8 s
    assert payload["assignment"] == [["ch1", "A"]]


# ---------------------------------------------------------------------------
# Speaker count


def test_speaker_count():
    reference = [("ch1", 0.0, 5.0), ("ch2", 5.0, 10.0)]
    same = speaker_count(reference, [("x", 0.0, 1.0), ("y", 1.0, 2.0)])
    assert (same.reference, same.hypothesis, same.correct, same.absolute_error) == (2, 2, True, 0)
    more = speaker_count(reference, [("x", 0.0, 1.0), ("y", 1.0, 2.0), ("z", 2.0, 3.0)])
    assert more.to_dict() == {"reference": 2, "hypothesis": 3, "correct": False, "absolute_error": 1}
    fewer = speaker_count(reference, [("x", 0.0, 10.0)])
    assert (fewer.correct, fewer.absolute_error) == (False, 1)
    # labels without speech time (zero length or beyond the duration) do not count
    ghost = speaker_count(reference, [("x", 0.0, 10.0), ("y", 3.0, 3.0), ("z", 20.0, 25.0)], duration_s=10.0)
    assert ghost.hypothesis == 1


# ---------------------------------------------------------------------------
# Overlap ratio / tiers / speaking time


def test_overlap_ratio():
    assert overlap_ratio([("ch1", 0.0, 10.0), ("ch2", 5.0, 15.0)]) == approx(5 / 15)
    assert overlap_ratio([("ch1", 0.0, 10.0), ("ch2", 10.0, 20.0)]) == 0.0
    assert overlap_ratio([("ch1", 0.0, 10.0), ("ch2", 0.0, 10.0)]) == approx(1.0)
    assert overlap_ratio([]) == 0.0
    # a speaker overlapping with themselves is not overlap
    assert overlap_ratio([("ch1", 0.0, 10.0), ("ch1", 5.0, 15.0)]) == 0.0
    # three speakers at once still count the time once
    assert overlap_ratio([("a", 0.0, 10.0), ("b", 0.0, 10.0), ("c", 0.0, 10.0), ("a", 10.0, 20.0)]) == approx(0.5)


def test_overlap_tier():
    assert overlap_tier(0.0) == "low"
    assert overlap_tier(0.0999) == "low"
    assert overlap_tier(0.10) == "medium"
    assert overlap_tier(0.1999) == "medium"
    assert overlap_tier(0.20) == "high"
    assert overlap_tier(1.0) == "high"
    with pytest.raises(ValueError):
        overlap_tier(-0.1)
    with pytest.raises(ValueError):
        overlap_tier(math.nan)


def test_speaking_time():
    times = speaking_time([("ch2", 0.0, 4.0), ("ch1", 0.0, 10.0), ("ch1", 5.0, 15.0), ("ch2", 20.0, 21.5)])
    assert times == {"ch1": approx(15.0), "ch2": approx(5.5)}
    assert list(times) == ["ch1", "ch2"]
    assert speaking_time([]) == {}


# ---------------------------------------------------------------------------
# RTTM


def test_write_rttm_format():
    text = write_rttm([("ch2", 1.5, 2.0), ("ch1", 0.0, 1.5)], "conv-1")
    assert text == (
        "SPEAKER conv-1 1 0.000 1.500 <NA> <NA> ch1 <NA> <NA>\nSPEAKER conv-1 1 1.500 0.500 <NA> <NA> ch2 <NA> <NA>\n"
    )
    assert write_rttm([], "conv-1") == ""
    assert write_rttm([("Speaker 1", 0.0, 1.0)], "my file").startswith(
        "SPEAKER my_file 1 0.000 1.000 <NA> <NA> Speaker_1"
    )
    with pytest.raises(ValueError, match="file id"):
        write_rttm([("a", 0.0, 1.0)], "   ")


def test_rttm_round_trip_and_parsing():
    segments = to_segments([("ch1", 0.0, 1.5), ("ch2", 1.25, 3.0), ("ch1", 4.0, 4.125)])
    parsed = read_rttm(write_rttm(segments, "conv"))
    assert len(parsed) == 3
    for got, want in zip(
        sorted(parsed, key=lambda s: s.start_s), sorted(segments, key=lambda s: s.start_s), strict=True
    ):
        assert got.speaker == want.speaker
        assert got.start_s == approx(want.start_s, abs=1e-9)
        assert got.end_s == approx(want.end_s, abs=1e-9)
    # scoring the parsed RTTM against the original is a perfect match
    assert der(segments, parsed).rate == approx(0.0, abs=1e-9)

    text = ";; a comment\n\nSPEAKER f 1 0.000 1.000 <NA> <NA> A <NA> <NA>\nLEXEME f 1 0.0 0.5 hello lex A 0.9 <NA>\n"
    assert read_rttm(text) == (Segment("A", 0.0, 1.0),)
    assert read_rttm("") == ()
    with pytest.raises(ValueError, match="line 1"):
        read_rttm("SPEAKER f 1 0.000 x <NA> <NA> A <NA> <NA>")
    with pytest.raises(ValueError, match="at least 8 fields"):
        read_rttm("SPEAKER f 1 0.000 1.000")
    with pytest.raises(ValueError, match="line 2"):
        read_rttm("SPEAKER f 1 0.0 1.0 <NA> <NA> A <NA> <NA>\nSPEAKER f 1 -1.0 1.0 <NA> <NA> A <NA> <NA>")


def test_percent():
    assert percent(0.12345) == "12.35%"
    assert percent(0.0) == "0.00%"
    assert percent(1.0) == "100.00%"
    assert percent(None) == "n/a"
