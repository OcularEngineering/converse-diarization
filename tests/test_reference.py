from __future__ import annotations

import math

import pytest

from converse_diarization.dataset import Conversation, Track, Word
from converse_diarization.metrics import Segment, der, overlap_ratio
from converse_diarization.reference import hypothesis_timeline, reference_timeline

approx = pytest.approx


def make_track(
    track_id: str, words: list[tuple[str, float, float]], *, duration_s: float, offset_s: float = 0.0
) -> Track:
    return Track(
        id=track_id,
        speaker_id=f"speaker-{track_id}",
        path=f"audio/conv/{track_id}.flac",
        sha256="0" * 64,
        duration_s=duration_s,
        sample_rate=48000,
        offset_s=offset_s,
        capture_ms=None,
        source_file=f"{track_id}.flac",
        reference=" ".join(text for text, _, _ in words),
        words=tuple(Word(text, start, end) for text, start, end in words),
    )


def make_conversation(tracks: tuple[Track, ...]) -> Conversation:
    duration = max(track.offset_s + track.duration_s for track in tracks)
    return Conversation(id="conv", language="en", locale="en-US", duration_s=duration, tracks=tracks)


def test_reference_timeline_merges_words_per_track_and_labels_by_track_id():
    ch1 = make_track(
        "ch1",
        [("hello", 0.0, 0.5), ("there", 0.6, 1.0), ("again", 1.5, 2.0)],  # 0.1 s gap merges, 0.5 s gap splits
        duration_s=10.0,
    )
    ch2 = make_track("ch2", [("yes", 0.8, 1.2), ("right", 1.4, 1.9)], duration_s=9.5, offset_s=0.5)
    timeline = reference_timeline(make_conversation((ch1, ch2)))
    assert timeline == (
        Segment("ch1", 0.0, 1.0),
        Segment("ch1", 1.5, 2.0),
        Segment("ch2", 0.8, 1.9),
    )
    # overlap: [0.8, 1.0] and [1.5, 1.9] = 0.6 s out of 2.0 s of speech
    assert overlap_ratio(timeline) == approx(0.6 / 2.0)


def test_reference_timeline_gap_boundary_is_exactly_0_2():
    merges = make_track("ch1", [("a", 0.0, 1.0), ("b", 1.2, 2.0)], duration_s=5.0)
    assert reference_timeline(make_conversation((merges,))) == (Segment("ch1", 0.0, 2.0),)
    splits = make_track("ch1", [("a", 0.0, 1.0), ("b", 1.201, 2.0)], duration_s=5.0)
    assert reference_timeline(make_conversation((splits,))) == (Segment("ch1", 0.0, 1.0), Segment("ch1", 1.201, 2.0))


def test_reference_timeline_clips_to_conversation_duration():
    ch1 = make_track("ch1", [("late", 9.0, 12.0), ("gone", 13.0, 14.0)], duration_s=10.0)
    ch2 = make_track("ch2", [("early", 0.0, 1.0)], duration_s=5.0)
    conversation = make_conversation((ch1, ch2))
    assert conversation.duration_s == 10.0
    assert reference_timeline(conversation) == (Segment("ch1", 9.0, 10.0), Segment("ch2", 0.0, 1.0))


def test_reference_timeline_skips_defective_words():
    track = Track(
        id="ch1",
        speaker_id="s",
        path="audio/conv/ch1.flac",
        sha256="0" * 64,
        duration_s=10.0,
        sample_rate=16000,
        offset_s=0.0,
        capture_ms=None,
        source_file="ch1.flac",
        reference="",
        words=(
            Word("ok", 0.0, 0.5),
            Word("", 0.6, 0.9),  # empty text
            Word("   ", 0.6, 0.9),  # whitespace only
            Word("backwards", 3.0, 2.0),  # end before start
            Word("nan", 4.0, math.nan),  # not finite
            Word("inf", math.inf, 5.0),
            Word("edge", -0.001, 0.2),  # tiny negative start is clamped to 0
            Word("fine", 6.0, 6.5),
        ),
    )
    conversation = Conversation(id="conv", language="en", locale=None, duration_s=10.0, tracks=(track,))
    assert reference_timeline(conversation) == (Segment("ch1", 0.0, 0.5), Segment("ch1", 6.0, 6.5))


def test_reference_timeline_empty_tracks():
    track = make_track("ch1", [], duration_s=3.0)
    assert reference_timeline(make_conversation((track,))) == ()


def test_hypothesis_timeline_word_granularity_merges_with_gap_rule():
    words = [
        {"speaker": "A", "start_s": 0.0, "end_s": 0.4, "text": "hi"},
        {"speaker": "A", "start_s": 0.6, "end_s": 1.0, "text": "there"},
        {"speaker": "A", "start_s": 1.3, "end_s": 1.8, "text": "friend"},  # 0.3 s gap -> new region
        {"speaker": "B", "start_s": 0.9, "end_s": 1.2, "text": "yo"},
    ]
    assert hypothesis_timeline(words, "word", 10.0) == (
        Segment("A", 0.0, 1.0),
        Segment("A", 1.3, 1.8),
        Segment("B", 0.9, 1.2),
    )


def test_hypothesis_timeline_segment_granularity_keeps_gaps():
    segments = [
        {"speaker": "A", "start_s": 0.0, "end_s": 0.4, "text": None},
        {"speaker": "A", "start_s": 0.6, "end_s": 1.0, "text": None},  # 0.2 s gap is NOT bridged for turns
        {"speaker": "A", "start_s": 1.0, "end_s": 1.5, "text": None},  # touching intervals are unioned
        {"speaker": "A", "start_s": 1.2, "end_s": 2.0, "text": None},  # overlapping intervals are unioned
        {"speaker": "B", "start_s": 9.0, "end_s": 12.0, "text": None},  # clipped to the duration
        {"speaker": "C", "start_s": 11.0, "end_s": 12.0, "text": None},  # entirely beyond the audio: dropped
    ]
    assert hypothesis_timeline(segments, "segment", 10.0) == (
        Segment("A", 0.0, 0.4),
        Segment("A", 0.6, 2.0),
        Segment("B", 9.0, 10.0),
    )
    # the same items at word granularity bridge the 0.2 s gap
    assert hypothesis_timeline(segments, "word", 10.0)[0] == Segment("A", 0.0, 2.0)


def test_hypothesis_timeline_rejects_unknown_granularity_and_bad_items():
    with pytest.raises(ValueError, match="granularity"):
        hypothesis_timeline([], "turns", 10.0)
    with pytest.raises(ValueError, match="missing key"):
        hypothesis_timeline([{"speaker": "A", "start_s": 0.0}], "segment", 10.0)
    assert hypothesis_timeline([], "segment", 10.0) == ()
    assert hypothesis_timeline([], "word", 10.0) == ()


def test_reference_and_hypothesis_timelines_score_together():
    ch1 = make_track("ch1", [("a", 0.0, 1.0), ("b", 1.1, 2.0)], duration_s=10.0)
    ch2 = make_track("ch2", [("c", 3.0, 4.0)], duration_s=10.0)
    reference = reference_timeline(make_conversation((ch1, ch2)))
    hypothesis = hypothesis_timeline(
        [
            {"speaker": "SPEAKER_01", "start_s": 0.0, "end_s": 2.0, "text": None},
            {"speaker": "SPEAKER_00", "start_s": 3.0, "end_s": 4.5, "text": None},
        ],
        "segment",
        10.0,
    )
    score = der(reference, hypothesis)
    assert score.counts.reference_s == approx(3.0)
    assert score.counts.false_alarm_s == approx(0.5)
    assert score.assignment == (("ch1", "SPEAKER_01"), ("ch2", "SPEAKER_00"))
