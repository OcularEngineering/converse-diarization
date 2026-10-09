"""Build the reference and hypothesis timelines that the scorer compares.

The reference comes from the per-speaker microphone tracks of a conversation: every
track carries word timings on the shared conversation timeline, so one speaker's
speech regions are simply their words merged with the :data:`MERGE_GAP_S` rule
(consecutive words closer than 0.2 s belong to the same region, longer pauses are
silence). The label of a region is the track id (``"ch1"``, ``"ch2"``).

Hypotheses are brought to the same shape. Models that return speaker-labelled words
get the identical 0.2 s merge; models that return their own speaker turns are kept
as-is apart from clipping and per-speaker union.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from converse_diarization.dataset import Conversation
from converse_diarization.metrics import MERGE_GAP_S, Segment, canonical, clip_segments, merge_segments, to_segments

_GRANULARITIES = ("segment", "word")


def reference_timeline(conversation: Conversation) -> tuple[Segment, ...]:
    """Who-spoke-when of a conversation from its tracks' word timings.

    Per track, words are merged into regions with the 0.2 s gap rule and labelled with the track id;
    regions are clipped to ``conversation.duration_s``. Words with empty text, non-finite or reversed
    timings are ignored; a start slightly before 0 is clamped to 0.
    """
    words: list[Segment] = []
    for track in conversation.tracks:
        for word in track.words:
            start, end = _timing(word.start_s), _timing(word.end_s)
            if start is None or end is None or not str(word.text).strip():
                continue
            if end < start or end < 0:
                continue
            words.append(Segment(track.id, max(0.0, start), end))
    merged = merge_segments(words, gap_s=MERGE_GAP_S)
    return clip_segments(merged, conversation.duration_s)


def hypothesis_timeline(segments: Iterable[dict[str, Any]], granularity: str, duration_s: float) -> tuple[Segment, ...]:
    """Normalize a provider's segments for scoring.

    ``granularity="word"``: the items are speaker-labelled words and get the same 0.2 s merge as the
    reference. ``granularity="segment"``: the items are speaker turns and are only clipped to
    ``duration_s`` with touching/overlapping intervals of one speaker unioned.
    """
    if granularity not in _GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}; expected one of {', '.join(_GRANULARITIES)}")
    items = to_segments(segments)
    if granularity == "word":
        items = merge_segments(items, gap_s=MERGE_GAP_S)
    return canonical(items, duration_s)


def _timing(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = ["hypothesis_timeline", "reference_timeline"]
