"""Scoring for Converse-Diarization: DER, JER, speaker count and overlap statistics.

Everything here works on exact time intervals. There is no frame grid: boundaries are
kept as floats and the scorer walks the timeline from one boundary event to the next
(a sweep line), so a 12.345 s boundary is scored at 12.345 s and not at the nearest
10 ms frame.

Terminology used throughout:

reference
    Who actually spoke when. Built from each speaker's own microphone track, so the
    labels are the track ids (``"ch1"``, ``"ch2"``).
hypothesis
    What the model said. Labels are the model's own (``"SPEAKER_00"``, ``"A"`` ...);
    the scorer finds the best one-to-one mapping between the two label sets.
canonical form
    Before scoring, both sides are clipped to the conversation duration and, per
    speaker, touching or overlapping intervals are unioned. After that every speaker
    has a set of disjoint intervals and "number of active speakers" is well defined.

Diarization error rate (DER)
    Walk the timeline. Inside each elementary interval the set of active reference
    speakers ``R`` and the set of active hypothesis speakers ``H`` are constant, so for
    its duration ``d``:

    * reference time grows by ``d * |R|`` (overlapping speech counts once per speaker),
    * missed speech grows by ``d * max(0, |R| - |H|)``,
    * false alarm grows by ``d * max(0, |H| - |R|)``,
    * ``common = d * min(|R|, |H|)`` is the time where *some* speaker could have been
      attributed correctly, and the pair matrix ``overlap[r][h]`` grows by ``d`` for every
      ``r in R, h in H``.

    A one-to-one assignment between reference and hypothesis labels that maximizes the
    summed ``overlap`` (Hungarian algorithm) gives the correctly attributed time; the
    rest of ``common`` is speaker confusion. ``DER = (missed + false alarm + confusion) /
    reference time``. Overlapping speech is always included.

    A collar of ``c`` seconds turns ``[b - c, b + c]`` around every reference boundary
    ``b`` into a no-score zone: nothing is accumulated there, neither errors nor
    reference time.

Jaccard error rate (JER)
    For a reference speaker ``r`` paired with hypothesis speaker ``h`` the error is
    ``1 - |r ∩ h| / |r ∪ h|``. Pairs are chosen one-to-one to minimize the total error;
    a reference speaker without a partner scores ``1.0``. JER is the mean over reference
    speakers.

Only ``numpy`` and ``scipy`` are needed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

MERGE_GAP_S = 0.2
"""Consecutive words of one speaker closer than this are one speech region."""

DEFAULT_COLLAR_S = 0.25
"""Secondary DER collar: half-width of the no-score zone around reference boundaries."""

_GAP_EPSILON_S = 1e-9
"""Tolerance so that ``1.1 - 0.9`` (which is ``0.2 + 2e-16`` in floating point) still counts as a 0.2 s gap."""

_RTTM_NA = "<NA>"


# ---------------------------------------------------------------------------
# Segments


@dataclass(frozen=True)
class Segment:
    """One labelled speech interval ``[start_s, end_s]`` in seconds on the conversation timeline."""

    speaker: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.speaker, str) or not self.speaker:
            raise ValueError(f"segment speaker must be a non-empty string, got {self.speaker!r}")
        start = _finite("start_s", self.start_s)
        end = _finite("end_s", self.end_s)
        if start < 0:
            raise ValueError(f"segment start_s must be >= 0, got {start!r} for speaker {self.speaker!r}")
        if end < start:
            raise ValueError(f"segment end_s ({end!r}) is before start_s ({start!r}) for speaker {self.speaker!r}")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "start_s": self.start_s, "end_s": self.end_s}


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number!r}")
    return number


def to_segments(items: Iterable[Any]) -> tuple[Segment, ...]:
    """Normalize an iterable of segment-like items.

    Accepted item shapes: :class:`Segment`; ``(speaker, start_s, end_s)`` tuples/lists; mappings with
    ``speaker``, ``start_s`` and ``end_s`` keys (extra keys such as ``text`` are ignored).
    """
    if items is None:
        return ()
    if isinstance(items, (str, bytes, Segment, Mapping)):
        raise ValueError(f"expected an iterable of segments, got {type(items).__name__}")
    segments: list[Segment] = []
    for index, item in enumerate(items):
        if isinstance(item, Segment):
            segments.append(item)
        elif isinstance(item, Mapping):
            missing = [key for key in ("speaker", "start_s", "end_s") if key not in item]
            if missing:
                raise ValueError(f"segment {index} is missing key(s): {', '.join(missing)}")
            segments.append(Segment(str(item["speaker"]), item["start_s"], item["end_s"]))
        elif isinstance(item, (tuple, list)):
            if len(item) != 3:
                raise ValueError(f"segment {index} must be (speaker, start_s, end_s), got {len(item)} values")
            speaker, start, end = item
            segments.append(Segment(str(speaker), start, end))
        else:
            raise ValueError(f"segment {index} has unsupported type {type(item).__name__}")
    return tuple(segments)


def _by_speaker(segments: Iterable[Segment]) -> dict[str, list[Segment]]:
    grouped: dict[str, list[Segment]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker, []).append(segment)
    for items in grouped.values():
        items.sort(key=lambda segment: (segment.start_s, segment.end_s))
    return grouped


def merge_segments(segments: Iterable[Any], *, gap_s: float = MERGE_GAP_S) -> tuple[Segment, ...]:
    """Per speaker, merge consecutive intervals whose silence gap is ``<= gap_s`` into one interval.

    With ``gap_s=0`` this is a plain union of touching/overlapping intervals. The result is sorted by
    speaker label, then start time. Intervals of different speakers are never merged.
    """
    gap = _finite("gap_s", gap_s)
    if gap < 0:
        raise ValueError(f"gap_s must be >= 0, got {gap!r}")
    merged: list[Segment] = []
    for speaker in sorted(grouped := _by_speaker(to_segments(segments))):
        current_start: float | None = None
        current_end = 0.0
        for segment in grouped[speaker]:
            if current_start is None:
                current_start, current_end = segment.start_s, segment.end_s
            elif segment.start_s - current_end <= gap + _GAP_EPSILON_S:
                current_end = max(current_end, segment.end_s)
            else:
                merged.append(Segment(speaker, current_start, current_end))
                current_start, current_end = segment.start_s, segment.end_s
        if current_start is not None:
            merged.append(Segment(speaker, current_start, current_end))
    return tuple(merged)


def clip_segments(segments: Iterable[Any], duration_s: float | None) -> tuple[Segment, ...]:
    """Clip every interval to ``[0, duration_s]`` and drop the ones left with no duration.

    ``duration_s=None`` means "no clipping" (zero-length intervals are still dropped).
    """
    limit: float | None = None
    if duration_s is not None:
        limit = _finite("duration_s", duration_s)
        if limit <= 0:
            raise ValueError(f"duration_s must be > 0, got {limit!r}")
    clipped: list[Segment] = []
    for segment in to_segments(segments):
        start, end = segment.start_s, segment.end_s
        if limit is not None:
            start, end = min(start, limit), min(end, limit)
        if end > start:
            clipped.append(
                segment if (start, end) == (segment.start_s, segment.end_s) else Segment(segment.speaker, start, end)
            )
    return tuple(clipped)


def canonical(segments: Iterable[Any], duration_s: float | None = None) -> tuple[Segment, ...]:
    """Clip to the conversation duration and union touching/overlapping intervals per speaker.

    After this every speaker owns disjoint, non-empty intervals and the output is sorted by speaker then start.
    """
    return merge_segments(clip_segments(segments, duration_s), gap_s=0.0)


def _speakers(segments: Iterable[Segment]) -> list[str]:
    return sorted({segment.speaker for segment in segments})


# ---------------------------------------------------------------------------
# Sweep line


@dataclass
class _Tally:
    reference_s: float = 0.0
    missed_s: float = 0.0
    false_alarm_s: float = 0.0
    common_s: float = 0.0
    ref_time: dict[str, float] = field(default_factory=dict)
    hyp_time: dict[str, float] = field(default_factory=dict)
    overlap: dict[tuple[str, str], float] = field(default_factory=dict)

    def add(self, duration: float, ref_active: list[str], hyp_active: list[str]) -> None:
        n_ref, n_hyp = len(ref_active), len(hyp_active)
        self.reference_s += duration * n_ref
        self.missed_s += duration * max(0, n_ref - n_hyp)
        self.false_alarm_s += duration * max(0, n_hyp - n_ref)
        self.common_s += duration * min(n_ref, n_hyp)
        for ref in ref_active:
            self.ref_time[ref] = self.ref_time.get(ref, 0.0) + duration
            for hyp in hyp_active:
                self.overlap[(ref, hyp)] = self.overlap.get((ref, hyp), 0.0) + duration
        for hyp in hyp_active:
            self.hyp_time[hyp] = self.hyp_time.get(hyp, 0.0) + duration


_REF, _HYP, _EXCLUDE = 0, 1, 2


def _sweep(
    reference: Iterable[Segment],
    hypothesis: Iterable[Segment],
    exclusions: Iterable[tuple[float, float]] = (),
) -> _Tally:
    """Accumulate time statistics over every elementary interval of the timeline.

    Events are the starts and ends of all reference intervals, hypothesis intervals and exclusion
    (collar) zones. Between two consecutive distinct event times the active sets are constant, so
    the interval is scored in one go; intervals inside an exclusion zone are skipped entirely.
    """
    events: list[tuple[float, int, int, str]] = []
    for segment in reference:
        events.append((segment.start_s, _REF, +1, segment.speaker))
        events.append((segment.end_s, _REF, -1, segment.speaker))
    for segment in hypothesis:
        events.append((segment.start_s, _HYP, +1, segment.speaker))
        events.append((segment.end_s, _HYP, -1, segment.speaker))
    for low, high in exclusions:
        events.append((low, _EXCLUDE, +1, ""))
        events.append((high, _EXCLUDE, -1, ""))
    events.sort(key=lambda event: event[0])

    tally = _Tally()
    active_ref: dict[str, int] = {}
    active_hyp: dict[str, int] = {}
    excluded = 0
    previous: float | None = None
    index, count = 0, len(events)
    while index < count:
        time = events[index][0]
        if previous is not None and time > previous and excluded == 0:
            tally.add(
                time - previous,
                [label for label, depth in active_ref.items() if depth > 0],
                [label for label, depth in active_hyp.items() if depth > 0],
            )
        while index < count and events[index][0] == time:
            _, kind, delta, label = events[index]
            if kind == _REF:
                active_ref[label] = active_ref.get(label, 0) + delta
            elif kind == _HYP:
                active_hyp[label] = active_hyp.get(label, 0) + delta
            else:
                excluded += delta
            index += 1
        previous = time
    return tally


def _collar_zones(reference: Iterable[Segment], collar_s: float) -> tuple[tuple[float, float], ...]:
    """Union of ``[b - collar_s, b + collar_s]`` over every reference boundary ``b``."""
    if collar_s <= 0:
        return ()
    zones: list[list[float]] = []
    for boundary in sorted({point for segment in reference for point in (segment.start_s, segment.end_s)}):
        low, high = boundary - collar_s, boundary + collar_s
        if zones and low <= zones[-1][1]:
            zones[-1][1] = max(zones[-1][1], high)
        else:
            zones.append([low, high])
    return tuple((low, high) for low, high in zones)


def _pair_matrix(tally: _Tally, ref_labels: list[str], hyp_labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(ref_labels), len(hyp_labels)), dtype=float)
    for i, ref in enumerate(ref_labels):
        for j, hyp in enumerate(hyp_labels):
            matrix[i, j] = tally.overlap.get((ref, hyp), 0.0)
    return matrix


# ---------------------------------------------------------------------------
# DER


@dataclass
class DerCounts:
    """Error seconds behind a DER. Add instances together to pool several conversations."""

    missed_s: float = 0.0
    false_alarm_s: float = 0.0
    confusion_s: float = 0.0
    reference_s: float = 0.0

    @property
    def errors_s(self) -> float:
        return self.missed_s + self.false_alarm_s + self.confusion_s

    @property
    def rate(self) -> float | None:
        if self.reference_s <= 0:
            return None
        return self.errors_s / self.reference_s

    def __add__(self, other: DerCounts) -> DerCounts:
        if not isinstance(other, DerCounts):
            return NotImplemented
        return DerCounts(
            missed_s=self.missed_s + other.missed_s,
            false_alarm_s=self.false_alarm_s + other.false_alarm_s,
            confusion_s=self.confusion_s + other.confusion_s,
            reference_s=self.reference_s + other.reference_s,
        )

    def __iadd__(self, other: DerCounts) -> DerCounts:
        if not isinstance(other, DerCounts):
            return NotImplemented
        self.missed_s += other.missed_s
        self.false_alarm_s += other.false_alarm_s
        self.confusion_s += other.confusion_s
        self.reference_s += other.reference_s
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "missed_s": round(self.missed_s, 3),
            "false_alarm_s": round(self.false_alarm_s, 3),
            "confusion_s": round(self.confusion_s, 3),
            "reference_s": round(self.reference_s, 3),
            "errors_s": round(self.errors_s, 3),
        }


@dataclass(frozen=True)
class DerScore:
    """DER of one conversation together with the inputs it was computed from (canonical form)."""

    counts: DerCounts
    collar_s: float
    assignment: tuple[tuple[str, str], ...]
    reference: tuple[Segment, ...]
    hypothesis: tuple[Segment, ...]

    @property
    def rate(self) -> float | None:
        return self.counts.rate

    def to_dict(self) -> dict[str, Any]:
        return self.counts.to_dict() | {
            "collar_s": self.collar_s,
            "assignment": [[ref, hyp] for ref, hyp in self.assignment],
        }


def der(
    reference: Iterable[Any],
    hypothesis: Iterable[Any],
    *,
    collar_s: float = 0.0,
    duration_s: float | None = None,
) -> DerScore:
    """Diarization error rate with overlap included and an optional reference-boundary collar.

    Both inputs are brought to canonical form (clipped to ``duration_s`` when given). The returned
    ``assignment`` lists the ``(reference, hypothesis)`` label pairs that were credited as correct;
    pairs that share no time are left out.
    """
    collar = _finite("collar_s", collar_s)
    if collar < 0:
        raise ValueError(f"collar_s must be >= 0, got {collar!r}")
    ref = canonical(reference, duration_s)
    hyp = canonical(hypothesis, duration_s)
    tally = _sweep(ref, hyp, _collar_zones(ref, collar))

    ref_labels, hyp_labels = _speakers(ref), _speakers(hyp)
    assignment: list[tuple[str, str]] = []
    correct_s = 0.0
    if ref_labels and hyp_labels:
        matrix = _pair_matrix(tally, ref_labels, hyp_labels)
        rows, cols = linear_sum_assignment(matrix, maximize=True)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            if matrix[i, j] > 0:
                correct_s += float(matrix[i, j])
                assignment.append((ref_labels[i], hyp_labels[j]))
    counts = DerCounts(
        missed_s=tally.missed_s,
        false_alarm_s=tally.false_alarm_s,
        confusion_s=max(0.0, tally.common_s - correct_s),
        reference_s=tally.reference_s,
    )
    return DerScore(counts=counts, collar_s=collar, assignment=tuple(assignment), reference=ref, hypothesis=hyp)


# ---------------------------------------------------------------------------
# JER


@dataclass(frozen=True)
class JerScore:
    """Jaccard error rate of one conversation."""

    error_sum: float
    reference_speakers: int
    per_speaker: dict[str, float]
    assignment: tuple[tuple[str, str], ...]

    @property
    def rate(self) -> float | None:
        if self.reference_speakers <= 0:
            return None
        return self.error_sum / self.reference_speakers

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "error_sum": round(self.error_sum, 6),
            "reference_speakers": self.reference_speakers,
            "per_speaker": {speaker: round(error, 6) for speaker, error in self.per_speaker.items()},
            "assignment": [[ref, hyp] for ref, hyp in self.assignment],
        }


def jer(reference: Iterable[Any], hypothesis: Iterable[Any], *, duration_s: float | None = None) -> JerScore:
    """Jaccard error rate: mean over reference speakers of ``1 - |r ∩ h| / |r ∪ h|`` under the best pairing.

    The pairing is one-to-one and minimizes the summed Jaccard error. Reference speakers that get no
    partner (or whose partner shares no time with them) score ``1.0`` and are not listed in ``assignment``.
    """
    ref = canonical(reference, duration_s)
    hyp = canonical(hypothesis, duration_s)
    tally = _sweep(ref, hyp)

    ref_labels, hyp_labels = _speakers(ref), _speakers(hyp)
    per_speaker = {label: 1.0 for label in ref_labels}
    assignment: list[tuple[str, str]] = []
    if ref_labels and hyp_labels:
        errors = np.ones((len(ref_labels), len(hyp_labels)), dtype=float)
        for i, ref_label in enumerate(ref_labels):
            for j, hyp_label in enumerate(hyp_labels):
                intersection = tally.overlap.get((ref_label, hyp_label), 0.0)
                union = tally.ref_time.get(ref_label, 0.0) + tally.hyp_time.get(hyp_label, 0.0) - intersection
                if union > 0:
                    errors[i, j] = min(1.0, max(0.0, 1.0 - intersection / union))
        rows, cols = linear_sum_assignment(errors)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            if tally.overlap.get((ref_labels[i], hyp_labels[j]), 0.0) > 0:
                per_speaker[ref_labels[i]] = float(errors[i, j])
                assignment.append((ref_labels[i], hyp_labels[j]))
    return JerScore(
        error_sum=sum(per_speaker.values()),
        reference_speakers=len(ref_labels),
        per_speaker=per_speaker,
        assignment=tuple(assignment),
    )


# ---------------------------------------------------------------------------
# Speaker count, overlap, speaking time


@dataclass(frozen=True)
class SpeakerCountScore:
    reference: int
    hypothesis: int
    correct: bool
    absolute_error: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "correct": self.correct,
            "absolute_error": self.absolute_error,
        }


def speaker_count(
    reference: Iterable[Any], hypothesis: Iterable[Any], *, duration_s: float | None = None
) -> SpeakerCountScore:
    """Compare the number of distinct labels with speech time (after clipping) on both sides."""
    n_ref = len(_speakers(canonical(reference, duration_s)))
    n_hyp = len(_speakers(canonical(hypothesis, duration_s)))
    return SpeakerCountScore(
        reference=n_ref, hypothesis=n_hyp, correct=n_ref == n_hyp, absolute_error=abs(n_ref - n_hyp)
    )


def overlap_ratio(segments: Iterable[Any]) -> float:
    """Time where at least two distinct speakers are active, divided by time where at least one is.

    Returns ``0.0`` when there is no speech at all. Overlapping intervals of the *same* speaker are not overlap.
    """
    events: list[tuple[float, int]] = []
    for segment in canonical(segments):
        events.append((segment.start_s, +1))
        events.append((segment.end_s, -1))
    events.sort()
    speech_s = overlapped_s = 0.0
    depth = 0
    previous: float | None = None
    for time, delta in events:
        if previous is not None and time > previous:
            if depth >= 1:
                speech_s += time - previous
            if depth >= 2:
                overlapped_s += time - previous
        depth += delta
        previous = time
    if speech_s <= 0:
        return 0.0
    return max(0.0, min(1.0, overlapped_s / speech_s))


def overlap_tier(ratio: float) -> str:
    """``"low"`` below 0.10, ``"medium"`` from 0.10 up to (but excluding) 0.20, ``"high"`` from 0.20."""
    value = _finite("ratio", ratio)
    if value < 0:
        raise ValueError(f"overlap ratio must be >= 0, got {value!r}")
    if value < 0.10:
        return "low"
    if value < 0.20:
        return "medium"
    return "high"


def speaking_time(segments: Iterable[Any]) -> dict[str, float]:
    """Seconds of speech per speaker (self-overlapping intervals are unioned first). Sorted by label."""
    totals: dict[str, float] = {}
    for segment in canonical(segments):
        totals[segment.speaker] = totals.get(segment.speaker, 0.0) + segment.duration_s
    return totals


# ---------------------------------------------------------------------------
# RTTM


def read_rttm(text: str) -> tuple[Segment, ...]:
    """Parse ``SPEAKER`` lines of an RTTM document; other line types, blank lines and ``;;`` comments are skipped."""
    segments: list[Segment] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        fields = line.split()
        if fields[0] != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ValueError(f"RTTM line {line_number}: expected at least 8 fields, got {len(fields)}: {line!r}")
        try:
            start = float(fields[3])
            duration = float(fields[4])
        except ValueError as exc:
            raise ValueError(f"RTTM line {line_number}: bad start/duration: {line!r}") from exc
        try:
            segments.append(Segment(fields[7], start, start + duration))
        except ValueError as exc:
            raise ValueError(f"RTTM line {line_number}: {exc}") from exc
    return tuple(segments)


def _rttm_token(name: str, value: str) -> str:
    token = "_".join(str(value).split())
    if not token:
        raise ValueError(f"RTTM {name} must not be empty")
    return token


def write_rttm(segments: Iterable[Any], file_id: str) -> str:
    """Render segments as RTTM ``SPEAKER`` lines sorted by start time (whitespace in labels becomes ``_``)."""
    file_token = _rttm_token("file id", file_id)
    lines = []
    for segment in sorted(to_segments(segments), key=lambda s: (s.start_s, s.end_s, s.speaker)):
        speaker = _rttm_token("speaker", segment.speaker)
        lines.append(
            f"SPEAKER {file_token} 1 {segment.start_s:.3f} {segment.duration_s:.3f} "
            f"{_RTTM_NA} {_RTTM_NA} {speaker} {_RTTM_NA} {_RTTM_NA}"
        )
    return "".join(line + "\n" for line in lines)


def percent(rate: float | None) -> str:
    """``0.1234 -> "12.34%"``; ``None -> "n/a"``."""
    if rate is None:
        return "n/a"
    return f"{rate * 100:.2f}%"


__all__ = [
    "DEFAULT_COLLAR_S",
    "MERGE_GAP_S",
    "DerCounts",
    "DerScore",
    "JerScore",
    "Segment",
    "SpeakerCountScore",
    "canonical",
    "clip_segments",
    "der",
    "jer",
    "merge_segments",
    "overlap_ratio",
    "overlap_tier",
    "percent",
    "read_rttm",
    "speaker_count",
    "speaking_time",
    "to_segments",
    "write_rttm",
]
