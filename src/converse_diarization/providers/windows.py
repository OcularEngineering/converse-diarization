"""Splitting a long mix into overlapping windows and linking speakers back across them.

Some hosted endpoints cannot process a whole 15-minute conversation in one request (an upload
cap, or a gateway that drops requests after about a minute). The mix is then cut into windows
that overlap by ``overlap_s``. Speaker labels restart on every request, so each later window is
renamed onto the speakers already known by matching who is speaking inside the overlap, and only
the audio past that overlap is kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment


def plan_windows(
    duration_s: float, *, window_s: float, overlap_s: float, max_clip_s: float
) -> list[tuple[float, float]]:
    """Return ``[start, end)`` spans that each stay within ``max_clip_s`` and overlap by ``overlap_s``."""
    if duration_s <= max_clip_s:
        return [(0.0, duration_s)]
    step = window_s - overlap_s
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s - 1e-6:
        end = min(start + window_s, duration_s)
        spans.append((start, end))
        if end >= duration_s - 1e-6:
            break
        start += step
    return spans


def link_speakers(
    base: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    *,
    overlap_start_s: float,
    overlap_end_s: float,
) -> list[dict[str, Any]]:
    """Rename ``extra`` speakers to ``base`` speakers using time overlap, then keep only the tail past the overlap."""
    mapping = _speaker_map(base, extra, overlap_start_s, overlap_end_s)
    kept = list(base)
    for segment in extra:
        start_s = max(float(segment["start_s"]), overlap_end_s)
        end_s = float(segment["end_s"])
        if end_s <= start_s:
            continue
        kept.append({**segment, "speaker": mapping[segment["speaker"]], "start_s": start_s})
    return kept


def _speaker_map(
    base: list[dict[str, Any]], extra: list[dict[str, Any]], overlap_start_s: float, overlap_end_s: float
) -> dict[str, str]:
    extra_labels = sorted({segment["speaker"] for segment in extra})
    base_labels = sorted({segment["speaker"] for segment in base})
    mapping: dict[str, str] = {}
    if extra_labels and base_labels:
        matrix = np.zeros((len(extra_labels), len(base_labels)))
        for row, extra_label in enumerate(extra_labels):
            for column, base_label in enumerate(base_labels):
                matrix[row, column] = _shared_time(base, extra, base_label, extra_label, overlap_start_s, overlap_end_s)
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        for row, column in zip(rows, columns, strict=True):
            if matrix[row, column] > 0:
                mapping[extra_labels[row]] = base_labels[column]
    taken = {segment["speaker"] for segment in base}
    taken.update(mapping.values())
    for label in extra_labels:
        if label in mapping:
            continue
        renamed = label
        suffix = 2
        while renamed in taken:
            renamed = f"{label}-{suffix}"
            suffix += 1
        mapping[label] = renamed
        taken.add(renamed)
    return mapping


def _shared_time(
    base: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    base_label: str,
    extra_label: str,
    overlap_start_s: float,
    overlap_end_s: float,
) -> float:
    total = 0.0
    for left in base:
        if left["speaker"] != base_label:
            continue
        for right in extra:
            if right["speaker"] != extra_label:
                continue
            start = max(float(left["start_s"]), float(right["start_s"]), overlap_start_s)
            end = min(float(left["end_s"]), float(right["end_s"]), overlap_end_s)
            if end > start:
                total += end - start
    return total


def slice_wav(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    """Write ``src[start_s:end_s]`` as 16 kHz mono PCM16 WAV (24 kHz is passed through)."""
    info = sf.info(str(src))
    start = max(0, int(round(start_s * info.samplerate)))
    stop = min(info.frames, int(round(end_s * info.samplerate)))
    data, rate = sf.read(str(src), start=start, stop=max(start + 1, stop), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate not in (16000, 24000):
        duration = mono.size / float(rate)
        count = max(1, int(round(duration * 16000)))
        src_t = np.arange(mono.size, dtype=np.float64) / float(rate)
        dst_t = np.arange(count, dtype=np.float64) / 16000.0
        mono = np.interp(dst_t, src_t, mono.astype(np.float64)).astype(np.float32)
        rate = 16000
    sf.write(str(dst), np.clip(mono, -1.0, 1.0), rate, subtype="PCM_16", format="WAV")


__all__ = ["link_speakers", "plan_windows", "slice_wav"]
