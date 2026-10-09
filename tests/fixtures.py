"""Small synthetic datasets for runner, mix and CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile

from converse_diarization.dataset import Dataset, build_dataset

SAMPLE_RATE = 16_000
BASE_MS = 1_757_534_367_463


def write_tone(path: Path, seconds: float, *, amplitude: float = 0.5, frequency: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    tone = (amplitude * np.sin(2 * np.pi * frequency * t)).astype("float32")
    soundfile.write(str(path), tone, SAMPLE_RATE, format="FLAC")
    return path


def track_row(
    session: str, speaker: str, capture_ms: int, words: list[tuple[str, float, float]], seconds: float
) -> dict:
    return {
        "session_id": session,
        "speaker_id": speaker,
        "file_name": f"{capture_ms}-{speaker}-hifi-audio-{capture_ms}.flac",
        "language": "en",
        "language_locale": "en-US",
        "duration_seconds": seconds,
        "transcript": " ".join(text for text, _, _ in words),
        "segments": [
            {
                "index": 0,
                "start": words[0][1] if words else 0.0,
                "end": words[-1][2] if words else 0.0,
                "text": " ".join(text for text, _, _ in words),
                "words": [{"word": text, "start": start, "end": end} for text, start, end in words],
            }
        ],
    }


def make_dataset(root: Path, *, amplitude: float = 0.5) -> Dataset:
    """Two conversations of two 4-second tracks each; the second track of ``s1`` starts 0.5 s late.

    Word timings are chosen so ``s1`` has overlapping speech (both talk during 1.0-1.5 s)
    and ``s2`` has none.
    """
    audio_dir = root / "audio-src"
    rows = []
    plans = {
        "s1": {
            "a": (BASE_MS, [("hi", 0.0, 0.4), ("there", 0.5, 1.5), ("friend", 2.5, 3.0)]),
            # offset 0.5 s: local words 0.5-1.0 land on 1.0-1.5 of the shared timeline
            "b": (BASE_MS + 500, [("hello", 0.5, 1.0), ("again", 2.0, 2.3), ("bye", 3.0, 3.4)]),
        },
        "s2": {
            "a": (BASE_MS + 10_000, [("one", 0.0, 1.0), ("two", 1.1, 2.0)]),
            "b": (BASE_MS + 10_000, [("three", 2.5, 3.0), ("four", 3.1, 3.9)]),
        },
    }
    for session, speakers in plans.items():
        for speaker, (capture_ms, words) in speakers.items():
            row = track_row(session, speaker, capture_ms, words, 4.0)
            write_tone(
                audio_dir / row["file_name"], 4.0, amplitude=amplitude, frequency=440.0 if speaker == "a" else 660.0
            )
            rows.append(row)
    return build_dataset(rows, lambda name: audio_dir / name, root / "dataset", source={"kind": "test"})
