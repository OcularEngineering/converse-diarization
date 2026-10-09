"""Build the single-channel input a diarization model hears.

Each Ocular AI conversation is recorded on two separate microphones, one per
speaker, and the dataset manifest records how far apart the two recordings
started (``Track.offset_s``). Diarization models expect one channel with both
voices in it, so the harness reconstructs that channel deterministically:

1. decode both tracks to float32 mono (an unexpected multichannel file is averaged),
2. resample each one to :data:`MIX_SAMPLE_RATE` (16 kHz) with a polyphase filter,
3. place each track on the shared timeline by zero-padding ``offset_s`` of
   silence in front of it and padding the shorter one at the end,
4. add the two signals sample-wise (never averaged, never ducked, so relative
   levels survive), scale the whole mix down only if its peak exceeds 1.0,
5. write mono PCM16 WAV.

The policy id, the sample rate and a SHA-256 of the WAV bytes are recorded next
to the mixes in ``mix.json`` so a score can be traced back to the exact input.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from converse_diarization.dataset import Conversation, Dataset, DatasetError, sha256_file

MIX_POLICY = "mono-mix-v1"
MIX_SAMPLE_RATE = 16_000
MIX_DIR = "mix"
MIX_INDEX = "mix.json"


class MixError(RuntimeError):
    pass


@dataclass(frozen=True)
class MixResult:
    conversation: str
    path: Path
    sha256: str
    sample_rate: int
    sample_count: int
    sources: tuple[str, ...]

    @property
    def duration_s(self) -> float:
        return self.sample_count / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation": self.conversation,
            "path": self.path.name,
            "sha256": self.sha256,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "duration_s": round(self.duration_s, 6),
            "sources": list(self.sources),
            "policy": MIX_POLICY,
        }


def policy() -> dict[str, Any]:
    """The versioned description of how a mix is built (stored in run.json)."""
    return {
        "id": MIX_POLICY,
        "sample_rate": MIX_SAMPLE_RATE,
        "decode": "float32 mono; multichannel sources averaged",
        "resample": "scipy.signal.resample_poly (polyphase, Kaiser window)",
        "alignment": "each track delayed by its manifest offset_s; shorter track zero-padded at the end",
        "sum": "sample-wise sum; divided by the absolute peak only when the peak exceeds 1.0",
        "encoding": "mono PCM16 WAV",
    }


def _decode(path: Path) -> tuple[np.ndarray, int]:
    try:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise MixError(f"cannot decode {path}: {exc}") from exc
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    if not np.all(np.isfinite(mono)):
        raise MixError(f"{path} contains non-finite samples")
    return mono.astype(np.float32), int(rate)


def _resample(samples: np.ndarray, rate: int, target: int = MIX_SAMPLE_RATE) -> np.ndarray:
    if rate == target:
        return samples
    from math import gcd

    from scipy.signal import resample_poly

    factor = gcd(rate, target)
    return resample_poly(samples, target // factor, rate // factor).astype(np.float32)


def mix_arrays(tracks: list[tuple[np.ndarray, float]], *, sample_rate: int = MIX_SAMPLE_RATE) -> np.ndarray:
    """Sum already-resampled mono arrays after delaying each by its offset in seconds."""
    if not tracks:
        raise MixError("nothing to mix")
    placed: list[np.ndarray] = []
    for samples, offset_s in tracks:
        if samples.ndim != 1:
            raise MixError("tracks must be one-dimensional mono arrays")
        if offset_s < 0:
            raise MixError("track offsets must be non-negative")
        lead = int(round(offset_s * sample_rate))
        placed.append(np.concatenate([np.zeros(lead, dtype=np.float32), samples]) if lead else samples)
    length = max(item.shape[0] for item in placed)
    mixed = np.zeros(length, dtype=np.float32)
    for item in placed:
        mixed[: item.shape[0]] += item
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed /= peak
    return mixed


def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(samples, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def mix_conversation(conversation: Conversation, root: Path, output_dir: Path) -> MixResult:
    """Build (or rebuild) the mono mix for one conversation and return where it landed."""
    decoded = []
    sources = []
    for track in conversation.tracks:
        audio = track.audio_path(root)
        if not audio.is_file():
            raise MixError(f"missing audio {audio}")
        actual = sha256_file(audio)
        if actual != track.sha256:
            raise MixError(
                f"{audio} does not match the manifest (sha256 {actual} != {track.sha256}); "
                "run `converse-diarization dataset verify` and `dataset pull`"
            )
        samples, rate = _decode(audio)
        decoded.append((_resample(samples, rate), track.offset_s))
        sources.append(track.sha256)
    mixed = mix_arrays(decoded)
    wav = _encode_wav(mixed, MIX_SAMPLE_RATE)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{conversation.id}.wav"
    path.write_bytes(wav)
    return MixResult(
        conversation=conversation.id,
        path=path,
        sha256=hashlib.sha256(wav).hexdigest(),
        sample_rate=MIX_SAMPLE_RATE,
        sample_count=int(mixed.shape[0]),
        sources=tuple(sources),
    )


def mix_dir(dataset: Dataset) -> Path:
    return dataset.root / MIX_DIR


def load_index(dataset: Dataset) -> dict[str, dict[str, Any]]:
    path = mix_dir(dataset) / MIX_INDEX
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["conversation"]: item for item in data.get("mixes", [])}


def _write_index(dataset: Dataset, index: dict[str, dict[str, Any]]) -> None:
    payload = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_digest": dataset.digest(),
        "policy": policy(),
        "mixes": [index[key] for key in sorted(index)],
    }
    (mix_dir(dataset) / MIX_INDEX).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_mix(dataset: Dataset, conversation: Conversation, *, rebuild: bool = False) -> MixResult:
    """Return a verified mix for ``conversation``, building it when missing, stale or damaged."""
    index = load_index(dataset)
    entry = index.get(conversation.id)
    path = mix_dir(dataset) / f"{conversation.id}.wav"
    expected_sources = [track.sha256 for track in conversation.tracks]
    if (
        not rebuild
        and entry
        and path.is_file()
        and entry.get("policy") == MIX_POLICY
        and entry.get("sources") == expected_sources
        and sha256_file(path) == entry.get("sha256")
    ):
        return MixResult(
            conversation=conversation.id,
            path=path,
            sha256=entry["sha256"],
            sample_rate=int(entry["sample_rate"]),
            sample_count=int(entry["sample_count"]),
            sources=tuple(expected_sources),
        )
    result = mix_conversation(conversation, dataset.root, mix_dir(dataset))
    index[conversation.id] = result.to_dict()
    _write_index(dataset, index)
    return result


def mix_dataset(dataset: Dataset, *, rebuild: bool = False) -> list[MixResult]:
    """Build every missing mix (all of them with ``rebuild``)."""
    results = []
    for conversation in dataset.conversations:
        try:
            results.append(ensure_mix(dataset, conversation, rebuild=rebuild))
        except DatasetError as exc:
            raise MixError(str(exc)) from exc
    return results


__all__ = [
    "MIX_DIR",
    "MIX_POLICY",
    "MIX_SAMPLE_RATE",
    "MixError",
    "MixResult",
    "ensure_mix",
    "mix_arrays",
    "mix_conversation",
    "mix_dataset",
    "mix_dir",
    "policy",
]
