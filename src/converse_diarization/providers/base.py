"""Provider contract shared by every speaker-diarization backend.

A provider receives the canonical 16 kHz mono mix of one conversation and
returns who spoke when. Two shapes are accepted:

``granularity = "segment"``
    The model emitted speaker turns with their own boundaries. They are scored
    as-is (no post-processing), the way dedicated diarization models and
    segment-timestamp transcription models are usually evaluated.
``granularity = "word"``
    The model emitted speaker-labelled words. Before scoring, consecutive words
    from the same speaker separated by at most ``MERGE_GAP_S`` are merged into
    one speech region, exactly how the reference is built from word timings.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from converse_diarization.config import ModelConfig
from converse_diarization.env import redact

Granularity = Literal["segment", "word"]


@dataclass(frozen=True)
class Hypothesis:
    """Speaker activity for one mixed conversation.

    ``segments`` items are plain dicts ``{"speaker": str, "start_s": float, "end_s": float, "text": str | None}``
    so they serialize straight into ``hypotheses.jsonl``. Speaker labels are the provider's own
    (``"SPEAKER_00"``, ``"A"``, ``"0"`` ...); the scorer finds the optimal mapping to the reference.
    """

    segments: tuple[dict[str, Any], ...]
    granularity: Granularity = "segment"
    duration_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(sorted({str(segment["speaker"]) for segment in self.segments}))


class ProviderError(RuntimeError):
    """A provider could not produce a hypothesis. Messages are redacted before they are stored."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(redact(message))
        self.retryable = retryable
        self.status = status


class Provider(ABC):
    """Turn one mono mix into a :class:`Hypothesis`."""

    key: ClassVar[str]
    option_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def config(self) -> ModelConfig:
        return self._config

    def required_env(self) -> tuple[str, ...]:
        """Environment variables the provider itself needs (in addition to ``config.env``)."""
        return ()

    def check(self) -> list[str]:
        """Return the problems that would prevent a run (missing env vars, unknown options). No network."""
        problems: list[str] = []
        names = list(self._config.env) + [name for name in self.required_env() if name not in self._config.env]
        problems.extend(missing_env(names))
        problems.extend(unknown_options(self._config.options, self.option_names))
        return problems

    @abstractmethod
    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        """Diarize ``audio`` (16 kHz mono PCM16 WAV); ``language`` is a BCP-47 primary tag such as ``"en"``."""

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def missing_env(names: Iterable[str]) -> list[str]:
    return [f"environment variable {name} is not set" for name in names if not os.environ.get(name)]


def unknown_options(options: Mapping[str, Any], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    unknown = sorted(name for name in options if name not in allowed_set)
    if not unknown:
        return []
    return [f"unknown option(s): {', '.join(unknown)}; allowed: {', '.join(sorted(allowed_set))}"]


def primary_language_tag(language: str) -> str:
    """``"en-US"`` -> ``"en"``."""
    return language.split("-", 1)[0].strip().lower()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def make_segment(speaker: object, start: object, end: object, text: object = None) -> dict[str, Any] | None:
    """Build one hypothesis segment dict, or ``None`` when the timing is unusable."""
    start_s, end_s = _number(start), _number(end)
    if speaker is None or start_s is None or end_s is None or end_s < start_s or start_s < 0:
        return None
    return {
        "speaker": str(speaker),
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "text": (str(text).strip() or None) if text is not None else None,
    }


__all__ = [
    "Granularity",
    "Hypothesis",
    "Provider",
    "ProviderError",
    "make_segment",
    "missing_env",
    "primary_language_tag",
    "unknown_options",
]
