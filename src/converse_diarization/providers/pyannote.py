"""Local speaker diarization with ``pyannote.audio`` pipelines (``uv sync --extra local``).

The pipeline checkpoint named by ``model`` (for example
``pyannote/speaker-diarization-community-1``) is gated on the Hugging Face Hub, so a
token is read from ``HF_TOKEN`` (or the variable named by ``hf_token_env``).
Overlapping speech is kept: the plain ``speaker_diarization`` output is used, never the
``exclusive_speaker_diarization`` variant.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError, make_segment

DEFAULT_HF_TOKEN_ENV = "HF_TOKEN"
DEVICES = ("auto", "cpu", "cuda", "mps")
INSTALL_HINT = "pyannote.audio is not installed; run: uv sync --extra local"


def _dependencies_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _optional_int(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 1)


class PyannoteProvider(Provider):
    key = "pyannote"
    option_names = frozenset({"hf_token_env", "device", "num_speakers", "min_speakers", "max_speakers"})

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        options = config.options
        self.hf_token_env = str(options.get("hf_token_env", DEFAULT_HF_TOKEN_ENV))
        self.device = str(options.get("device", "auto")).lower()
        self.num_speakers = options.get("num_speakers")
        self.min_speakers = options.get("min_speakers")
        self.max_speakers = options.get("max_speakers")
        self._pipeline: Any = None

    def required_env(self) -> tuple[str, ...]:
        return (self.hf_token_env,)

    def check(self) -> list[str]:
        problems = super().check()
        if not _dependencies_available():
            problems.append(INSTALL_HINT)
        if self.device not in DEVICES:
            problems.append(f"device must be one of {', '.join(DEVICES)}; got {self.device!r}")
        for name, value in (
            ("num_speakers", self.num_speakers),
            ("min_speakers", self.min_speakers),
            ("max_speakers", self.max_speakers),
        ):
            if not _optional_int(value):
                problems.append(f"{name} must be a positive integer")
        if (
            _optional_int(self.min_speakers)
            and _optional_int(self.max_speakers)
            and self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            problems.append("min_speakers must not exceed max_speakers")
        if self.num_speakers is not None and (self.min_speakers is not None or self.max_speakers is not None):
            problems.append("num_speakers cannot be combined with min_speakers/max_speakers")
        return problems

    # -- model loading ------------------------------------------------------------

    def _token(self) -> str:
        value = os.environ.get(self.hf_token_env)
        if not value:
            raise ProviderError(f"environment variable {self.hf_token_env} is not set")
        return value

    def _resolve_device(self, torch: Any) -> Any:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ProviderError(INSTALL_HINT) from exc
        token = self._token()
        try:
            pipeline = Pipeline.from_pretrained(self.config.model, token=token)
        except TypeError:
            # Older pyannote.audio releases spell the keyword ``use_auth_token``.
            pipeline = Pipeline.from_pretrained(self.config.model, use_auth_token=token)
        except Exception as exc:
            raise ProviderError(f"pyannote: loading pipeline {self.config.model!r} failed: {exc}") from exc
        if pipeline is None:
            raise ProviderError(
                f"pyannote: pipeline {self.config.model!r} could not be loaded; accept the model's terms on the "
                f"Hugging Face Hub and make sure {self.hf_token_env} is a valid token"
            )
        device = self._resolve_device(torch)
        if hasattr(pipeline, "to"):
            pipeline = pipeline.to(device) or pipeline
        self._pipeline = pipeline
        return pipeline

    def _pipeline_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = int(self.num_speakers)
        if self.min_speakers is not None:
            kwargs["min_speakers"] = int(self.min_speakers)
        if self.max_speakers is not None:
            kwargs["max_speakers"] = int(self.max_speakers)
        return kwargs

    # -- inference ----------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        audio = Path(audio)
        if not audio.is_file():
            raise ProviderError(f"audio file not found: {audio}")
        pipeline = self._load()
        kwargs = self._pipeline_kwargs()
        try:
            result = pipeline(str(audio), **kwargs)
        except Exception as exc:
            raise ProviderError(f"pyannote: diarizing {conversation_id} failed: {exc}") from exc
        # pyannote.audio 4.x returns an output object; 3.x returns the Annotation itself.
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        segments = self._segments(annotation)
        if not segments:
            raise ProviderError(f"pyannote: pipeline returned no speaker turns for {conversation_id}")
        raw = {
            "model": self.config.model,
            "device": str(self.device),
            "parameters": kwargs,
            "speakers": sorted({segment["speaker"] for segment in segments}),
        }
        return Hypothesis(segments=tuple(segments), granularity="segment", duration_s=audio_info(audio)[0], raw=raw)

    @staticmethod
    def _segments(annotation: Any) -> list[dict[str, Any]]:
        if not hasattr(annotation, "itertracks"):
            raise ProviderError(f"pyannote: unexpected pipeline output of type {type(annotation).__name__}")
        segments: list[dict[str, Any]] = []
        for turn, _track, label in annotation.itertracks(yield_label=True):
            start, end = float(turn.start), float(turn.end)
            if not (math.isfinite(start) and math.isfinite(end)):
                continue
            segment = make_segment(label, max(start, 0.0), max(end, 0.0))
            if segment is not None:
                segments.append(segment)
        segments.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
        return segments

    def close(self) -> None:
        self._pipeline = None


__all__ = ["PyannoteProvider"]
