"""Speaker-diarization providers keyed by the ``provider`` field of a model config."""

from __future__ import annotations

from converse_diarization.config import ConfigError, ModelConfig
from converse_diarization.providers.assemblyai import AssemblyAIProvider
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError
from converse_diarization.providers.deepgram import DeepgramProvider
from converse_diarization.providers.elevenlabs import ElevenLabsProvider
from converse_diarization.providers.openai_compatible import OpenAICompatibleProvider
from converse_diarization.providers.pyannote import PyannoteProvider
from converse_diarization.providers.rttm import RttmProvider

PROVIDERS: dict[str, type[Provider]] = {
    OpenAICompatibleProvider.key: OpenAICompatibleProvider,
    DeepgramProvider.key: DeepgramProvider,
    AssemblyAIProvider.key: AssemblyAIProvider,
    ElevenLabsProvider.key: ElevenLabsProvider,
    PyannoteProvider.key: PyannoteProvider,
    RttmProvider.key: RttmProvider,
}


def create_provider(config: ModelConfig) -> Provider:
    try:
        cls = PROVIDERS[config.provider]
    except KeyError:
        available = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {config.provider!r}; available: {available}") from None
    return cls(config)


__all__ = ["PROVIDERS", "Hypothesis", "Provider", "ProviderError", "create_provider"]
