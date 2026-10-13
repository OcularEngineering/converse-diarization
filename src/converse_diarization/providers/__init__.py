"""Speaker-diarization providers keyed by the ``provider`` field of a model config."""

from __future__ import annotations

from converse_diarization.config import ConfigError, ModelConfig
from converse_diarization.providers.assemblyai import AssemblyAIProvider
from converse_diarization.providers.azure import AzureProvider
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError
from converse_diarization.providers.deepgram import DeepgramProvider
from converse_diarization.providers.diarizen import DiariZenProvider
from converse_diarization.providers.elevenlabs import ElevenLabsProvider
from converse_diarization.providers.meta import MetaProvider
from converse_diarization.providers.nemo import NemoProvider
from converse_diarization.providers.openai_compatible import OpenAICompatibleProvider
from converse_diarization.providers.openrouter import OpenRouterProvider
from converse_diarization.providers.pyannote import PyannoteProvider
from converse_diarization.providers.rttm import RttmProvider
from converse_diarization.providers.xai import XAIProvider

PROVIDERS: dict[str, type[Provider]] = {
    OpenAICompatibleProvider.key: OpenAICompatibleProvider,
    OpenRouterProvider.key: OpenRouterProvider,
    DeepgramProvider.key: DeepgramProvider,
    DiariZenProvider.key: DiariZenProvider,
    AssemblyAIProvider.key: AssemblyAIProvider,
    AzureProvider.key: AzureProvider,
    ElevenLabsProvider.key: ElevenLabsProvider,
    MetaProvider.key: MetaProvider,
    NemoProvider.key: NemoProvider,
    PyannoteProvider.key: PyannoteProvider,
    RttmProvider.key: RttmProvider,
    XAIProvider.key: XAIProvider,
}


def create_provider(config: ModelConfig) -> Provider:
    try:
        cls = PROVIDERS[config.provider]
    except KeyError:
        available = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {config.provider!r}; available: {available}") from None
    return cls(config)


__all__ = ["PROVIDERS", "Hypothesis", "Provider", "ProviderError", "create_provider"]
