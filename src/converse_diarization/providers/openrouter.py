"""Speaker diarization through OpenRouter's ``POST /api/v1/audio/transcriptions``.

OpenRouter fronts many speech-to-text vendors behind one key. Diarization is not a
normalised parameter there: it is switched on through the serving vendor's own option,
sent under ``provider.options.<endpoint slug>``. That object only travels in the JSON
request body (base64 ``input_audio``), not in the OpenAI-style multipart form, so this
provider always posts JSON with ``response_format = "verbose_json"`` and asks for both
segment and word timestamps. OpenRouter forwards only the options block of the vendor
that actually serves the request, so the known blocks below can all be sent at once.

Speaker labels come back as ``speaker`` indices on ``words`` (preferred, merged into
regions by the scorer) or on ``segments`` (scored as-is).
"""

from __future__ import annotations

import base64
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import UPLOAD_FORMATS, audio_info, ffmpeg_available, prepare_upload
from converse_diarization.providers.base import (
    Hypothesis,
    Provider,
    ProviderError,
    make_segment,
    primary_language_tag,
)
from converse_diarization.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_TIMEOUT_S = 900.0
# 15-minute PCM16 WAV mixes are ~29 MB; base64 JSON past 25 MB is rejected for most vendors.
DEFAULT_UPLOAD = "mp3"

# Diarization switches in each vendor's own field names, keyed by OpenRouter endpoint slug
# (the `tag` in GET /api/v1/models/<model>/endpoints). Extend or override with the
# `provider_options` option; set a slug to {} to send nothing for it.
KNOWN_PROVIDER_OPTIONS: dict[str, dict[str, Any]] = {
    "deepgram": {"diarize": True, "punctuate": True},
    "azure": {"diarization": {"enabled": True}},
    "elevenlabs": {"diarize": True},
    "assemblyai": {"speaker_labels": True},
}

_FORMAT_BY_CONTENT_TYPE = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
}

_SPEAKER_KEYS = ("speaker", "speaker_id", "speaker_label")


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _labelled(items: object, text_key: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker = _first(item, _SPEAKER_KEYS)
        if speaker is None or (isinstance(speaker, str) and not speaker.strip()):
            continue
        segment = make_segment(speaker, item.get("start"), item.get("end"), item.get(text_key))
        if segment is not None:
            segments.append(segment)
    return segments


class OpenRouterProvider(Provider):
    key = "openrouter"
    option_names = frozenset(
        {"base_url", "api_key_env", "timeout_s", "upload", "language_hint", "provider_options", "zdr"}
    )

    def __init__(
        self,
        config: ModelConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.upload = str(options.get("upload", DEFAULT_UPLOAD))
        self.language_hint = bool(options.get("language_hint", True))
        self.zdr = options.get("zdr", False)
        overrides = options.get("provider_options")
        self.provider_options: dict[str, Any] = {
            **KNOWN_PROVIDER_OPTIONS,
            **(overrides if isinstance(overrides, dict) else {}),
        }
        self._overrides = overrides
        self._transport = transport
        self._sleep = sleep
        self._client_instance: httpx.Client | None = None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/audio/transcriptions"

    def check(self) -> list[str]:
        problems = super().check()
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        if "/" not in self.config.model:
            problems.append(f"model must be an OpenRouter id such as deepgram/nova-3; got {self.config.model!r}")
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not isinstance(self.zdr, bool):
            problems.append("zdr must be true or false")
        if self._overrides is not None and (
            not isinstance(self._overrides, dict) or not all(isinstance(v, dict) for v in self._overrides.values())
        ):
            problems.append(
                "provider_options must map endpoint slugs to option tables, e.g. { deepgram = { diarize = true } }"
            )
        return problems

    # -- plumbing -----------------------------------------------------------------

    def _client(self) -> httpx.Client:
        if self._client_instance is None:
            self._client_instance = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s, connect=30.0), transport=self._transport
            )
        return self._client_instance

    def _api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        return value

    def _body(self, language: str, *, audio_b64: str, audio_format: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "input_audio": {"data": audio_b64, "format": audio_format},
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
        }
        if self.language_hint:
            body["language"] = primary_language_tag(language)
        provider: dict[str, Any] = {}
        options = {slug: opts for slug, opts in self.provider_options.items() if opts}
        if options:
            provider["options"] = options
        if self.zdr is True:
            provider["zdr"] = True
        if provider:
            body["provider"] = provider
        return body

    # -- request ------------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        audio_format = _FORMAT_BY_CONTENT_TYPE.get(content_type, upload_path.suffix.lstrip(".").lower() or "wav")
        audio_b64 = base64.b64encode(upload_path.read_bytes()).decode("ascii")
        body = self._body(language, audio_b64=audio_b64, audio_format=audio_format)
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        client = self._client()
        url = self.endpoint

        def send() -> httpx.Response:
            return client.post(url, headers=headers, json=body, timeout=self.timeout_s)

        response = send_with_retry(send, sleep=self._sleep)
        if response.status_code != 200:
            raise error_from_response(response, context=f"openrouter: request for {conversation_id} failed")
        parameters = {key: value for key, value in body.items() if key != "input_audio"}
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": parameters,
            "upload": {"format": audio_format, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        generation_id = response.headers.get("x-generation-id")
        if generation_id:
            raw["generation_id"] = generation_id
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"openrouter: {url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"openrouter: {url} returned a JSON {type(data).__name__}, expected an object")
        return self._hypothesis(data, audio=audio, raw=raw)

    def _hypothesis(self, data: dict[str, Any], *, audio: Path, raw: dict[str, Any]) -> Hypothesis:
        segments = _labelled(data.get("words"), "word")
        granularity = "word"
        if not segments:
            segments = _labelled(data.get("segments"), "text")
            granularity = "segment"
        if not segments:
            raise ProviderError(
                f"openrouter: {self.config.model} returned no speaker labels on `words` or `segments`; "
                "the serving vendor may not support diarization through OpenRouter, or its option name "
                "differs (set `provider_options` for its endpoint slug)"
            )
        usage = data.get("usage")
        if isinstance(usage, dict):
            raw["usage"] = usage
        duration = data.get("duration")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration):
            duration_s: float | None = float(duration)
        else:
            duration_s = audio_info(audio)[0]
        return Hypothesis(segments=tuple(segments), granularity=granularity, duration_s=duration_s, raw=raw)

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["KNOWN_PROVIDER_OPTIONS", "OpenRouterProvider"]
