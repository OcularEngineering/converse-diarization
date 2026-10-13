"""Azure Speech fast transcription for MAI-Transcribe-2.

``POST {speech host}/speechtotext/transcriptions:transcribe``. Enhanced mode
with word timestamps only returns ``words`` when ``modelOptions`` is nested
inside ``enhancedMode``. Each word inherits the speaker of its phrase. Those
words are scored with the 0.2 s merge.

A Foundry project URL (``https://<resource>.services.ai.azure.com/api/projects/<name>``)
is reduced to the host. Speech lives at the resource root; the project path
is rejected.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

DEFAULT_API_KEY_ENV = "AZURE_FOUNDRY_KEY"
DEFAULT_ENDPOINT_ENV = "AZURE_FOUNDRY_ENDPOINT"
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_API_VERSION = "2025-10-15"
DEFAULT_UPLOAD = "flac"
DEFAULT_TRANSCRIBE_STYLE = "verbatim"
_STYLES = frozenset({"verbatim", "clean"})
_TRANSCRIBE_PATH = "/speechtotext/transcriptions:transcribe"


def speech_base_url(endpoint: str) -> str:
    """Speech host for a resource root or a Foundry project URL."""
    parsed = urlparse(str(endpoint).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"endpoint must be an http(s) URL; got {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _milliseconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) / 1000.0


class AzureProvider(Provider):
    key = "azure"
    option_names = frozenset(
        {
            "base_url",
            "api_key_env",
            "endpoint_env",
            "timeout_s",
            "upload",
            "language_hint",
            "api_version",
            "transcribe_style",
        }
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
        self.base_url = str(options.get("base_url", "")).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.endpoint_env = str(options.get("endpoint_env", DEFAULT_ENDPOINT_ENV))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.upload = str(options.get("upload", DEFAULT_UPLOAD))
        self.language_hint = bool(options.get("language_hint", True))
        self.api_version = str(options.get("api_version", DEFAULT_API_VERSION))
        self.transcribe_style = str(options.get("transcribe_style", DEFAULT_TRANSCRIBE_STYLE))
        self._transport = transport
        self._sleep = sleep
        self._client_instance: httpx.Client | None = None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        names = [self.api_key_env]
        if not self.base_url:
            names.append(self.endpoint_env)
        return tuple(names)

    def check(self) -> list[str]:
        problems = super().check()
        if self.base_url:
            try:
                speech_base_url(self.base_url)
            except ValueError as exc:
                problems.append(str(exc))
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not self.api_version:
            problems.append("api_version must be set")
        if self.transcribe_style not in _STYLES:
            problems.append(
                f"transcribe_style must be one of {', '.join(sorted(_STYLES))}; got {self.transcribe_style!r}"
            )
        if not isinstance(self.config.options.get("language_hint", True), bool):
            problems.append("language_hint must be true or false")
        return problems

    def _client(self) -> httpx.Client:
        if self._client_instance is None:
            self._client_instance = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s, connect=30.0), transport=self._transport
            )
        return self._client_instance

    def _secret(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise ProviderError(f"environment variable {name} is not set")
        return value

    def _speech_base(self) -> str:
        raw = self.base_url or self._secret(self.endpoint_env)
        try:
            return speech_base_url(raw)
        except ValueError as exc:
            raise ProviderError(str(exc)) from exc

    def _definition(self, language: str) -> dict[str, Any]:
        enhanced: dict[str, Any] = {
            "enabled": True,
            "model": self.config.model,
            "modelOptions": {"timestamps": "word", "transcribeStyle": self.transcribe_style},
        }
        definition: dict[str, Any] = {
            "diarization": {"enabled": True},
            "enhancedMode": enhanced,
        }
        if self.language_hint:
            definition["locales"] = [primary_language_tag(language)]
        return definition

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        base = self._speech_base()
        url = f"{base}{_TRANSCRIBE_PATH}?api-version={self.api_version}"
        definition = self._definition(language)
        headers = {"Ocp-Apim-Subscription-Key": self._secret(self.api_key_env)}
        client = self._client()

        def send() -> httpx.Response:
            with upload_path.open("rb") as handle:
                return client.post(
                    url,
                    headers=headers,
                    data={"definition": json.dumps(definition)},
                    files={"audio": (upload_path.name, handle, content_type)},
                    timeout=self.timeout_s,
                )

        response = send_with_retry(send, sleep=self._sleep)
        if response.status_code != 200:
            raise error_from_response(response, context=f"azure: request for {conversation_id} failed")
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": definition,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        request_id = response.headers.get("x-request-id") or response.headers.get("apim-request-id")
        if request_id:
            raw["request_id"] = request_id
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"azure: {url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"azure: {url} returned a JSON {type(data).__name__}, expected an object")
        return self._hypothesis(data, audio=audio, raw=raw)

    @staticmethod
    def _hypothesis(data: dict[str, Any], *, audio: Path, raw: dict[str, Any]) -> Hypothesis:
        phrases = data.get("phrases")
        if not isinstance(phrases, list):
            raise ProviderError("azure: response lacks a `phrases` list")
        words: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        for phrase in phrases:
            if not isinstance(phrase, dict):
                continue
            speaker = phrase.get("speaker")
            labelled = phrase.get("words")
            if isinstance(labelled, list) and labelled:
                for word in labelled:
                    if not isinstance(word, dict):
                        continue
                    start = _milliseconds(word.get("offsetMilliseconds"))
                    duration = _milliseconds(word.get("durationMilliseconds"))
                    if start is None or duration is None:
                        continue
                    segment = make_segment(speaker, start, start + duration, word.get("text"))
                    if segment is not None:
                        words.append(segment)
                continue
            start = _milliseconds(phrase.get("offsetMilliseconds"))
            duration = _milliseconds(phrase.get("durationMilliseconds"))
            if start is None or duration is None:
                continue
            segment = make_segment(speaker, start, start + duration, phrase.get("text"))
            if segment is not None:
                turns.append(segment)
        if words:
            segments = words
            granularity = "word"
        elif turns:
            segments = turns
            granularity = "segment"
        else:
            raise ProviderError(
                "azure: response carries no speaker-labelled phrases; diarization and enhanced mode must be accepted"
            )
        duration_ms = data.get("durationMilliseconds")
        duration_s = _milliseconds(duration_ms)
        if duration_s is None:
            duration_s = audio_info(audio)[0]
        return Hypothesis(segments=tuple(segments), granularity=granularity, duration_s=duration_s, raw=raw)

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["AzureProvider", "speech_base_url"]
