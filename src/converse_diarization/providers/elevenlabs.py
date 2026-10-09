"""ElevenLabs Scribe speech-to-text with ``diarize=true`` (``POST /v1/speech-to-text``).

Every token in the response carries a ``type``; only ``"word"`` tokens are kept
(``"spacing"`` and ``"audio_event"`` tokens are not speech).
"""

from __future__ import annotations

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

DEFAULT_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_API_KEY_ENV = "ELEVENLABS_API_KEY"
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_UPLOAD = "flac"


class ElevenLabsProvider(Provider):
    key = "elevenlabs"
    option_names = frozenset({"base_url", "api_key_env", "timeout_s", "upload", "language_hint"})

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
        self._transport = transport
        self._sleep = sleep
        self._client_instance: httpx.Client | None = None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/speech-to-text"

    def check(self) -> list[str]:
        problems = super().check()
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
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

    def _form(self, language: str) -> dict[str, str]:
        form = {
            "model_id": self.config.model,
            "diarize": "true",
            "timestamps_granularity": "word",
            "tag_audio_events": "false",
        }
        if self.language_hint:
            form["language_code"] = primary_language_tag(language)
        return form

    # -- request ------------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        form = self._form(language)
        headers = {"xi-api-key": self._api_key()}
        client = self._client()
        url = self.endpoint

        def send() -> httpx.Response:
            with upload_path.open("rb") as handle:
                return client.post(
                    url,
                    headers=headers,
                    data=form,
                    files={"file": (upload_path.name, handle, content_type)},
                    timeout=self.timeout_s,
                )

        response = send_with_retry(send, sleep=self._sleep)
        if response.status_code != 200:
            raise error_from_response(response, context=f"elevenlabs: request for {conversation_id} failed")
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": form,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
        if request_id:
            raw["request_id"] = request_id
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"elevenlabs: {url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"elevenlabs: {url} returned a JSON {type(data).__name__}, expected an object")
        return self._hypothesis(data, audio=audio, raw=raw)

    @staticmethod
    def _hypothesis(data: dict[str, Any], *, audio: Path, raw: dict[str, Any]) -> Hypothesis:
        words = data.get("words")
        if not isinstance(words, list):
            raise ProviderError("elevenlabs: response lacks a `words` list")
        segments: list[dict[str, Any]] = []
        for word in words:
            if not isinstance(word, dict) or word.get("type", "word") != "word":
                continue
            speaker = word.get("speaker_id")
            if speaker is None:
                continue
            segment = make_segment(speaker, word.get("start"), word.get("end"), word.get("text"))
            if segment is not None:
                segments.append(segment)
        if not segments:
            raise ProviderError(
                "elevenlabs: response carries no speaker-labelled words; check that the model supports diarize=true"
            )
        # The API does not report the audio duration; read it from the file header.
        return Hypothesis(segments=tuple(segments), granularity="word", duration_s=audio_info(audio)[0], raw=raw)

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["ElevenLabsProvider"]
