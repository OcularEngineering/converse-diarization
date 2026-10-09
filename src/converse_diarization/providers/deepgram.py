"""Deepgram pre-recorded transcription with ``diarize=true`` (``POST /v1/listen``).

The audio is streamed as the raw request body. Speaker labels arrive on every word
(integers), which the scorer merges into speech regions; utterances are used when a
server omits per-word speakers.
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

DEFAULT_BASE_URL = "https://api.deepgram.com/v1"
DEFAULT_API_KEY_ENV = "DEEPGRAM_API_KEY"
DEFAULT_TIMEOUT_S = 900.0
# The mix is already 16 kHz mono PCM16 WAV; "flac" means "send the file as it is".
DEFAULT_UPLOAD = "flac"


def _flag(value: object) -> str:
    return "true" if value is True else "false"


class DeepgramProvider(Provider):
    key = "deepgram"
    option_names = frozenset(
        {"base_url", "api_key_env", "timeout_s", "upload", "language_hint", "smart_format", "multichannel"}
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
        self.smart_format = options.get("smart_format")
        self.multichannel = options.get("multichannel", False)
        self._transport = transport
        self._sleep = sleep
        self._client_instance: httpx.Client | None = None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/listen"

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
        if self.smart_format is not None and not isinstance(self.smart_format, bool):
            problems.append("smart_format must be true or false")
        if not isinstance(self.multichannel, bool):
            problems.append("multichannel must be true or false")
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

    def _params(self, language: str) -> dict[str, str]:
        params = {
            "model": self.config.model,
            "diarize": "true",
            "utterances": "true",
            "punctuate": "true",
            "multichannel": _flag(self.multichannel),
        }
        if self.language_hint:
            params["language"] = primary_language_tag(language)
        if self.smart_format is not None:
            params["smart_format"] = _flag(self.smart_format)
        return params

    # -- request ------------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        params = self._params(language)
        headers = {"Authorization": f"Token {self._api_key()}", "Content-Type": content_type}
        client = self._client()
        url = self.endpoint

        def send() -> httpx.Response:
            with upload_path.open("rb") as handle:
                return client.post(url, headers=headers, params=params, content=handle, timeout=self.timeout_s)

        response = send_with_retry(send, sleep=self._sleep)
        if response.status_code != 200:
            raise error_from_response(response, context=f"deepgram: request for {conversation_id} failed")
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": params,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        request_id = response.headers.get("dg-request-id") or response.headers.get("x-request-id")
        if request_id:
            raw["request_id"] = request_id
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"deepgram: {url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"deepgram: {url} returned a JSON {type(data).__name__}, expected an object")
        return self._hypothesis(data, audio=audio, raw=raw)

    @staticmethod
    def _words(results: dict[str, Any]) -> list[dict[str, Any]]:
        channels = results.get("channels")
        if not isinstance(channels, list) or not channels or not isinstance(channels[0], dict):
            return []
        alternatives = channels[0].get("alternatives")
        if not isinstance(alternatives, list) or not alternatives or not isinstance(alternatives[0], dict):
            return []
        words = alternatives[0].get("words")
        if not isinstance(words, list):
            return []
        segments: list[dict[str, Any]] = []
        for word in words:
            if not isinstance(word, dict) or word.get("speaker") is None:
                continue
            text = word.get("punctuated_word") or word.get("word")
            segment = make_segment(word["speaker"], word.get("start"), word.get("end"), text)
            if segment is not None:
                segments.append(segment)
        return segments

    @staticmethod
    def _utterances(results: dict[str, Any]) -> list[dict[str, Any]]:
        utterances = results.get("utterances")
        if not isinstance(utterances, list):
            return []
        segments: list[dict[str, Any]] = []
        for utterance in utterances:
            if not isinstance(utterance, dict) or utterance.get("speaker") is None:
                continue
            segment = make_segment(
                utterance["speaker"], utterance.get("start"), utterance.get("end"), utterance.get("transcript")
            )
            if segment is not None:
                segments.append(segment)
        return segments

    def _hypothesis(self, data: dict[str, Any], *, audio: Path, raw: dict[str, Any]) -> Hypothesis:
        results = data.get("results")
        if not isinstance(results, dict):
            raise ProviderError("deepgram: response lacks a `results` object")
        segments = self._words(results)
        granularity = "word"
        if not segments:
            segments = self._utterances(results)
            granularity = "segment"
        if not segments:
            raise ProviderError(
                "deepgram: response carries no speaker labels on words or utterances; "
                "check that the model supports diarize=true"
            )
        metadata = data.get("metadata")
        duration = metadata.get("duration") if isinstance(metadata, dict) else None
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


__all__ = ["DeepgramProvider"]
