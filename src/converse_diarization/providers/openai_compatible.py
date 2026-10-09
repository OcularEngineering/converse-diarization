"""Speaker-labelled transcription through ``POST /audio/transcriptions`` (OpenAI and compatible servers).

Models such as ``gpt-4o-transcribe-diarize`` return speaker turns when asked for
``response_format = "diarized_json"``. Other servers that speak the same wire format
may attach speakers to segments or to words; both shapes are accepted.
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

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_RESPONSE_FORMAT = "diarized_json"
DEFAULT_CHUNKING_STRATEGY = "auto"
# Fifteen-minute 16 kHz PCM16 WAVs are ~29 MB, above the 25 MB cap common to these endpoints.
DEFAULT_UPLOAD = "mp3"
DEFAULT_TIMEOUT_S = 900.0

_SPEAKER_KEYS = ("speaker", "speaker_id", "speaker_label")
_START_KEYS = ("start", "start_s", "start_time")
_END_KEYS = ("end", "end_s", "end_time")


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _labelled_segments(items: object) -> list[dict[str, Any]]:
    """Turn a list of segment- or word-like dicts into hypothesis segments, dropping unlabelled ones."""
    if not isinstance(items, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker = _first(item, _SPEAKER_KEYS)
        if speaker is None or (isinstance(speaker, str) and not speaker.strip()):
            continue
        segment = make_segment(speaker, _first(item, _START_KEYS), _first(item, _END_KEYS), item.get("text"))
        if segment is not None:
            segments.append(segment)
    return segments


class OpenAICompatibleProvider(Provider):
    key = "openai-compatible"
    option_names = frozenset(
        {
            "base_url",
            "api_key_env",
            "timeout_s",
            "upload",
            "response_format",
            "chunking_strategy",
            "diarize",
            "timestamp_granularities",
            "language_hint",
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
        self.base_url = str(options.get("base_url") or "").rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.upload = str(options.get("upload", DEFAULT_UPLOAD))
        self.response_format = str(options.get("response_format", DEFAULT_RESPONSE_FORMAT))
        chunking = options.get("chunking_strategy", DEFAULT_CHUNKING_STRATEGY)
        self.chunking_strategy = str(chunking) if chunking not in (None, "") else None
        self.diarize_flag = options.get("diarize", False)
        self.timestamp_granularities = options.get("timestamp_granularities")
        self.language_hint = bool(options.get("language_hint", True))
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
        if not self.base_url:
            problems.append("base_url is required (e.g. https://api.openai.com/v1)")
        elif not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not isinstance(self.diarize_flag, bool):
            problems.append("diarize must be true or false")
        granularities = self.timestamp_granularities
        if granularities is not None and (
            not isinstance(granularities, list) or not all(isinstance(item, str) for item in granularities)
        ):
            problems.append('timestamp_granularities must be a list of strings such as ["word", "segment"]')
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

    def _form(self, language: str) -> dict[str, Any]:
        form: dict[str, Any] = {"model": self.config.model, "response_format": self.response_format}
        if self.chunking_strategy:
            form["chunking_strategy"] = self.chunking_strategy
        if self.diarize_flag is True:
            form["diarize"] = "true"
        if isinstance(self.timestamp_granularities, list) and self.timestamp_granularities:
            form["timestamp_granularities[]"] = [str(item) for item in self.timestamp_granularities]
        if self.language_hint:
            form["language"] = primary_language_tag(language)
        return form

    # -- request ------------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        if not self.base_url:
            raise ProviderError("openai-compatible: base_url option is required")
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        form = self._form(language)
        headers = {"Authorization": f"Bearer {self._api_key()}"}
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
            raise error_from_response(response, context=f"diarization request for {conversation_id} to {url} failed")
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": form,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        request_id = response.headers.get("x-request-id")
        if request_id:
            raw["request_id"] = request_id
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{url} returned a JSON {type(data).__name__}, expected an object")
        return self._hypothesis(data, audio=audio, raw=raw)

    def _hypothesis(self, data: dict[str, Any], *, audio: Path, raw: dict[str, Any]) -> Hypothesis:
        segments = _labelled_segments(data.get("segments"))
        granularity = "segment"
        if not segments:
            segments = _labelled_segments(data.get("words"))
            granularity = "word"
        if not segments:
            raise ProviderError(
                f"{self.endpoint} did not return speaker labels on `segments` or `words` "
                f"(response_format = {self.response_format!r}); the model or server may not support "
                'diarization, or a different response_format (e.g. "diarized_json") may be required'
            )
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


__all__ = ["OpenAICompatibleProvider"]
