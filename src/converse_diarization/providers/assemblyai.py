"""AssemblyAI asynchronous transcription with ``speaker_labels`` (upload, submit, poll).

Word and utterance timings arrive in milliseconds and are converted to seconds here.
``_sleep`` is module-level so tests can replace it.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import httpx

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info, content_type_for
from converse_diarization.providers.base import (
    Hypothesis,
    Provider,
    ProviderError,
    make_segment,
    primary_language_tag,
)
from converse_diarization.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://api.assemblyai.com"
DEFAULT_API_KEY_ENV = "ASSEMBLYAI_API_KEY"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_POLL_WAIT_S = 5.0
DEFAULT_DEADLINE_S = 1800.0
PENDING = frozenset({"queued", "processing"})

_sleep = time.sleep


def _pause(seconds: float) -> None:
    _sleep(seconds)


def _ms_to_s(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) / 1000.0


def _timed_segments(items: object, text_key: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("speaker") is None:
            continue
        segment = make_segment(
            item["speaker"], _ms_to_s(item.get("start")), _ms_to_s(item.get("end")), item.get(text_key)
        )
        if segment is not None:
            segments.append(segment)
    return segments


class AssemblyAIProvider(Provider):
    key = "assemblyai"
    option_names = frozenset({"base_url", "api_key_env", "timeout_s", "poll_wait_s", "deadline_s", "language_hint"})

    def __init__(self, config: ModelConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.poll_wait_s = float(options.get("poll_wait_s", DEFAULT_POLL_WAIT_S))
        self.deadline_s = float(options.get("deadline_s", DEFAULT_DEADLINE_S))
        self.language_hint = bool(options.get("language_hint", True))
        self._transport = transport
        self._client_instance: httpx.Client | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    def check(self) -> list[str]:
        problems = super().check()
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        for name, value in (
            ("timeout_s", self.timeout_s),
            ("poll_wait_s", self.poll_wait_s),
            ("deadline_s", self.deadline_s),
        ):
            if not math.isfinite(value) or value <= 0:
                problems.append(f"{name} must be a positive number")
        return problems

    # -- plumbing -----------------------------------------------------------------

    def _client(self) -> httpx.Client:
        if self._client_instance is None:
            self._client_instance = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s, connect=30.0), transport=self._transport
            )
        return self._client_instance

    def _auth(self) -> dict[str, str]:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        return {"authorization": value}

    @staticmethod
    def _json_object(response: httpx.Response, *, context: str) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{context} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{context} returned a JSON {type(data).__name__}, expected an object")
        return data

    def _request_body(self, audio_url: str, language: str) -> dict[str, Any]:
        body: dict[str, Any] = {"audio_url": audio_url, "speaker_labels": True}
        if self.language_hint:
            body["language_code"] = primary_language_tag(language)
        if self.config.model and self.config.model != "default":
            body["speech_model"] = self.config.model
        return body

    # -- job flow -----------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        audio = Path(audio)
        if not audio.is_file():
            raise ProviderError(f"audio file not found: {audio}")
        headers = self._auth()
        client = self._client()
        upload_url = f"{self.base_url}/v2/upload"
        transcript_url = f"{self.base_url}/v2/transcript"
        content_type = content_type_for(audio)

        def send_upload() -> httpx.Response:
            with audio.open("rb") as handle:
                return client.post(
                    upload_url,
                    headers={**headers, "content-type": "application/octet-stream"},
                    content=handle,
                    timeout=max(self.timeout_s, 600.0),
                )

        uploaded = send_with_retry(send_upload, sleep=_pause)
        if uploaded.status_code >= 400:
            raise error_from_response(uploaded, context=f"assemblyai: uploading {conversation_id} failed")
        audio_url = self._json_object(uploaded, context="assemblyai: upload").get("upload_url")
        if not isinstance(audio_url, str) or not audio_url:
            raise ProviderError("assemblyai: upload response lacks `upload_url`")

        body = self._request_body(audio_url, language)
        created = send_with_retry(
            lambda: client.post(transcript_url, headers=headers, json=body, timeout=self.timeout_s), sleep=_pause
        )
        if created.status_code >= 400:
            raise error_from_response(created, context=f"assemblyai: submitting {conversation_id} failed")
        job = self._json_object(created, context="assemblyai: transcript creation")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ProviderError("assemblyai: transcript creation response lacks `id`")

        result = self._poll(client, f"{transcript_url}/{job_id}", headers, job_id=job_id)
        raw: dict[str, Any] = {
            "endpoint": transcript_url,
            "transcript_id": job_id,
            "parameters": {key: value for key, value in body.items() if key != "audio_url"},
            "upload": {"format": "flac", "content_type": content_type, "bytes": audio.stat().st_size},
        }
        return self._hypothesis(result, audio=audio, job_id=job_id, raw=raw)

    def _poll(self, client: httpx.Client, url: str, headers: dict[str, str], *, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.deadline_s
        while True:
            response = send_with_retry(lambda: client.get(url, headers=headers, timeout=self.timeout_s), sleep=_pause)
            if response.status_code >= 400:
                raise error_from_response(response, context=f"assemblyai: polling transcript {job_id} failed")
            state = self._json_object(response, context=f"assemblyai: poll of transcript {job_id}")
            status = str(state.get("status") or "").lower()
            if status == "completed":
                return state
            if status == "error":
                raise ProviderError(f"assemblyai: transcript {job_id} failed: {state.get('error') or 'unknown error'}")
            if status and status not in PENDING:
                raise ProviderError(f"assemblyai: transcript {job_id} reported unexpected status {status!r}")
            if time.monotonic() + self.poll_wait_s > deadline:
                raise ProviderError(
                    f"assemblyai: transcript {job_id} did not finish within {int(self.deadline_s)} s", retryable=True
                )
            _pause(self.poll_wait_s)

    @staticmethod
    def _hypothesis(result: dict[str, Any], *, audio: Path, job_id: str, raw: dict[str, Any]) -> Hypothesis:
        segments = _timed_segments(result.get("words"), "text")
        granularity = "word"
        if not segments:
            segments = _timed_segments(result.get("utterances"), "text")
            granularity = "segment"
        if not segments:
            raise ProviderError(
                f"assemblyai: transcript {job_id} completed without speaker labels on words or utterances"
            )
        duration = result.get("audio_duration")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration):
            duration_s: float | None = float(duration)
        else:
            duration_s = audio_info(audio)[0]
        return Hypothesis(segments=tuple(segments), granularity=granularity, duration_s=duration_s, raw=raw)

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None


__all__ = ["AssemblyAIProvider"]
