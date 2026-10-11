"""Meta Muse Voice Transcribe in ``DIARIZATION`` mode (``POST /v1/asr/transcribe``).

The file endpoint accepts at most 10 minutes of mono 16-bit PCM WAV. Longer mixes
are split into windows that overlap by ``overlap_s``. Speaker labels restart on
every request, so each later window is renamed by matching speaker time inside
the overlap, then only the audio past that overlap is kept.
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

import httpx
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError, make_segment
from converse_diarization.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_API_KEY_ENV = "META_API_KEY"
DEFAULT_TIMEOUT_S = 900.0
# Nine-minute windows pass the 10-minute API cap and still die at the gateway (HTTP 504).
# Three minutes returns in under a minute.
DEFAULT_WINDOW_S = 180.0
DEFAULT_OVERLAP_S = 30.0
DEFAULT_MAX_CLIP_S = 600.0

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin Chinese",
}


def plan_windows(duration_s: float, *, window_s: float, overlap_s: float, max_clip_s: float) -> list[tuple[float, float]]:
    """Return ``[start, end)`` spans that each stay within ``max_clip_s`` and overlap by ``overlap_s``."""
    if duration_s <= max_clip_s:
        return [(0.0, duration_s)]
    step = window_s - overlap_s
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s - 1e-6:
        end = min(start + window_s, duration_s)
        spans.append((start, end))
        if end >= duration_s - 1e-6:
            break
        start += step
    return spans


def link_speakers(
    base: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    *,
    overlap_start_s: float,
    overlap_end_s: float,
) -> list[dict[str, Any]]:
    """Rename ``extra`` speakers to ``base`` speakers using time overlap, then keep only the tail past the overlap."""
    mapping = _speaker_map(base, extra, overlap_start_s, overlap_end_s)
    kept = list(base)
    for segment in extra:
        start_s = max(float(segment["start_s"]), overlap_end_s)
        end_s = float(segment["end_s"])
        if end_s <= start_s:
            continue
        kept.append({**segment, "speaker": mapping[segment["speaker"]], "start_s": start_s})
    return kept


def _speaker_map(
    base: list[dict[str, Any]], extra: list[dict[str, Any]], overlap_start_s: float, overlap_end_s: float
) -> dict[str, str]:
    extra_labels = sorted({segment["speaker"] for segment in extra})
    base_labels = sorted({segment["speaker"] for segment in base})
    mapping: dict[str, str] = {}
    if extra_labels and base_labels:
        matrix = np.zeros((len(extra_labels), len(base_labels)))
        for row, extra_label in enumerate(extra_labels):
            for column, base_label in enumerate(base_labels):
                matrix[row, column] = _shared_time(
                    base, extra, base_label, extra_label, overlap_start_s, overlap_end_s
                )
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        for row, column in zip(rows, columns, strict=True):
            if matrix[row, column] > 0:
                mapping[extra_labels[row]] = base_labels[column]
    taken = {segment["speaker"] for segment in base}
    taken.update(mapping.values())
    for label in extra_labels:
        if label in mapping:
            continue
        renamed = label
        suffix = 2
        while renamed in taken:
            renamed = f"{label}-{suffix}"
            suffix += 1
        mapping[label] = renamed
        taken.add(renamed)
    return mapping


def _shared_time(
    base: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    base_label: str,
    extra_label: str,
    overlap_start_s: float,
    overlap_end_s: float,
) -> float:
    total = 0.0
    for left in base:
        if left["speaker"] != base_label:
            continue
        for right in extra:
            if right["speaker"] != extra_label:
                continue
            start = max(float(left["start_s"]), float(right["start_s"]), overlap_start_s)
            end = min(float(left["end_s"]), float(right["end_s"]), overlap_end_s)
            if end > start:
                total += end - start
    return total


class MetaProvider(Provider):
    key = "meta"
    option_names = frozenset(
        {
            "base_url",
            "api_key_env",
            "timeout_s",
            "language_hint",
            "window_s",
            "overlap_s",
            "max_clip_s",
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
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.language_hint = bool(options.get("language_hint", True))
        self.window_s = float(options.get("window_s", DEFAULT_WINDOW_S))
        self.overlap_s = float(options.get("overlap_s", DEFAULT_OVERLAP_S))
        self.max_clip_s = float(options.get("max_clip_s", DEFAULT_MAX_CLIP_S))
        self._transport = transport
        self._sleep = sleep
        self._client_instance: httpx.Client | None = None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/asr/transcribe"

    def check(self) -> list[str]:
        problems = super().check()
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not math.isfinite(self.max_clip_s) or self.max_clip_s <= 0:
            problems.append("max_clip_s must be a positive number")
        elif not math.isfinite(self.window_s) or not 0 < self.window_s <= self.max_clip_s:
            problems.append("window_s must be positive and no greater than max_clip_s")
        elif not math.isfinite(self.overlap_s) or not 0 < self.overlap_s < self.window_s:
            problems.append("overlap_s must be positive and shorter than window_s")
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

    def _request_body(self, language: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "audioEncoding": "WAV",
            "mode": "DIARIZATION",
        }
        if self.language_hint:
            name = _LANGUAGE_NAMES.get(language.split("-", 1)[0].strip().lower())
            if name:
                body["languageBias"] = [name]
        return body

    # -- request ------------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        duration_s, _rate = audio_info(audio)
        windows = plan_windows(
            duration_s, window_s=self.window_s, overlap_s=self.overlap_s, max_clip_s=self.max_clip_s
        )
        segments: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for index, (start_s, end_s) in enumerate(windows):
            clip = self._workdir / f"{conversation_id}-{index}.wav"
            _slice_wav(audio, clip, start_s, end_s)
            data, parameters = self._transcribe(clip, language=language, conversation_id=conversation_id)
            local = _turns(data, offset_s=start_s)
            if index == 0:
                segments = local
            else:
                segments = link_speakers(
                    segments,
                    local,
                    overlap_start_s=start_s,
                    overlap_end_s=min(start_s + self.overlap_s, end_s),
                )
            calls.append({"start_s": start_s, "end_s": end_s, "parameters": parameters, "response": data})
        if not segments:
            raise ProviderError("meta: response carries no speaker-labelled turns; mode must be DIARIZATION")
        raw = {"endpoint": self.endpoint, "windows": calls}
        return Hypothesis(segments=tuple(segments), granularity="segment", duration_s=duration_s, raw=raw)

    def _transcribe(self, audio: Path, *, language: str, conversation_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self._request_body(language)
        headers = {"Authorization": f"Bearer {self._api_key()}", "Accept": "application/json"}
        client = self._client()
        url = self.endpoint

        def send() -> httpx.Response:
            with audio.open("rb") as handle:
                return client.post(
                    url,
                    headers=headers,
                    files={
                        "request": ("request.json", json.dumps(payload), "application/json"),
                        "audio": (audio.name, handle, "audio/wav"),
                    },
                    timeout=self.timeout_s,
                )

        response = send_with_retry(send, sleep=self._sleep)
        if response.status_code != 200:
            raise error_from_response(response, context=f"meta: request for {conversation_id} failed")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"meta: {url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"meta: {url} returned a JSON {type(data).__name__}, expected an object")
        return data, payload

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


def _slice_wav(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    info = sf.info(str(src))
    start = max(0, int(round(start_s * info.samplerate)))
    stop = min(info.frames, int(round(end_s * info.samplerate)))
    data, rate = sf.read(str(src), start=start, stop=max(start + 1, stop), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate not in (16000, 24000):
        duration = mono.size / float(rate)
        count = max(1, int(round(duration * 16000)))
        src_t = np.arange(mono.size, dtype=np.float64) / float(rate)
        dst_t = np.arange(count, dtype=np.float64) / 16000.0
        mono = np.interp(dst_t, src_t, mono.astype(np.float64)).astype(np.float32)
        rate = 16000
    sf.write(str(dst), np.clip(mono, -1.0, 1.0), rate, subtype="PCM_16", format="WAV")


def _turns(data: dict[str, Any], *, offset_s: float) -> list[dict[str, Any]]:
    turns = data.get("turns")
    if not isinstance(turns, list):
        raise ProviderError("meta: response lacks a `turns` list")
    segments: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        start_ms = turn.get("startMs")
        end_ms = turn.get("endMs")
        start = None if start_ms is None else float(start_ms) / 1000.0 + offset_s
        end = None if end_ms is None else float(end_ms) / 1000.0 + offset_s
        segment = make_segment(turn.get("speaker"), start, end, turn.get("transcript"))
        if segment is not None:
            segments.append(segment)
    return segments


__all__ = ["MetaProvider", "link_speakers", "plan_windows"]
