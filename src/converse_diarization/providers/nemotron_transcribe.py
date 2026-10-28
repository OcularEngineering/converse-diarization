"""NVIDIA Nemotron Transcribe served behind an OpenAI-style ``POST /v1/chat/completions``.

Nemotron 3.5 Transcribe (``nvidia/Nemotron-3.5-Transcribe-30B-A4B``) is an audio-in, text-out
chat model whose checkpoint bundles a speaker encoder and CTC timestamp heads. Asked for a
speaker-tagged transcript with ``capture_ctc_timestamps`` on, the server returns, next to the
transcript, a top-level ``ctc_timestamps`` object with:

``words``
    word text, start and end in seconds, and the transcript's ``<spk:N>`` tag;
``diarization``
    speaker-activity segments derived from the audio by the speaker encoder;
``speaker_tag_to_diarization_speaker``
    the mapping between transcript tags and diarization speakers;
``error``
    ``null`` on success or the alignment failure reason.

The model card says to use the ``diarization`` segments, not the word timestamps, when
evaluating diarization, so that is the default (``segments = "diarization"``, scored as
segments). ``segments = "words"`` scores the speaker-tagged words instead, merged with the
0.2 s rule like every other words-level system on the board.

The request follows NVIDIA's leaderboard evaluation guide for the model: no system prompt, the
guide's multi-speaker ASR instruction, reasoning off, ``temperature=0``, ``top_p=1``, ``seed=0``,
``skip_special_tokens=false`` so the tags survive, and a 16,384-token completion budget. The
guide's generation-loop cleanup (collapse a character repeated 21+ times, or a block repeated
20+ times) is applied exactly once to the decoded transcript before it is recorded. The
server's base URL is read from an environment variable (default ``NEMOTRON_TRANSCRIBE_ENDPOINT``)
so that deployments never appear in a config file; an optional bearer token can be named with
``api_key_env``.

The model decodes up to an hour of audio in one request, so by default the whole mix is sent at
once. A deployment behind a gateway that drops long requests can set ``max_clip_s``: mixes longer
than that are cut into ``window_s`` windows overlapping by ``overlap_s``, each window is
transcribed on its own, and the windows' local speakers are linked by who is speaking in the
overlap (the same scheme the ``meta`` provider uses).
"""

from __future__ import annotations

import base64
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from converse_diarization import __version__
from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError, make_segment
from converse_diarization.providers.http import error_from_response, send_with_retry
from converse_diarization.providers.windows import link_speakers, plan_windows, slice_wav

DEFAULT_ENDPOINT_ENV = "NEMOTRON_TRANSCRIBE_ENDPOINT"
DEFAULT_MAX_TOKENS = 16_384
DEFAULT_TIMEOUT_S = 1_200.0
DEFAULT_OVERLAP_S = 30.0
UPLOADS = ("flac", "wav")  # the service accepts WAV and FLAC data URLs; FLAC is a third of the size
SEGMENT_SOURCES = ("diarization", "words")

# The evaluation guide's default multi-speaker ASR prompt, verbatim.
DIARIZATION_PROMPT = (
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no "
    "punctuation and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, "
    "<spk:2>, etc.; assign one stable tag per speaker and keep speaker identities consistent. Output only the "
    "speaker-tagged transcript."
)

_SPEAKER_TAG = re.compile(r"<?\s*spk:?\s*(\d+)\s*>?", re.IGNORECASE)

# The evaluation guide's generation-loop cleanup (after the Qwen3-ASR repetition fix).
_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    """Collapse a character repeated 21+ times, then an exact 1-4096 character block repeated 20+ times."""
    text = _CHAR_RUN.sub(lambda m: m.group(1), text)
    return _BLOCK_RUN.sub(lambda m: m.group(1), text)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def speaker_label(value: object) -> str | None:
    """``"<spk:1>"``, ``"spk:1"``, ``1`` and ``"speaker_1"`` all become ``"1"``; ``None`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    match = _SPEAKER_TAG.fullmatch(text)
    if match:
        return match.group(1)
    return text


def _segment_from(item: object) -> dict[str, Any] | None:
    """One ``diarization`` entry as a segment dict; ``None`` if the entry is not a timed speaker span."""
    if isinstance(item, dict):
        start = _first(item, "start", "start_s", "start_time", "begin", "onset")
        end = _first(item, "end", "end_s", "end_time", "offset")
        if end is None and start is not None:
            duration = _first(item, "duration", "duration_s", "dur")
            if _number(start) is not None and _number(duration) is not None:
                end = _number(start) + _number(duration)
        speaker = speaker_label(_first(item, "speaker", "speaker_id", "label", "spk", "tag"))
        return make_segment(speaker, start, end, _first(item, "text", "word"))
    if isinstance(item, (list, tuple)) and len(item) == 3:
        start, end, speaker = item
        return make_segment(speaker_label(speaker), start, end)
    return None


def diarization_segments(value: object) -> list[dict[str, Any]]:
    """Accept a list of spans, or a ``{speaker: [[start, end], ...]}`` mapping, as segment dicts."""
    segments: list[dict[str, Any]] = []
    if isinstance(value, dict):
        # Either {"segments": [...]} or {speaker: [[start, end], ...]}.
        inner = value.get("segments")
        if isinstance(inner, list):
            return diarization_segments(inner)
        for speaker, spans in value.items():
            if not isinstance(spans, list):
                continue
            for span in spans:
                if isinstance(span, (list, tuple)) and len(span) >= 2:
                    segment = make_segment(speaker_label(speaker), span[0], span[1])
                elif isinstance(span, dict):
                    segment = _segment_from({**span, "speaker": span.get("speaker", speaker)})
                else:
                    segment = None
                if segment is not None:
                    segments.append(segment)
        return segments
    if isinstance(value, list):
        for item in value:
            segment = _segment_from(item)
            if segment is not None:
                segments.append(segment)
    return segments


def word_segments(value: object) -> list[dict[str, Any]]:
    """``ctc_timestamps.words`` as speaker-labelled word segments; words without a speaker tag are dropped."""
    segments: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return segments
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker = speaker_label(_first(item, "speaker", "speaker_tag", "spk", "tag"))
        if speaker is None:
            continue
        segment = make_segment(
            speaker, _first(item, "start", "start_s"), _first(item, "end", "end_s"), item.get("word")
        )
        if segment is not None:
            segments.append(segment)
    return segments


class NemotronTranscribeProvider(Provider):
    key = "nemotron-transcribe"
    option_names = frozenset(
        {
            "endpoint_env",
            "api_key_env",
            "upload",
            "timeout_s",
            "prompt",
            "segments",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "user_agent",
            "max_clip_s",
            "window_s",
            "overlap_s",
        }
    )

    def __init__(self, config: ModelConfig, *, transport: httpx.BaseTransport | None = None, sleep=None) -> None:
        super().__init__(config)
        options = config.options
        self.endpoint_env = str(options.get("endpoint_env", DEFAULT_ENDPOINT_ENV))
        self.api_key_env = options.get("api_key_env")
        self.upload = str(options.get("upload", "flac"))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.prompt = str(options.get("prompt", DIARIZATION_PROMPT))
        self.segments = str(options.get("segments", "diarization"))
        self.max_tokens = options.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = options.get("temperature", 0)
        self.top_p = options.get("top_p", 1.0)
        self.seed = options.get("seed", 0)
        self.user_agent = options.get("user_agent")
        self.max_clip_s = options.get("max_clip_s")  # None: the whole mix in one request
        self.window_s = options.get("window_s", self.max_clip_s)
        self.overlap_s = options.get("overlap_s", DEFAULT_OVERLAP_S)
        self._transport = transport
        self._sleep = sleep
        self._client_instance: httpx.Client | None = None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        names = [self.endpoint_env]
        if self.api_key_env:
            names.append(str(self.api_key_env))
        return tuple(names)

    def check(self) -> list[str]:
        problems = super().check()
        if self.upload not in UPLOADS:
            problems.append(f"upload must be one of {', '.join(UPLOADS)}; got {self.upload!r}")
        if self.segments not in SEGMENT_SOURCES:
            problems.append(f"segments must be one of {', '.join(SEGMENT_SOURCES)}; got {self.segments!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not self.prompt.strip():
            problems.append("prompt must not be empty")
        elif "spk" not in self.prompt.lower():
            problems.append(
                "prompt must ask for speaker tags (<spk:0> <spk:1> ...); an untagged transcript is one speaker"
            )
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            problems.append("max_tokens must be a positive integer")
        if _number(self.temperature) is None or self.temperature < 0:
            problems.append("temperature must be a number >= 0")
        if _number(self.top_p) is None or not 0 < self.top_p <= 1:
            problems.append("top_p must be a number in (0, 1]")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            problems.append("seed must be an integer or omitted")
        if self.max_clip_s is not None:
            max_clip = _number(self.max_clip_s)
            window = _number(self.window_s)
            overlap = _number(self.overlap_s)
            if max_clip is None or max_clip <= 0:
                problems.append("max_clip_s must be a positive number or omitted")
            elif window is None or not 0 < window <= max_clip:
                problems.append("window_s must be positive and no greater than max_clip_s")
            elif overlap is None or not 0 < overlap < window:
                problems.append("overlap_s must be positive and shorter than window_s")
        endpoint = os.environ.get(self.endpoint_env)
        if endpoint and not endpoint.startswith(("http://", "https://")):
            problems.append(f"{self.endpoint_env} must start with http:// or https://")
        return problems

    # -- plumbing -----------------------------------------------------------------

    def _endpoint(self) -> str:
        value = (os.environ.get(self.endpoint_env) or "").strip().rstrip("/")
        if not value:
            raise ProviderError(f"environment variable {self.endpoint_env} is not set")
        return value if value.endswith("/v1/chat/completions") else f"{value}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": str(self.user_agent) if self.user_agent else f"converse-diarization/{__version__}",
        }
        if self.api_key_env:
            token = os.environ.get(str(self.api_key_env))
            if not token:
                raise ProviderError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _client(self) -> httpx.Client:
        if self._client_instance is None:
            self._client_instance = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s, connect=30.0), transport=self._transport, follow_redirects=False
            )
        return self._client_instance

    def _upload(self, audio: Path) -> tuple[bytes, str]:
        """The mix as bytes plus its content type; FLAC is encoded losslessly from the 16 kHz WAV."""
        if self.upload == "wav":
            return audio.read_bytes(), "audio/wav"
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        target = self._workdir / f"{audio.stem}.flac"
        data, rate = sf.read(str(audio), dtype="int16", always_2d=True)
        sf.write(str(target), data[:, 0], rate, format="FLAC", subtype="PCM_16")
        return target.read_bytes(), "audio/flac"

    def _payload(self, audio_bytes: bytes, content_type: str) -> dict[str, Any]:
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "audio_url", "audio_url": {"url": f"data:{content_type};base64,{encoded}"}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": int(self.max_tokens),
            "stream": False,
            "n": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "mm_processor_kwargs": {"capture_ctc_timestamps": True},
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def _settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "prompt": self.prompt,
            "system_prompt": None,
            "segments": self.segments,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": int(self.max_tokens),
            "enable_thinking": False,
            "capture_ctc_timestamps": True,
        }
        if self.max_clip_s is not None:
            settings["max_clip_s"] = float(self.max_clip_s)
            settings["window_s"] = float(self.window_s)
            settings["overlap_s"] = float(self.overlap_s)
        return settings

    def _windows(self, duration_s: float) -> list[tuple[float, float]]:
        if self.max_clip_s is None:
            return [(0.0, duration_s)]
        return plan_windows(
            duration_s,
            window_s=float(self.window_s),
            overlap_s=float(self.overlap_s),
            max_clip_s=float(self.max_clip_s),
        )

    # -- request ------------------------------------------------------------------

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        duration_s = audio_info(audio)[0]
        windows = self._windows(duration_s)
        if len(windows) == 1:
            segments, granularity, raw = self._transcribe(audio, conversation_id=conversation_id, offset_s=0.0)
            return Hypothesis(segments=tuple(segments), granularity=granularity, duration_s=duration_s, raw=raw)

        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="converse-diarization-"))
        segments: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        granularity = "segment"
        for index, (start_s, end_s) in enumerate(windows):
            clip = self._workdir / f"{conversation_id}-{index}.wav"
            slice_wav(audio, clip, start_s, end_s)
            local, granularity, call = self._transcribe(
                clip, conversation_id=f"{conversation_id} window {index}", offset_s=start_s
            )
            if index == 0:
                segments = local
            else:
                segments = link_speakers(
                    segments,
                    local,
                    overlap_start_s=start_s,
                    overlap_end_s=min(start_s + float(self.overlap_s), end_s),
                )
            calls.append({"start_s": start_s, "end_s": end_s, **call})
        segments.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
        raw = {
            "endpoint": calls[0]["endpoint"],
            "model": self.config.model,
            "settings": self._settings(),
            "windows": calls,
        }
        return Hypothesis(segments=tuple(segments), granularity=granularity, duration_s=duration_s, raw=raw)

    def _transcribe(
        self, audio: Path, *, conversation_id: str, offset_s: float
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        """One chat completion for ``audio``; segments are shifted by ``offset_s`` onto the mix timeline."""
        url = self._endpoint()
        headers = self._headers()
        audio_bytes, content_type = self._upload(audio)
        payload = self._payload(audio_bytes, content_type)
        client = self._client()

        def send() -> httpx.Response:
            return client.post(url, headers=headers, json=payload, timeout=self.timeout_s)

        response = send_with_retry(send, **({"sleep": self._sleep} if self._sleep is not None else {}))
        if response.status_code != 200:
            raise error_from_response(response, context=f"{self.key}: request for {conversation_id} failed")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.key}: {url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{self.key}: {url} returned a JSON {type(data).__name__}, expected an object")
        if data.get("error"):
            raise ProviderError(f"{self.key}: {url} returned an error: {str(data['error'])[:300]}")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError(f"{self.key}: {url} response has no choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish != "stop":
            raise ProviderError(
                f"{self.key}: incomplete transcript for {conversation_id} (finish_reason={finish!r}); raise max_tokens"
            )
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        decoded = message.get("content") if isinstance(message.get("content"), str) else ""
        content = cleanup_generation_loops(decoded)

        alignment = data.get("ctc_timestamps")
        if not isinstance(alignment, dict):
            raise ProviderError(
                f"{self.key}: response carries no `ctc_timestamps` object; the server must run with the "
                "ctc_timestamp_middleware and the V1 model runner (see the model card)"
            )
        if alignment.get("error"):
            raise ProviderError(
                f"{self.key}: timestamp alignment failed for {conversation_id}: {alignment['error']!s}"[:400]
            )

        raw: dict[str, Any] = {
            "endpoint": url,
            "model": self.config.model,
            "settings": self._settings(),
            "finish_reason": finish,
            "usage": data.get("usage"),
            "upload": {"format": self.upload, "content_type": content_type, "bytes": len(audio_bytes)},
            "response_model": data.get("model"),
            "content": content,
            "generation_loops_collapsed": content != decoded,
            "speaker_tag_to_diarization_speaker": alignment.get("speaker_tag_to_diarization_speaker"),
            "words": len(alignment.get("words") or []) if isinstance(alignment.get("words"), list) else None,
        }
        request_id = response.headers.get("x-request-id") or data.get("id")
        if request_id:
            raw["request_id"] = request_id

        if self.segments == "words":
            segments = word_segments(alignment.get("words"))
            if not segments:
                raise ProviderError(
                    f"{self.key}: `ctc_timestamps.words` carries no speaker-tagged words for {conversation_id}; "
                    "check that the prompt asks for <spk:N> tags and skip_special_tokens is false"
                )
            granularity = "word"
        else:
            segments = diarization_segments(alignment.get("diarization"))
            if not segments:
                shape = type(alignment.get("diarization")).__name__
                raise ProviderError(
                    f"{self.key}: `ctc_timestamps.diarization` ({shape}) carries no speaker segments for "
                    f"{conversation_id}: {str(alignment.get('diarization'))[:200]}"
                )
            granularity = "segment"
        if offset_s:
            segments = [
                {**segment, "start_s": segment["start_s"] + offset_s, "end_s": segment["end_s"] + offset_s}
                for segment in segments
            ]
        segments.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
        return segments, granularity, raw

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = [
    "DIARIZATION_PROMPT",
    "NemotronTranscribeProvider",
    "cleanup_generation_loops",
    "diarization_segments",
    "speaker_label",
    "word_segments",
]
