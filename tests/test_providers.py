from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from converse_diarization.config import ConfigError, ModelConfig
from converse_diarization.providers import PROVIDERS, Hypothesis, ProviderError, create_provider
from converse_diarization.providers.assemblyai import AssemblyAIProvider
from converse_diarization.providers.deepgram import DeepgramProvider
from converse_diarization.providers.elevenlabs import ElevenLabsProvider
from converse_diarization.providers.openai_compatible import OpenAICompatibleProvider
from converse_diarization.providers.pyannote import PyannoteProvider
from converse_diarization.providers.rttm import RttmProvider, parse_rttm

# --- helpers -------------------------------------------------------------------


def _config(provider: str, model: str, **options: Any) -> ModelConfig:
    return ModelConfig(name=model, display_name=model, vendor=provider, provider=provider, model=model, options=options)


def _multipart(request: httpx.Request) -> tuple[dict[str, list[str]], dict[str, tuple[str, str, bytes]]]:
    """Split a multipart/form-data body into ``(fields, files)``; repeated fields keep every value."""
    content_type = request.headers["content-type"]
    assert content_type.startswith("multipart/form-data"), content_type
    boundary = content_type.split("boundary=", 1)[1].strip().encode()
    fields: dict[str, list[str]] = {}
    files: dict[str, tuple[str, str, bytes]] = {}
    for part in request.content.split(b"--" + boundary)[1:]:
        if part.strip() in (b"", b"--"):
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        body = body[:-2] if body.endswith(b"\r\n") else body
        headers = head.decode()
        name = re.search(r'name="([^"]+)"', headers).group(1)
        filename = re.search(r'filename="([^"]+)"', headers)
        if filename:
            mime = re.search(r"Content-Type:\s*(\S+)", headers, re.IGNORECASE).group(1)
            files[name] = (filename.group(1), mime, body)
        else:
            fields.setdefault(name, []).append(body.decode())
    return fields, files


class Recorder:
    """A ``MockTransport`` handler that records requests and answers from a queue or callback."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def _queue(*responses: httpx.Response):
    items = list(responses)

    def respond(request: httpx.Request) -> httpx.Response:
        assert items, f"unexpected extra request {request.method} {request.url}"
        return items.pop(0)

    return respond


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "mix.wav"
    sf.write(str(path), np.zeros(8000, dtype=np.float32), 16000, subtype="PCM_16", format="WAV")  # 0.5 s
    return path


# --- registry ------------------------------------------------------------------


def test_registry_exposes_every_provider():
    assert set(PROVIDERS) == {"openai-compatible", "deepgram", "assemblyai", "elevenlabs", "pyannote", "rttm"}
    assert isinstance(create_provider(_config("deepgram", "nova-3")), DeepgramProvider)
    for cls in PROVIDERS.values():
        assert cls.option_names, f"{cls.key} must declare option_names"


def test_create_provider_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown provider 'nope'"):
        create_provider(_config("nope", "x"))


# --- OpenAI-compatible ---------------------------------------------------------

OPENAI_KEY = "sk-testtesttesttesttesttest"
DIARIZED = {
    "task": "transcribe",
    "duration": 1.5,
    "text": "hello there general",
    "segments": [
        {
            "id": "s0",
            "type": "transcript.text.segment",
            "start": 0.0,
            "end": 0.7,
            "text": "hello there",
            "speaker": "A",
        },
        {"id": "s1", "type": "transcript.text.segment", "start": 0.8, "end": 1.5, "text": "general", "speaker": "B"},
    ],
}


def _openai(monkeypatch: pytest.MonkeyPatch, recorder: Recorder, **options: Any) -> OpenAICompatibleProvider:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    options.setdefault("base_url", "https://api.openai.com/v1/")
    options.setdefault("upload", "flac")
    config = _config("openai-compatible", "gpt-4o-transcribe-diarize", **options)
    return OpenAICompatibleProvider(config, transport=recorder.transport, sleep=lambda _: None)


def test_openai_compatible_posts_multipart_and_parses_diarized_segments(monkeypatch: pytest.MonkeyPatch, wav: Path):
    recorder = Recorder(_queue(httpx.Response(200, json=DIARIZED, headers={"x-request-id": "req_1"})))
    provider = _openai(monkeypatch, recorder, timestamp_granularities=["word", "segment"], diarize=True)

    hypothesis = provider.diarize(wav, language="en-US", conversation_id="conv-1")

    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.granularity == "segment"
    assert hypothesis.duration_s == 1.5
    assert hypothesis.segments == (
        {"speaker": "A", "start_s": 0.0, "end_s": 0.7, "text": "hello there"},
        {"speaker": "B", "start_s": 0.8, "end_s": 1.5, "text": "general"},
    )
    assert hypothesis.speakers == ("A", "B")
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    fields, files = _multipart(request)
    assert fields["model"] == ["gpt-4o-transcribe-diarize"]
    assert fields["response_format"] == ["diarized_json"]
    assert fields["chunking_strategy"] == ["auto"]
    assert fields["diarize"] == ["true"]
    assert fields["language"] == ["en"]
    assert fields["timestamp_granularities[]"] == ["word", "segment"]
    name, mime, payload = files["file"]
    assert name == "mix.wav" and mime == "audio/wav" and payload == wav.read_bytes()
    assert hypothesis.raw["request_id"] == "req_1"
    assert hypothesis.raw["upload"]["format"] == "flac"
    assert OPENAI_KEY not in json.dumps(hypothesis.raw)


def test_openai_compatible_omits_optional_fields_and_falls_back_to_words(monkeypatch: pytest.MonkeyPatch, wav: Path):
    body = {
        "text": "hi you",
        "words": [
            {"word": "hi", "start_time": 0.0, "end_time": 0.3, "speaker_id": "SPEAKER_00", "text": "hi"},
            {"word": "you", "start_time": 0.4, "end_time": 0.5, "speaker_label": "SPEAKER_01", "text": "you"},
        ],
    }
    recorder = Recorder(_queue(httpx.Response(200, json=body)))
    provider = _openai(
        monkeypatch, recorder, base_url="http://localhost:8000/v1", chunking_strategy="", language_hint=False
    )

    hypothesis = provider.diarize(wav, language="en", conversation_id="conv-2")

    assert hypothesis.granularity == "word"
    assert hypothesis.duration_s == pytest.approx(0.5)  # not reported by the server: read from the file
    assert hypothesis.segments == (
        {"speaker": "SPEAKER_00", "start_s": 0.0, "end_s": 0.3, "text": "hi"},
        {"speaker": "SPEAKER_01", "start_s": 0.4, "end_s": 0.5, "text": "you"},
    )
    request = recorder.requests[0]
    assert str(request.url) == "http://localhost:8000/v1/audio/transcriptions"
    fields, _ = _multipart(request)
    assert "chunking_strategy" not in fields
    assert "language" not in fields
    assert "diarize" not in fields
    assert "timestamp_granularities[]" not in fields


def test_openai_compatible_without_speaker_labels_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, wav: Path):
    body = {"text": "plain", "segments": [{"start": 0.0, "end": 1.0, "text": "plain"}]}
    recorder = Recorder(_queue(httpx.Response(200, json=body)))
    provider = _openai(monkeypatch, recorder, response_format="verbose_json")

    with pytest.raises(ProviderError) as info:
        provider.diarize(wav, language="en", conversation_id="conv-3")

    assert info.value.retryable is False
    assert "speaker labels" in str(info.value)
    assert "verbose_json" in str(info.value)


def test_openai_compatible_retries_500_then_succeeds(monkeypatch: pytest.MonkeyPatch, wav: Path):
    recorder = Recorder(_queue(httpx.Response(500, text="boom"), httpx.Response(200, json=DIARIZED)))
    provider = _openai(monkeypatch, recorder)

    hypothesis = provider.diarize(wav, language="en", conversation_id="conv-4")

    assert len(recorder.requests) == 2
    assert hypothesis.speakers == ("A", "B")
    # The file is re-read for every attempt.
    assert _multipart(recorder.requests[1])[1]["file"][2] == wav.read_bytes()


def test_openai_compatible_permanent_429_is_not_retried(monkeypatch: pytest.MonkeyPatch, wav: Path):
    body = {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota"}}
    recorder = Recorder(lambda request: httpx.Response(429, json=body))
    provider = _openai(monkeypatch, recorder)

    with pytest.raises(ProviderError) as info:
        provider.diarize(wav, language="en", conversation_id="conv-5")

    assert len(recorder.requests) == 1
    assert info.value.status == 429
    assert info.value.retryable is False
    assert "not retrying" in str(info.value)


def test_openai_compatible_errors_never_leak_the_key(monkeypatch: pytest.MonkeyPatch, wav: Path):
    recorder = Recorder(lambda request: httpx.Response(401, text=f"bad key {OPENAI_KEY}"))
    provider = _openai(monkeypatch, recorder)

    with pytest.raises(ProviderError) as info:
        provider.diarize(wav, language="en", conversation_id="conv-6")

    assert info.value.status == 401 and info.value.retryable is False
    assert OPENAI_KEY not in str(info.value)


def test_openai_compatible_check_validates_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    problems = OpenAICompatibleProvider(_config("openai-compatible", "m", upload="flac")).check()
    assert any("OPENAI_API_KEY" in item for item in problems)
    assert any("base_url is required" in item for item in problems)

    monkeypatch.setenv("MY_KEY", "x" * 20)
    monkeypatch.setattr("converse_diarization.providers.openai_compatible.ffmpeg_available", lambda: False)
    good = OpenAICompatibleProvider(_config("openai-compatible", "m", base_url="https://x/v1", api_key_env="MY_KEY"))
    assert good.check() == ['upload = "mp3" requires an ffmpeg binary on PATH']
    assert good.required_env() == ("MY_KEY",)

    bad = OpenAICompatibleProvider(
        _config(
            "openai-compatible",
            "m",
            base_url="ftp://x",
            api_key_env="MY_KEY",
            upload="aiff",
            timeout_s=0,
            diarize="yes",
            timestamp_granularities="word",
            bogus=1,
        )
    )
    problems = bad.check()
    assert any("base_url must start" in item for item in problems)
    assert any("upload must be" in item for item in problems)
    assert any("timeout_s" in item for item in problems)
    assert any("diarize" in item for item in problems)
    assert any("timestamp_granularities" in item for item in problems)
    assert any("unknown option" in item and "bogus" in item for item in problems)


# --- Deepgram ------------------------------------------------------------------

DEEPGRAM_KEY = "dgkeydgkeydgkeydgkeydgkey1234"


def _deepgram_body(*, words: bool = True) -> dict[str, Any]:
    alternative: dict[str, Any] = {"transcript": "Hi there. Hello.", "confidence": 0.99}
    if words:
        alternative["words"] = [
            {"word": "hi", "punctuated_word": "Hi", "start": 0.1, "end": 0.3, "confidence": 0.9, "speaker": 0},
            {"word": "there", "punctuated_word": "there.", "start": 0.35, "end": 0.6, "confidence": 0.9, "speaker": 0},
            {"word": "hello", "punctuated_word": "Hello.", "start": 0.7, "end": 1.0, "confidence": 0.9, "speaker": 1},
        ]
    return {
        "metadata": {"request_id": "dg-1", "duration": 1.25, "channels": 1},
        "results": {
            "channels": [{"alternatives": [alternative]}],
            "utterances": [
                {"start": 0.1, "end": 0.6, "transcript": "Hi there.", "speaker": 0},
                {"start": 0.7, "end": 1.0, "transcript": "Hello.", "speaker": 1},
            ],
        },
    }


def test_deepgram_streams_raw_audio_and_parses_words(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", DEEPGRAM_KEY)
    recorder = Recorder(_queue(httpx.Response(200, json=_deepgram_body())))
    provider = DeepgramProvider(_config("deepgram", "nova-3", smart_format=True), transport=recorder.transport)

    hypothesis = provider.diarize(wav, language="en-GB", conversation_id="conv-1")

    assert hypothesis.granularity == "word"
    assert hypothesis.duration_s == 1.25
    assert hypothesis.segments == (
        {"speaker": "0", "start_s": 0.1, "end_s": 0.3, "text": "Hi"},
        {"speaker": "0", "start_s": 0.35, "end_s": 0.6, "text": "there."},
        {"speaker": "1", "start_s": 0.7, "end_s": 1.0, "text": "Hello."},
    )
    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.scheme == "https" and request.url.host == "api.deepgram.com" and request.url.path == "/v1/listen"
    params = dict(request.url.params)
    assert params == {
        "model": "nova-3",
        "diarize": "true",
        "utterances": "true",
        "punctuate": "true",
        "multichannel": "false",
        "language": "en",
        "smart_format": "true",
    }
    assert request.headers["authorization"] == f"Token {DEEPGRAM_KEY}"
    assert request.headers["content-type"] == "audio/wav"
    assert request.content == wav.read_bytes()
    assert DEEPGRAM_KEY not in json.dumps(hypothesis.raw)


def test_deepgram_falls_back_to_utterances_and_can_omit_language(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", DEEPGRAM_KEY)
    recorder = Recorder(_queue(httpx.Response(200, json=_deepgram_body(words=False))))
    provider = DeepgramProvider(_config("deepgram", "nova-2", language_hint=False), transport=recorder.transport)

    hypothesis = provider.diarize(wav, language="fr", conversation_id="conv-2")

    assert hypothesis.granularity == "segment"
    assert hypothesis.segments == (
        {"speaker": "0", "start_s": 0.1, "end_s": 0.6, "text": "Hi there."},
        {"speaker": "1", "start_s": 0.7, "end_s": 1.0, "text": "Hello."},
    )
    assert "language" not in dict(recorder.requests[0].url.params)


def test_deepgram_check_and_http_errors(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    problems = DeepgramProvider(_config("deepgram", "nova-3", smart_format="yes", typo=1)).check()
    assert any("DEEPGRAM_API_KEY" in item for item in problems)
    assert any("smart_format" in item for item in problems)
    assert any("typo" in item for item in problems)

    monkeypatch.setenv("DEEPGRAM_API_KEY", DEEPGRAM_KEY)
    assert DeepgramProvider(_config("deepgram", "nova-3")).check() == []
    recorder = Recorder(lambda request: httpx.Response(403, json={"err_msg": "forbidden"}))
    provider = DeepgramProvider(_config("deepgram", "nova-3"), transport=recorder.transport)
    with pytest.raises(ProviderError) as info:
        provider.diarize(wav, language="en", conversation_id="conv-3")
    assert info.value.status == 403 and info.value.retryable is False and "forbidden" in str(info.value)


# --- AssemblyAI ----------------------------------------------------------------

ASSEMBLYAI_KEY = "aai-secret-key-0123456789abcdef"


def _assemblyai_router(poll_states: list[dict[str, Any]], *, wav: Path):
    states = list(poll_states)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == ASSEMBLYAI_KEY
        if request.method == "POST" and request.url.path == "/v2/upload":
            assert request.headers["content-type"] == "application/octet-stream"
            assert request.content == wav.read_bytes()
            return httpx.Response(200, json={"upload_url": "https://cdn.assemblyai.com/upload/abc"})
        if request.method == "POST" and request.url.path == "/v2/transcript":
            body = json.loads(request.content)
            assert body["audio_url"] == "https://cdn.assemblyai.com/upload/abc"
            assert body["speaker_labels"] is True
            return httpx.Response(200, json={"id": "tr_1", "status": "queued"})
        if request.method == "GET" and request.url.path == "/v2/transcript/tr_1":
            return httpx.Response(200, json=states.pop(0))
        raise AssertionError(f"unexpected call {request.method} {request.url}")

    return respond


def test_assemblyai_upload_submit_poll_converts_milliseconds(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", ASSEMBLYAI_KEY)
    sleeps: list[float] = []
    monkeypatch.setattr("converse_diarization.providers.assemblyai._sleep", sleeps.append)
    completed = {
        "id": "tr_1",
        "status": "completed",
        "audio_duration": 2,
        "words": [
            {"text": "Hi", "start": 250, "end": 650, "confidence": 0.9, "speaker": "A"},
            {"text": "there", "start": 700, "end": 1000, "confidence": 0.9, "speaker": "A"},
            {"text": "Hello", "start": 1200, "end": 1600, "confidence": 0.9, "speaker": "B"},
        ],
        "utterances": [{"speaker": "A", "start": 250, "end": 1000, "text": "Hi there"}],
    }
    recorder = Recorder(_assemblyai_router([{"status": "queued"}, {"status": "processing"}, completed], wav=wav))
    provider = AssemblyAIProvider(_config("assemblyai", "slam-1", poll_wait_s=3), transport=recorder.transport)

    hypothesis = provider.diarize(wav, language="en-AU", conversation_id="conv-1")

    assert hypothesis.granularity == "word"
    assert hypothesis.duration_s == 2.0
    assert hypothesis.segments == (
        {"speaker": "A", "start_s": 0.25, "end_s": 0.65, "text": "Hi"},
        {"speaker": "A", "start_s": 0.7, "end_s": 1.0, "text": "there"},
        {"speaker": "B", "start_s": 1.2, "end_s": 1.6, "text": "Hello"},
    )
    sequence = [(request.method, request.url.path) for request in recorder.requests]
    assert sequence == [
        ("POST", "/v2/upload"),
        ("POST", "/v2/transcript"),
        ("GET", "/v2/transcript/tr_1"),
        ("GET", "/v2/transcript/tr_1"),
        ("GET", "/v2/transcript/tr_1"),
    ]
    assert all(request.url.host == "api.assemblyai.com" for request in recorder.requests)
    submitted = json.loads(recorder.requests[1].content)
    assert submitted["language_code"] == "en"
    assert submitted["speech_model"] == "slam-1"
    assert sleeps == [3.0, 3.0]
    assert hypothesis.raw["transcript_id"] == "tr_1"
    assert "audio_url" not in hypothesis.raw["parameters"]
    assert ASSEMBLYAI_KEY not in json.dumps(hypothesis.raw)


def test_assemblyai_default_model_and_utterance_fallback(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", ASSEMBLYAI_KEY)
    monkeypatch.setattr("converse_diarization.providers.assemblyai._sleep", lambda _: None)
    completed = {
        "status": "completed",
        "words": [{"text": "Hi", "start": 0, "end": 300}],
        "utterances": [{"speaker": "A", "start": 0, "end": 300, "text": "Hi"}],
    }
    recorder = Recorder(_assemblyai_router([completed], wav=wav))
    provider = AssemblyAIProvider(
        _config("assemblyai", "default", base_url="https://api.eu.assemblyai.com/", language_hint=False),
        transport=recorder.transport,
    )

    hypothesis = provider.diarize(wav, language="de", conversation_id="conv-2")

    assert hypothesis.granularity == "segment"
    assert hypothesis.segments == ({"speaker": "A", "start_s": 0.0, "end_s": 0.3, "text": "Hi"},)
    assert hypothesis.duration_s == pytest.approx(0.5)
    submitted = json.loads(recorder.requests[1].content)
    assert "speech_model" not in submitted and "language_code" not in submitted
    assert recorder.requests[0].url.host == "api.eu.assemblyai.com"


def test_assemblyai_error_status_raises(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", ASSEMBLYAI_KEY)
    monkeypatch.setattr("converse_diarization.providers.assemblyai._sleep", lambda _: None)
    states = [{"status": "queued"}, {"status": "error", "error": "Audio file is too short"}]
    recorder = Recorder(_assemblyai_router(states, wav=wav))
    provider = AssemblyAIProvider(_config("assemblyai", "best"), transport=recorder.transport)

    with pytest.raises(ProviderError, match="Audio file is too short") as info:
        provider.diarize(wav, language="en", conversation_id="conv-3")
    assert info.value.retryable is False


def test_assemblyai_deadline_is_retryable(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", ASSEMBLYAI_KEY)
    monkeypatch.setattr("converse_diarization.providers.assemblyai._sleep", lambda _: None)
    recorder = Recorder(_assemblyai_router([{"status": "processing"}] * 3, wav=wav))
    provider = AssemblyAIProvider(
        _config("assemblyai", "best", deadline_s=1, poll_wait_s=5), transport=recorder.transport
    )

    with pytest.raises(ProviderError, match="did not finish") as info:
        provider.diarize(wav, language="en", conversation_id="conv-4")
    assert info.value.retryable is True


def test_assemblyai_check(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    problems = AssemblyAIProvider(_config("assemblyai", "best", poll_wait_s=0, upload="mp3")).check()
    assert any("ASSEMBLYAI_API_KEY" in item for item in problems)
    assert any("poll_wait_s" in item for item in problems)
    assert any("unknown option" in item and "upload" in item for item in problems)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", ASSEMBLYAI_KEY)
    assert AssemblyAIProvider(_config("assemblyai", "best")).check() == []


# --- ElevenLabs ----------------------------------------------------------------

ELEVENLABS_KEY = "xi_0123456789abcdef0123456789abcdef"


def test_elevenlabs_posts_multipart_and_keeps_only_words(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", ELEVENLABS_KEY)
    body = {
        "language_code": "en",
        "language_probability": 0.99,
        "text": "Hi there (laughs) hello",
        "words": [
            {"text": "Hi", "start": 0.0, "end": 0.2, "type": "word", "speaker_id": "speaker_0"},
            {"text": " ", "start": 0.2, "end": 0.25, "type": "spacing", "speaker_id": "speaker_0"},
            {"text": "there", "start": 0.25, "end": 0.5, "type": "word", "speaker_id": "speaker_0"},
            {"text": "(laughs)", "start": 0.5, "end": 0.9, "type": "audio_event", "speaker_id": "speaker_1"},
            {"text": "hello", "start": 1.0, "end": 1.4, "type": "word", "speaker_id": "speaker_1"},
        ],
    }
    recorder = Recorder(_queue(httpx.Response(200, json=body, headers={"request-id": "xi-1"})))
    provider = ElevenLabsProvider(_config("elevenlabs", "scribe_v1"), transport=recorder.transport)

    hypothesis = provider.diarize(wav, language="en-US", conversation_id="conv-1")

    assert hypothesis.granularity == "word"
    assert hypothesis.duration_s == pytest.approx(0.5)
    assert hypothesis.segments == (
        {"speaker": "speaker_0", "start_s": 0.0, "end_s": 0.2, "text": "Hi"},
        {"speaker": "speaker_0", "start_s": 0.25, "end_s": 0.5, "text": "there"},
        {"speaker": "speaker_1", "start_s": 1.0, "end_s": 1.4, "text": "hello"},
    )
    request = recorder.requests[0]
    assert str(request.url) == "https://api.elevenlabs.io/v1/speech-to-text"
    assert request.headers["xi-api-key"] == ELEVENLABS_KEY
    assert "authorization" not in request.headers
    fields, files = _multipart(request)
    assert fields == {
        "model_id": ["scribe_v1"],
        "diarize": ["true"],
        "timestamps_granularity": ["word"],
        "tag_audio_events": ["false"],
        "language_code": ["en"],
    }
    assert files["file"][0] == "mix.wav" and files["file"][1] == "audio/wav" and files["file"][2] == wav.read_bytes()
    assert hypothesis.raw["request_id"] == "xi-1"
    assert ELEVENLABS_KEY not in json.dumps(hypothesis.raw)


def test_elevenlabs_language_hint_off_and_missing_speakers(monkeypatch: pytest.MonkeyPatch, wav: Path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", ELEVENLABS_KEY)
    body = {"text": "hi", "words": [{"text": "hi", "start": 0.0, "end": 0.2, "type": "word"}]}
    recorder = Recorder(_queue(httpx.Response(200, json=body)))
    provider = ElevenLabsProvider(_config("elevenlabs", "scribe_v1", language_hint=False), transport=recorder.transport)

    with pytest.raises(ProviderError, match="speaker-labelled words"):
        provider.diarize(wav, language="en", conversation_id="conv-2")
    assert "language_code" not in _multipart(recorder.requests[0])[0]


def test_elevenlabs_check(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    problems = ElevenLabsProvider(_config("elevenlabs", "scribe_v1", upload="ogg", nope=1)).check()
    assert any("ELEVENLABS_API_KEY" in item for item in problems)
    assert any("upload must be" in item for item in problems)
    assert any("nope" in item for item in problems)
    monkeypatch.setenv("ELEVENLABS_API_KEY", ELEVENLABS_KEY)
    assert ElevenLabsProvider(_config("elevenlabs", "scribe_v1")).check() == []


# --- RTTM ----------------------------------------------------------------------

RTTM_TEXT = """\
SPEAKER conv-1 1 0.500 1.250 <NA> <NA> SPEAKER_01 <NA> <NA>
;; a comment line
SPEAKER conv-1 1 0.000 0.600 <NA> <NA> SPEAKER_00 <NA> <NA>

SPEAKER conv-1 1 2.0 0.3333 <NA> <NA> SPEAKER_00 <NA> <NA>
"""


def test_rttm_provider_reads_files_by_conversation_id(tmp_path: Path, wav: Path):
    hyp_dir = tmp_path / "hyps"
    hyp_dir.mkdir()
    (hyp_dir / "conv-1.rttm").write_text(RTTM_TEXT, encoding="utf-8")
    provider = RttmProvider(_config("rttm", "offline-model", dir=str(hyp_dir)))
    assert provider.check() == []

    hypothesis = provider.diarize(wav, language="en", conversation_id="conv-1")

    assert hypothesis.granularity == "segment"
    assert hypothesis.duration_s == pytest.approx(0.5)
    assert hypothesis.segments == (
        {"speaker": "SPEAKER_00", "start_s": 0.0, "end_s": 0.6, "text": None},
        {"speaker": "SPEAKER_01", "start_s": 0.5, "end_s": 1.75, "text": None},
        {"speaker": "SPEAKER_00", "start_s": 2.0, "end_s": 2.333, "text": None},
    )
    assert hypothesis.raw["speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    with pytest.raises(ProviderError, match="no hypothesis file for conv-9") as info:
        provider.diarize(wav, language="en", conversation_id="conv-9")
    assert info.value.retryable is False

    words = RttmProvider(_config("rttm", "m", dir=str(hyp_dir), granularity="word"))
    assert words.diarize(wav, language="en", conversation_id="conv-1").granularity == "word"


def test_rttm_provider_check_and_parse_errors(tmp_path: Path):
    problems = RttmProvider(_config("rttm", "m")).check()
    assert any("dir is required" in item for item in problems)
    problems = RttmProvider(_config("rttm", "m", dir=str(tmp_path / "missing"), granularity="frame")).check()
    assert any("does not exist" in item for item in problems)
    assert any("granularity" in item for item in problems)

    with pytest.raises(ProviderError, match="malformed RTTM line"):
        parse_rttm("SPEAKER f 1 0.0 1.0")
    with pytest.raises(ProviderError, match="not numbers"):
        parse_rttm("SPEAKER f 1 zero 1.0 <NA> <NA> A <NA> <NA>")
    with pytest.raises(ProviderError, match="unusable"):
        parse_rttm("SPEAKER f 1 1.0 -2.0 <NA> <NA> A <NA> <NA>")
    assert parse_rttm("SPKR-INFO f 1 <NA> <NA> <NA> unknown A <NA> <NA>") == []


# --- pyannote ------------------------------------------------------------------


def test_pyannote_check_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 30)
    provider = PyannoteProvider(_config("pyannote", "pyannote/speaker-diarization-community-1"))

    problems = provider.check()

    assert "pyannote.audio is not installed; run: uv sync --extra local" in problems
    assert provider.required_env() == ("HF_TOKEN",)
    with pytest.raises(ProviderError, match="uv sync --extra local"):
        provider.diarize(Path(__file__), language="en", conversation_id="c")


def test_pyannote_check_validates_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MY_HF", raising=False)
    provider = PyannoteProvider(
        _config(
            "pyannote",
            "pyannote/speaker-diarization-3.1",
            hf_token_env="MY_HF",
            device="tpu",
            num_speakers=0,
            min_speakers=3,
            max_speakers=2,
            extra=True,
        )
    )
    problems = provider.check()
    assert any("MY_HF" in item for item in problems)
    assert any("device" in item for item in problems)
    assert any("num_speakers must be" in item for item in problems)
    assert any("min_speakers must not exceed" in item for item in problems)
    assert any("unknown option" in item and "extra" in item for item in problems)


def test_pyannote_diarize_with_fake_pipeline(monkeypatch: pytest.MonkeyPatch, wav: Path):
    token = "hf_" + "b" * 30
    monkeypatch.setenv("HF_TOKEN", token)
    captured: dict[str, Any] = {}

    class Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start, self.end = start, end

    class Annotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield Turn(1.0, 2.5), "t1", "SPEAKER_01"
            yield Turn(0.0, 1.2), "t0", "SPEAKER_00"

    class Output:
        speaker_diarization = Annotation()
        exclusive_speaker_diarization = "must not be used"

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: Any):
            captured["from_pretrained"] = {"model": model, **kwargs}
            captured["loads"] = captured.get("loads", 0) + 1
            return cls()

        def to(self, device: Any):
            captured["device"] = device
            return self

        def __call__(self, path: str, **kwargs: Any):
            captured["call"] = {"path": path, **kwargs}
            return Output()

    pyannote_pkg = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = Pipeline  # type: ignore[attr-defined]
    pyannote_pkg.audio = pyannote_audio  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: f"device:{name}"  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    provider = PyannoteProvider(_config("pyannote", "pyannote/speaker-diarization-community-1", num_speakers=2))
    assert provider.check() == []

    hypothesis = provider.diarize(wav, language="en", conversation_id="conv-1")

    assert hypothesis.granularity == "segment"
    assert hypothesis.duration_s == pytest.approx(0.5)
    assert hypothesis.segments == (
        {"speaker": "SPEAKER_00", "start_s": 0.0, "end_s": 1.2, "text": None},
        {"speaker": "SPEAKER_01", "start_s": 1.0, "end_s": 2.5, "text": None},
    )
    assert captured["from_pretrained"] == {"model": "pyannote/speaker-diarization-community-1", "token": token}
    assert captured["device"] == "device:mps"
    assert captured["call"] == {"path": str(wav), "num_speakers": 2}
    assert token not in json.dumps(hypothesis.raw)
    provider.diarize(wav, language="en", conversation_id="conv-2")
    assert captured["loads"] == 1  # the pipeline is cached on the instance

    # pyannote.audio 3.x returns the Annotation directly.
    Pipeline.__call__ = lambda self, path, **kwargs: Annotation()  # type: ignore[method-assign]
    assert provider.diarize(wav, language="en", conversation_id="conv-3").speakers == ("SPEAKER_00", "SPEAKER_01")
