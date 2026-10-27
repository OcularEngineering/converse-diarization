"""Local Qwen3-ASR speaker pipeline through FunASR (open weights, Apache-2.0 model).

Qwen3-ASR transcribes and Qwen3-ForcedAligner stamps every word; FunASR's speaker stage
(FSMN-VAD speech regions, CAM++ embeddings on 1.5 s windows, clustering) decides who is
speaking when. Each word takes the speaker of the turn that contains it and same-speaker
words up to 0.2 s apart are merged before scoring, the rule used for the reference and for
the other transcription-based providers. ``turns = "speaker"`` scores the clustered speaker
timeline as segments instead.

The ``qwen-asr`` package pins its own transformers and FunASR brings its own stack, so
inference runs in a separate interpreter named by the ``python`` option (default
``.cache/qwen-asr-venv/bin/python``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError, make_segment, primary_language_tag

DEFAULT_PYTHON = Path(".cache/qwen-asr-venv/bin/python")
WORKER = Path(__file__).resolve().parents[1] / "qwen_asr_worker.py"
INSTALL_HINT = (
    "Qwen3-ASR interpreter not found. It needs its own environment: uv venv .cache/qwen-asr-venv --python 3.12; "
    "uv pip install --python .cache/qwen-asr-venv/bin/python 'qwen-asr==0.0.6' 'transformers==4.57.6' accelerate "
    f"'funasr>=1.4.16' torch torchaudio; expected {DEFAULT_PYTHON}"
)
DEFAULTS = {
    "vad_model": "funasr/fsmn-vad",
    "spk_model": "funasr/campplus",
    "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "device": "auto",
    "turns": "word",
}
TURN_MODES = ("word", "speaker")
# Qwen3-ASR takes language names, not tags. Unknown tags fall back to auto-detection.
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
}


def language_name(language: str) -> str | None:
    return LANGUAGE_NAMES.get(primary_language_tag(language))


def segments_from_payload(payload: dict[str, Any], *, turns: str) -> list[dict[str, Any]]:
    """Turn one worker JSON object into hypothesis segments (words or speaker turns)."""
    if "error" in payload:
        raise ProviderError(f"qwen-asr: {payload['error']}")
    key = "words" if turns == "word" else "turns"
    items = payload.get(key)
    if not isinstance(items, list):
        raise ProviderError(f"qwen-asr: worker response lacks a {key} list")
    kept: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        segment = make_segment(item.get("speaker"), item.get("start_s"), item.get("end_s"), item.get("text"))
        if segment is not None:
            kept.append(segment)
    return kept


class QwenASRProvider(Provider):
    key = "qwen-asr"
    option_names = frozenset({"python", "language_hint", *DEFAULTS})

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        raw = config.options.get("python", str(DEFAULT_PYTHON))
        self.python = Path(str(raw)).expanduser()
        self.settings = {name: str(config.options.get(name, default)) for name, default in DEFAULTS.items()}
        self.language_hint = bool(config.options.get("language_hint", True))
        self._process: subprocess.Popen[str] | None = None

    def _missing_python(self) -> str:
        return INSTALL_HINT if self.python == DEFAULT_PYTHON else f"python interpreter not found: {self.python}"

    def check(self) -> list[str]:
        problems = super().check()
        if not self.config.model:
            problems.append("model is required: a Hugging Face repo id such as Qwen/Qwen3-ASR-1.7B")
        if self.settings["turns"] not in TURN_MODES:
            problems.append(f"turns must be one of {', '.join(TURN_MODES)}; got {self.settings['turns']!r}")
        for name in ("vad_model", "spk_model", "forced_aligner"):
            if not self.settings[name]:
                problems.append(f"{name} must name a model (the speaker pipeline needs all three)")
        if not isinstance(self.config.options.get("language_hint", True), bool):
            problems.append("language_hint must be true or false")
        if not self.python.is_file():
            problems.append(self._missing_python())
        elif not WORKER.is_file():
            problems.append(f"qwen-asr worker script is missing: {WORKER}")
        return problems

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self.python.is_file():
            raise ProviderError(self._missing_python())
        argv = [str(self.python), "-u", str(WORKER), self.config.model]
        argv += [self.settings[name] for name in ("vad_model", "spk_model", "forced_aligner", "device")]
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._process = process
        assert process.stdout is not None
        ready = process.stdout.readline()
        if ready.strip() != "ready":
            code = process.poll()
            raise ProviderError(
                f"qwen-asr: worker for {self.config.model} exited before it was ready"
                + (f" (code {code})" if code is not None else "")
            )
        return process

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        audio = Path(audio)
        if not audio.is_file():
            raise ProviderError(f"audio file not found: {audio}")
        process = self._ensure_started()
        assert process.stdin is not None and process.stdout is not None
        request = {"audio": str(audio), "language": language_name(language) if self.language_hint else None}
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise ProviderError(f"qwen-asr: worker stopped while diarizing {conversation_id} (code {code})")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"qwen-asr: worker returned non-JSON for {conversation_id}: {line[:300]}") from exc
        if not isinstance(payload, dict):
            raise ProviderError(f"qwen-asr: worker returned a JSON {type(payload).__name__}, expected an object")
        turns = self.settings["turns"]
        segments = segments_from_payload(payload, turns=turns)
        if not segments:
            raise ProviderError(f"qwen-asr: pipeline returned no speaker-labelled {turns}s for {conversation_id}")
        return Hypothesis(
            segments=tuple(segments),
            granularity="word" if turns == "word" else "segment",
            duration_s=audio_info(audio)[0],
            raw={
                "model": self.config.model,
                "python": str(self.python),
                "pipeline": dict(self.settings),
                "device": payload.get("device"),
                "language": payload.get("language"),
                "language_hint": request["language"],
                "speakers": sorted({s["speaker"] for s in segments}),
                "turns": payload.get("turns"),
                "sentences": payload.get("sentences"),
            },
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=30)


__all__ = ["DEFAULTS", "QwenASRProvider", "language_name", "segments_from_payload"]
