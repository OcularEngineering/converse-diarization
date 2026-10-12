"""Local DiariZen speaker diarization.

DiariZen (``DiariZenPipeline.from_pretrained``) returns overlapping speaker turns,
scored as segments. Its pipeline subclasses a pyannote 3.1 fork, so inference runs
in a separate interpreter named by the ``python`` option. The default is
``.cache/diarizen-venv/bin/python``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError

DEFAULT_PYTHON = Path(".cache/diarizen-venv/bin/python")
WORKER = Path(__file__).resolve().parents[1] / "diarizen_worker.py"
INSTALL_HINT = (
    "DiariZen interpreter not found. It needs its own environment because it vendors pyannote 3.1; "
    f"expected {DEFAULT_PYTHON}"
)


def segments_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one worker JSON object into hypothesis segments."""
    if "error" in payload:
        raise ProviderError(f"diarizen: {payload['error']}")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ProviderError("diarizen: worker response lacks a segments list")
    kept: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict) or item.get("speaker") is None:
            continue
        try:
            start, end = float(item["start_s"]), float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start or start < 0:
            continue
        kept.append({"speaker": str(item["speaker"]), "start_s": round(start, 3), "end_s": round(end, 3), "text": None})
    return kept


class DiariZenProvider(Provider):
    key = "diarizen"
    option_names = frozenset({"python"})

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        raw = config.options.get("python", str(DEFAULT_PYTHON))
        self.python = Path(str(raw)).expanduser()
        self._process: subprocess.Popen[str] | None = None

    def check(self) -> list[str]:
        problems = super().check()
        if not self.config.model:
            problems.append("model is required: a Hugging Face repo id such as BUT-FIT/diarizen-wavlm-large-s80-md-v2")
        if not self.python.is_file():
            problems.append(INSTALL_HINT if self.python == DEFAULT_PYTHON else f"python interpreter not found: {self.python}")
        elif not WORKER.is_file():
            problems.append(f"diarizen worker script is missing: {WORKER}")
        return problems

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self.python.is_file():
            raise ProviderError(INSTALL_HINT if self.python == DEFAULT_PYTHON else f"python interpreter not found: {self.python}")
        process = subprocess.Popen(
            [str(self.python), "-u", str(WORKER), self.config.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process
        assert process.stdout is not None
        ready = process.stdout.readline()
        if ready.strip() != "ready":
            code = process.poll()
            raise ProviderError(
                f"diarizen: worker for {self.config.model} exited before it was ready"
                + (f" (code {code})" if code is not None else "")
            )
        return process

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        audio = Path(audio)
        if not audio.is_file():
            raise ProviderError(f"audio file not found: {audio}")
        process = self._ensure_started()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(f"{audio}\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise ProviderError(f"diarizen: worker stopped while diarizing {conversation_id} (code {code})")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"diarizen: worker returned non-JSON for {conversation_id}: {line[:300]}") from exc
        if not isinstance(payload, dict):
            raise ProviderError(f"diarizen: worker returned a JSON {type(payload).__name__}, expected an object")
        segments = segments_from_payload(payload)
        if not segments:
            raise ProviderError(f"diarizen: pipeline returned no speaker turns for {conversation_id}")
        return Hypothesis(
            segments=tuple(segments),
            granularity="segment",
            duration_s=audio_info(audio)[0],
            raw={"model": self.config.model, "python": str(self.python), "speakers": sorted({s["speaker"] for s in segments})},
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=30)


__all__ = ["DiariZenProvider", "segments_from_payload"]
