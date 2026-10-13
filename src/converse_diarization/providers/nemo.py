"""Local NVIDIA NeMo Sortformer diarization (Nemotron 3 Diarization, Streaming Sortformer).

``SortformerEncLabelModel.diarize`` returns arrival-ordered speaker turns, scored as
segments. NeMo pins its own torch / lightning / hydra stack, so inference runs in a
separate interpreter named by the ``python`` option (default
``.cache/nemotron-venv/bin/python``). Streaming parameters default to the published
offline profile for Nemotron 3 Diarization, all in 80 ms frames.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError

DEFAULT_PYTHON = Path(".cache/nemotron-venv/bin/python")
WORKER = Path(__file__).resolve().parents[1] / "nemo_worker.py"
INSTALL_HINT = (
    "NeMo interpreter not found. It needs its own environment: uv venv .cache/nemotron-venv --python 3.12; "
    "uv pip install --python .cache/nemotron-venv/bin/python Cython packaging "
    "'nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git' "
    f"(the PyPI 3.0.0 release predates the RoPE encoder this checkpoint uses); expected {DEFAULT_PYTHON}"
)

# "Very high latency (offline)" row of the Nemotron 3 Diarization model card: 30.4 s input buffer.
OFFLINE_PROFILE = {
    "spkcache_len": 264,
    "fifo_len": 40,
    "chunk_len": 340,
    "chunk_right_context": 40,
    "spkcache_update_period": 300,
}


def segments_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one worker JSON object into hypothesis segments."""
    if "error" in payload:
        raise ProviderError(f"nemo: {payload['error']}")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ProviderError("nemo: worker response lacks a segments list")
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


class NemoProvider(Provider):
    key = "nemo"
    option_names = frozenset({"python", *OFFLINE_PROFILE})

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        raw = config.options.get("python", str(DEFAULT_PYTHON))
        self.python = Path(str(raw)).expanduser()
        self.profile = {name: config.options.get(name, default) for name, default in OFFLINE_PROFILE.items()}
        self._process: subprocess.Popen[str] | None = None

    def _missing_python(self) -> str:
        return INSTALL_HINT if self.python == DEFAULT_PYTHON else f"python interpreter not found: {self.python}"

    def check(self) -> list[str]:
        problems = super().check()
        if not self.config.model:
            problems.append("model is required: a Hugging Face repo id (nvidia/Nemotron-3-Diarization) or a .nemo file")
        for name, value in self.profile.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                problems.append(f"{name} must be a non-negative integer number of 80 ms frames; got {value!r}")
        if not self.python.is_file():
            problems.append(self._missing_python())
        elif not WORKER.is_file():
            problems.append(f"nemo worker script is missing: {WORKER}")
        return problems

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self.python.is_file():
            raise ProviderError(self._missing_python())
        argv = [str(self.python), "-u", str(WORKER), self.config.model]
        argv += [str(int(self.profile[name])) for name in OFFLINE_PROFILE]
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._process = process
        assert process.stdout is not None
        ready = process.stdout.readline()
        if ready.strip() != "ready":
            code = process.poll()
            raise ProviderError(
                f"nemo: worker for {self.config.model} exited before it was ready"
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
            raise ProviderError(f"nemo: worker stopped while diarizing {conversation_id} (code {code})")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"nemo: worker returned non-JSON for {conversation_id}: {line[:300]}") from exc
        if not isinstance(payload, dict):
            raise ProviderError(f"nemo: worker returned a JSON {type(payload).__name__}, expected an object")
        segments = segments_from_payload(payload)
        if not segments:
            raise ProviderError(f"nemo: model returned no speaker turns for {conversation_id}")
        return Hypothesis(
            segments=tuple(segments),
            granularity="segment",
            duration_s=audio_info(audio)[0],
            raw={
                "model": self.config.model,
                "python": str(self.python),
                "streaming": dict(self.profile),
                "speakers": sorted({s["speaker"] for s in segments}),
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


__all__ = ["NemoProvider", "OFFLINE_PROFILE", "segments_from_payload"]
