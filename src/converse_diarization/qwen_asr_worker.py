"""Long-running FunASR process for the Qwen3-ASR speaker pipeline.

One JSON request per stdin line (``{"audio": path, "language": "English" | null}``), one
JSON object per stdout line. Run this with the ``.cache/qwen-asr-venv`` interpreter: the
``qwen-asr`` package pins transformers, and FunASR brings its own VAD / speaker stack.

argv: model vad_model spk_model forced_aligner device

The pipeline is FunASR ``AutoModel``: FSMN-VAD speech segments -> Qwen3-ASR transcription
-> Qwen3-ForcedAligner word timestamps -> CAM++ speaker embeddings on 1.5 s windows,
clustered into speaker turns. FunASR only exposes the turns indirectly (it labels whole
sentences with them), so ``distribute_spk`` is wrapped to capture the turn timeline.

FunASR 1.4.x reads the aligner's ``start_time`` / ``end_time`` as milliseconds and truncates
them with ``int()``, but ``qwen-asr`` reports seconds, which would flatten every word onto a
whole second. ``transcribe`` is wrapped to hand FunASR millisecond values instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# This file lives inside the package tree. Drop that directory from sys.path so a
# local module cannot shadow the standard library (providers/http.py vs http.client).
_here = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if entry not in {"", _here} and str(Path(entry).resolve()) != _here]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _timestamps(result: dict[str, Any]) -> list[tuple[float, float]]:
    """FunASR's merged ``timestamp`` list (``[[start_ms, end_ms], ...]``) as seconds."""
    items: list[tuple[float, float]] = []
    raw = result.get("timestamp")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    items.append((float(item[0]) / 1000.0, float(item[1]) / 1000.0))
                except (TypeError, ValueError):
                    continue
    return items


def _milliseconds(transcribe):
    """Wrap ``Qwen3ASRModel.transcribe`` so aligned spans reach FunASR in milliseconds."""

    def wrapped(*args, **kwargs):
        results = transcribe(*args, **kwargs)
        for result in results:
            stamps = getattr(result, "time_stamps", None)
            if stamps is None:
                continue
            items = [
                type(item)(
                    text=item.text,
                    start_time=round(float(item.start_time) * 1000.0),
                    end_time=round(float(item.end_time) * 1000.0),
                )
                for item in stamps.items
            ]
            result.time_stamps = type(stamps)(items=items)
        return results

    return wrapped


def _speaker_for(mid: float, turns: list[list[float]]) -> int | None:
    best, best_gap = None, None
    for start, end, label in turns:
        if start <= mid <= end:
            return int(label)
        gap = start - mid if mid < start else mid - end
        if best_gap is None or gap < best_gap:
            best, best_gap = int(label), gap
    return best


def main() -> int:
    model_name, vad_model, spk_model, forced_aligner, requested_device = sys.argv[1:6]
    stdout = sys.stdout
    sys.stdout = sys.stderr  # library prints must not reach the JSON channel

    import funasr.auto.auto_model as auto_model
    from funasr import AutoModel

    captured: dict[str, list[list[float]]] = {}
    original_distribute = auto_model.distribute_spk

    def distribute_spk(sentence_list, sv_output):
        captured["turns"] = [[float(s), float(e), int(label)] for s, e, label in sv_output]
        return original_distribute(sentence_list, sv_output)

    auto_model.distribute_spk = distribute_spk

    device = _pick_device(requested_device)
    model = AutoModel(
        model=model_name,
        hub="hf",
        vad_model=vad_model,
        spk_model=spk_model,
        forced_aligner=forced_aligner,
        device=device,
        disable_update=True,
        disable_pbar=True,
    )
    asr = model.model.qwen3_asr_model
    asr.transcribe = _milliseconds(asr.transcribe)

    stdout.write("ready\n")
    stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            captured.clear()
            generate_kwargs: dict[str, Any] = {"input": request["audio"], "batch_size_s": 300}
            if request.get("language"):
                generate_kwargs["language"] = request["language"]
            results = model.generate(**generate_kwargs)
            result = results[0] if results else {}
            turns = captured.get("turns", [])
            words = []
            for start_s, end_s in _timestamps(result):
                if end_s < start_s or start_s < 0:
                    continue
                speaker = _speaker_for((start_s + end_s) / 2.0, turns)
                if speaker is None:
                    continue
                words.append(
                    {"speaker": str(speaker), "start_s": round(start_s, 3), "end_s": round(end_s, 3), "text": None}
                )
            sentences = [
                {
                    "speaker": str(item.get("spk")) if item.get("spk") is not None else None,
                    "start_s": round(float(item.get("start", 0)) / 1000.0, 3),
                    "end_s": round(float(item.get("end", 0)) / 1000.0, 3),
                    "text": item.get("text"),
                }
                for item in result.get("sentence_info", [])
                if isinstance(item, dict)
            ]
            payload = {
                "device": device,
                "language": result.get("language"),
                "text": result.get("text"),
                "turns": [
                    {"speaker": str(label), "start_s": round(s, 3), "end_s": round(e, 3)} for s, e, label in turns
                ],
                "words": words,
                "sentences": sentences,
            }
        except Exception as exc:  # the harness records this and continues
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
