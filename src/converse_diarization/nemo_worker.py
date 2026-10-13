"""Long-running NeMo Sortformer process. One audio path per stdin line, one JSON object per stdout line.

Run this with the NeMo interpreter (``.cache/nemotron-venv``), not the harness environment.
``nemo-toolkit[asr]`` pins its own torch, lightning and hydra stack.

argv: model, then streaming parameters in 80 ms frames:
    spkcache_len fifo_len chunk_len chunk_right_context spkcache_update_period
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This file lives inside the package tree. Drop that directory from sys.path so a
# local module cannot shadow the standard library (providers/http.py vs http.client).
_here = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if entry not in {"", _here} and str(Path(entry).resolve()) != _here]


def main() -> int:
    model_name = sys.argv[1]
    spkcache_len, fifo_len, chunk_len, right_context, update_period = (int(value) for value in sys.argv[2:7])
    stdout = sys.stdout
    sys.stdout = sys.stderr  # library prints must not reach the JSON channel

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    if Path(model_name).is_file():
        model = SortformerEncLabelModel.restore_from(restore_path=model_name, map_location="cpu", strict=False)
    else:
        model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    model.eval()
    modules = model.sortformer_modules
    modules.spkcache_len = spkcache_len
    modules.fifo_len = fifo_len
    modules.chunk_len = chunk_len
    modules.chunk_right_context = right_context
    modules.spkcache_update_period = update_period
    model._check_streaming_parameters()

    stdout.write("ready\n")
    stdout.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            with torch.inference_mode():
                predicted = model.diarize(audio=[path], batch_size=1)
            segments = []
            for item in predicted[0]:
                # NeMo returns "begin end speaker_index" strings.
                begin, end, speaker = str(item).split()
                start_s, end_s = float(begin), float(end)
                if end_s < start_s or start_s < 0:
                    continue
                segments.append(
                    {"speaker": str(speaker), "start_s": round(start_s, 3), "end_s": round(end_s, 3), "text": None}
                )
            payload = {"segments": segments}
        except Exception as exc:  # the harness records this and continues
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
