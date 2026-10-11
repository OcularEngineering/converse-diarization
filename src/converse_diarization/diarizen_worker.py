"""Long-running DiariZen process. One audio path per stdin line, one JSON object per stdout line.

Run this with the DiariZen interpreter (``.cache/diarizen-venv``), not the harness
environment. DiariZen ships a pyannote 3.1 fork that cannot share the harness env.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This file lives inside the package tree. Drop that directory from sys.path so a
# local module cannot shadow the standard library (providers/http.py vs http.client).
_here = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if entry not in {"", _here} and str(Path(entry).resolve()) != _here]

import torch

_load = torch.load


def _load_trusted(*args, **kwargs):
    # Official Hugging Face checkpoints predate PyTorch's weights_only default.
    kwargs["weights_only"] = False
    return _load(*args, **kwargs)


torch.load = _load_trusted

from diarizen.pipelines.inference import DiariZenPipeline  # noqa: E402


def main() -> int:
    model = sys.argv[1]
    stdout = sys.stdout
    sys.stdout = sys.stderr
    pipeline = DiariZenPipeline.from_pretrained(model)
    stdout.write("ready\n")
    stdout.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            annotation = pipeline(path, sess_name="session")
            segments = []
            for turn, _track, speaker in annotation.itertracks(yield_label=True):
                start, end = float(turn.start), float(turn.end)
                if end < start or start < 0:
                    continue
                segments.append(
                    {"speaker": str(speaker), "start_s": round(start, 3), "end_s": round(end, 3), "text": None}
                )
            payload = {"segments": segments}
        except Exception as exc:  # the harness records this and continues
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
