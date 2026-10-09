"""Import hypotheses produced elsewhere from a directory of ``<conversation_id>.rttm`` files.

Use this to score a model that ran on another machine (a GPU box, a vendor's batch job)
without re-running it. RTTM lines look like::

    SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from converse_diarization.config import ModelConfig
from converse_diarization.providers.audio import audio_info
from converse_diarization.providers.base import Hypothesis, Provider, ProviderError, make_segment

GRANULARITIES = ("segment", "word")


def parse_rttm(text: str, *, source: str = "<rttm>") -> list[dict[str, Any]]:
    """Parse RTTM text into hypothesis segment dicts (``SPEAKER`` records only)."""
    segments: list[dict[str, Any]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(("#", ";;")):
            continue
        fields = line.split()
        if fields[0].upper() != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ProviderError(f"{source}:{number}: malformed RTTM line (expected at least 8 fields): {line[:120]}")
        try:
            start = float(fields[3])
            duration = float(fields[4])
        except ValueError:
            raise ProviderError(f"{source}:{number}: RTTM start/duration are not numbers: {line[:120]}") from None
        segment = make_segment(fields[7], start, start + duration)
        if segment is None:
            raise ProviderError(f"{source}:{number}: RTTM timing is unusable: {line[:120]}")
        segments.append(segment)
    segments.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
    return segments


class RttmProvider(Provider):
    key = "rttm"
    option_names = frozenset({"dir", "granularity"})

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        options = config.options
        directory = options.get("dir")
        self.dir: Path | None = Path(str(directory)).expanduser() if directory else None
        self.granularity = str(options.get("granularity", "segment"))

    def check(self) -> list[str]:
        problems = super().check()
        if self.dir is None:
            problems.append("dir is required: a directory of <conversation_id>.rttm files")
        elif not self.dir.is_dir():
            problems.append(f"dir {self.dir} does not exist or is not a directory")
        if self.granularity not in GRANULARITIES:
            problems.append(f"granularity must be one of {', '.join(GRANULARITIES)}; got {self.granularity!r}")
        return problems

    def path_for(self, conversation_id: str) -> Path:
        if self.dir is None:
            raise ProviderError("rttm: dir option is required")
        return self.dir / f"{conversation_id}.rttm"

    def diarize(self, audio: Path, *, language: str, conversation_id: str) -> Hypothesis:
        path = self.path_for(conversation_id)
        if not path.is_file():
            raise ProviderError(f"rttm: no hypothesis file for {conversation_id}: {path}", retryable=False)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProviderError(f"rttm: cannot read {path}: {exc}") from exc
        segments = parse_rttm(text, source=str(path))
        if not segments:
            raise ProviderError(f"rttm: {path} contains no SPEAKER records")
        duration_s: float | None
        try:
            duration_s = audio_info(audio)[0]
        except ProviderError:
            duration_s = None
        granularity = "word" if self.granularity == "word" else "segment"
        raw = {"path": str(path), "speakers": sorted({segment["speaker"] for segment in segments})}
        return Hypothesis(segments=tuple(segments), granularity=granularity, duration_s=duration_s, raw=raw)


__all__ = ["RttmProvider", "parse_rttm"]
