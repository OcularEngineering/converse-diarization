"""Minimal .env loading and secret hygiene helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")
_SECRET_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"omi_sk_[A-Za-z0-9_-]{8,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
)


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_dotenv(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Load ``KEY=value`` pairs from ``.env`` into ``os.environ``. Returns the keys that were set."""
    path = path or Path.cwd() / ".env"
    if not path.is_file():
        return []
    loaded: list[str] = []
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        if not value:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def redact(text: str) -> str:
    """Mask anything that looks like an API key before it is written to disk or stderr."""
    for pattern in _SECRET_SHAPES:
        text = pattern.sub("[redacted]", text)
    for key, value in os.environ.items():
        if value and len(value) >= 12 and ("KEY" in key or "TOKEN" in key or "SECRET" in key):
            text = text.replace(value, "[redacted]")
    return text
