"""Ocular AI Converse-Diarization: diarize mixed two-speaker conversations and score DER, JER and speaker count.

Part of the Ocular AI Converse benchmark family. Released under the MIT License.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("converse-diarization")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
