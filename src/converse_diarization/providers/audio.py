"""Audio inspection and upload preparation (FLAC passthrough, 16 kHz WAV, or MP3 via ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from converse_diarization.providers.base import ProviderError

UPLOAD_FORMATS = ("flac", "wav16k", "mp3")
_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}
_TARGET_RATE = 16_000


def content_type_for(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    try:
        return _CONTENT_TYPES[suffix]
    except KeyError:
        raise ProviderError(f"unsupported audio type {suffix!r}; expected one of {', '.join(_CONTENT_TYPES)}") from None


def audio_info(path: Path | str) -> tuple[float, int]:
    """Return ``(duration_s, sample_rate)`` without decoding the whole file."""
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ProviderError(f"cannot read audio header of {Path(path).name}: {exc}") from exc
    return float(info.duration), int(info.samplerate)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _workdir(workdir: Path | None) -> Path:
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir
    return Path(tempfile.mkdtemp(prefix="converse-diarization-"))


def _to_wav16k(src: Path, dst: Path) -> None:
    try:
        data, rate = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ProviderError(f"cannot decode {src.name}: {exc}") from exc
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != _TARGET_RATE and mono.size:
        duration = mono.size / float(rate)
        count = max(1, int(round(duration * _TARGET_RATE)))
        src_t = np.arange(mono.size, dtype=np.float64) / float(rate)
        dst_t = np.arange(count, dtype=np.float64) / float(_TARGET_RATE)
        mono = np.interp(dst_t, src_t, mono.astype(np.float64)).astype(np.float32)
    mono = np.clip(mono, -1.0, 1.0)
    sf.write(str(dst), mono, _TARGET_RATE, subtype="PCM_16", format="WAV")


def _to_mp3(src: Path, dst: Path) -> None:
    if not ffmpeg_available():
        raise ProviderError(
            "ffmpeg is required to prepare MP3 uploads but was not found on PATH; "
            'install it (e.g. `brew install ffmpeg`) or set upload = "flac" / "wav16k" in the model config'
        )
    command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000"]
    command += ["-b:a", "64k", str(dst)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderError(f"ffmpeg failed while encoding {src.name}: {exc}") from exc
    if completed.returncode != 0 or not dst.is_file():
        detail = " ".join(completed.stderr.split())[:300]
        raise ProviderError(f"ffmpeg exited with status {completed.returncode} encoding {src.name}: {detail}")


def prepare_upload(path: Path, *, fmt: str, workdir: Path | None = None) -> tuple[Path, str]:
    """Return ``(upload_path, content_type)`` for ``path`` encoded as ``fmt``.

    ``"flac"`` returns the original file untouched. ``"wav16k"`` writes mono 16 kHz PCM16 WAV
    and ``"mp3"`` writes mono 16 kHz 64 kbps MP3 with ffmpeg. Converted files are placed in
    ``workdir`` or a fresh ``converse-diarization-*`` temporary directory.
    """
    path = Path(path)
    if fmt not in UPLOAD_FORMATS:
        raise ProviderError(f"unknown upload format {fmt!r}; expected one of {', '.join(UPLOAD_FORMATS)}")
    if not path.is_file():
        raise ProviderError(f"audio file not found: {path}")
    if fmt == "flac":
        return path, content_type_for(path)
    target_dir = _workdir(workdir)
    if fmt == "wav16k":
        dst = target_dir / f"{path.stem}.16k.wav"
        _to_wav16k(path, dst)
        return dst, "audio/wav"
    dst = target_dir / f"{path.stem}.mp3"
    _to_mp3(path, dst)
    return dst, "audio/mpeg"


__all__ = ["UPLOAD_FORMATS", "audio_info", "content_type_for", "ffmpeg_available", "prepare_upload"]
