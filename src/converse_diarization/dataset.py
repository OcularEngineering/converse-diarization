"""Materialize Ocular AI conversations into a local, self-describing dataset folder.

Ocular AI publishes full-duplex two-speaker conversations as a Hugging Face
AudioFolder: ``train/metadata.jsonl`` holds one row per speaker track and each
row's ``file_name`` points at a FLAC file next to it. Every FLAC is one
speaker's own microphone for the whole conversation, so a session is two
tracks that overlap in time but did not start at the exact same instant.

:func:`pull_dataset` (Hugging Face) and :func:`build_from_folder` (local
AudioFolder) both produce the same on-disk layout::

    <root>/
      manifest.json
      audio/
        <conversation_id>/
          ch1.flac
          ch2.flac

``manifest.json`` is the only thing the rest of the harness reads::

    {
      "schema": 1,
      "source": {"kind": "huggingface", "repo_id": "...", "revision": "...", ...},
      "created_at": "2026-10-07T12:00:00Z",
      "conversations": [
        {
          "id": "<session_id>",
          "language": "en",
          "locale": "en-US",
          "duration_s": 912.3,
          "tracks": [
            {
              "id": "ch1",
              "speaker_id": "...",
              "path": "audio/<session_id>/ch1.flac",
              "sha256": "...",
              "duration_s": 910.8,
              "sample_rate": 48000,
              "offset_s": 0.0,
              "capture_ms": 1757534367463,
              "source_file": "1757534367463-ab12cd-hifi-audio-1757534367463.flac",
              "reference": "full-track transcript with <breath/> style tags left in",
              "words": [{"text": "hello", "start_s": 0.42, "end_s": 0.71}, ...],
              "metadata": {"accent_self_reported": "...", ...}
            }
          ]
        }
      ]
    }

Tracks inside a conversation are ordered by their capture time (then
``speaker_id``) and named ``ch1``, ``ch2``, ... . ``offset_s`` is how long
after the earliest track in the session this track started recording; word
timestamps in the manifest are already shifted onto that shared conversation
timeline. Audio is copied byte-for-byte and never resampled or padded.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile

DEFAULT_REPO_ID = "OcularAIInc/AMERICAN-ENGLISH-TRANSCRIBED-HIFI-FULL-DUPLEX-TWO-SPEAKER-CONVERSATIONAL-DATASET-SAMPLE"
DEFAULT_REVISION = "29b74ad449eb99a1f9937dc0f75d35333ca866d5"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1

_METADATA_KEYS = (
    "accent_self_reported",
    "age_range",
    "gender_self_reported",
    "native_language",
    "take_id",
    "datetime_utc",
)
_REQUIRED_ROW_KEYS = ("session_id", "file_name", "speaker_id")
_CAPTURE_MS = re.compile(r"-(\d+)\.[A-Za-z0-9]+$")
_CHUNK = 1024 * 1024
_DURATION_TOLERANCE_S = 0.05


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start_s": self.start_s, "end_s": self.end_s}

    @classmethod
    def from_dict(cls, data: dict) -> Word:
        return cls(text=str(data["text"]), start_s=float(data["start_s"]), end_s=float(data["end_s"]))


@dataclass(frozen=True)
class Track:
    id: str
    speaker_id: str
    path: str
    sha256: str
    duration_s: float
    sample_rate: int
    offset_s: float
    capture_ms: int | None
    source_file: str
    reference: str
    words: tuple[Word, ...] = ()
    metadata: dict = field(default_factory=dict)

    def audio_path(self, root: Path) -> Path:
        return Path(root) / Path(*self.path.split("/"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "path": self.path,
            "sha256": self.sha256,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "offset_s": self.offset_s,
            "capture_ms": self.capture_ms,
            "source_file": self.source_file,
            "reference": self.reference,
            "words": [word.to_dict() for word in self.words],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Track:
        capture_ms = data.get("capture_ms")
        return cls(
            id=str(data["id"]),
            speaker_id=str(data["speaker_id"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            duration_s=float(data["duration_s"]),
            sample_rate=int(data["sample_rate"]),
            offset_s=float(data.get("offset_s", 0.0)),
            capture_ms=None if capture_ms is None else int(capture_ms),
            source_file=str(data.get("source_file", "")),
            reference=str(data.get("reference", "")),
            words=tuple(Word.from_dict(item) for item in data.get("words", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Conversation:
    id: str
    language: str
    locale: str | None
    duration_s: float
    tracks: tuple[Track, ...]

    def track(self, track_id: str) -> Track:
        for track in self.tracks:
            if track.id == track_id:
                return track
        available = ", ".join(track.id for track in self.tracks) or "<none>"
        raise DatasetError(f"conversation {self.id!r} has no track {track_id!r} (available: {available})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "locale": self.locale,
            "duration_s": self.duration_s,
            "tracks": [track.to_dict() for track in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Conversation:
        return cls(
            id=str(data["id"]),
            language=str(data.get("language") or "en"),
            locale=data.get("locale"),
            duration_s=float(data["duration_s"]),
            tracks=tuple(Track.from_dict(item) for item in data.get("tracks", [])),
        )


@dataclass(frozen=True)
class Dataset:
    root: Path
    source: dict
    created_at: str
    conversations: tuple[Conversation, ...]

    @property
    def tracks(self) -> list[tuple[Conversation, Track]]:
        return [(conversation, track) for conversation in self.conversations for track in conversation.tracks]

    def conversation(self, conversation_id: str) -> Conversation:
        for conversation in self.conversations:
            if conversation.id == conversation_id:
                return conversation
        raise DatasetError(f"dataset at {self.root} has no conversation {conversation_id!r}")

    def digest(self) -> str:
        """Stable 16-hex-char identity of the audio content: which tracks exist and their checksums."""
        triples = sorted((conversation.id, track.id, track.sha256) for conversation, track in self.tracks)
        payload = json.dumps(triples, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source": dict(self.source),
            "created_at": self.created_at,
            "conversations": [conversation.to_dict() for conversation in self.conversations],
        }


# ---------------------------------------------------------------------------
# Helpers


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file, skipping blank lines."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"metadata file not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{line_number}: expected a JSON object, got {type(row).__name__}")
            rows.append(row)
    return rows


def capture_ms_from_name(file_name: str) -> int | None:
    """Return the trailing ``<capture_ms>`` of ``<capture_ms>-<speaker>-hifi-audio-<capture_ms>.flac``."""
    match = _CAPTURE_MS.search(Path(file_name).name)
    if not match:
        return None
    return int(match.group(1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _audio_info(path: Path) -> tuple[float, int]:
    try:
        info = soundfile.info(str(path))
    except Exception as exc:  # soundfile raises its own RuntimeError/LibsndfileError
        raise DatasetError(f"cannot read audio file {path}: {exc}") from exc
    if not info.samplerate:
        raise DatasetError(f"audio file {path} reports a zero sample rate")
    return info.frames / info.samplerate, int(info.samplerate)


def _reference_text(row: dict) -> str:
    transcript = row.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        return transcript.strip()
    segments = row.get("segments") or []
    ordered = sorted(
        (segment for segment in segments if isinstance(segment, dict)),
        key=lambda segment: (segment.get("index") is None, segment.get("index", 0)),
    )
    parts = [str(segment.get("text", "")).strip() for segment in ordered]
    return " ".join(part for part in parts if part)


def _words(row: dict, offset_s: float) -> tuple[Word, ...]:
    words: list[Word] = []
    for segment in row.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        for item in segment.get("words") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("word", "")).strip()
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not text or end <= start:
                continue
            words.append(Word(text=text, start_s=round(start + offset_s, 3), end_s=round(end + offset_s, 3)))
    words.sort(key=lambda word: (word.start_s, word.end_s))
    return tuple(words)


def _validate_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetError(f"row {index}: expected a JSON object, got {type(row).__name__}")
        missing = [key for key in _REQUIRED_ROW_KEYS if not row.get(key)]
        if missing:
            raise DatasetError(f"row {index}: missing required field(s): {', '.join(missing)}")


def _copy_audio(src: Path, dst: Path) -> str:
    """Copy ``src`` to ``dst`` unless an identical file is already there. Returns the sha256."""
    src = Path(src)
    if not src.is_file():
        raise DatasetError(f"audio file not found: {src}")
    source_sha = sha256_file(src)
    if dst.is_file() and sha256_file(dst) == source_sha:
        return source_sha
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return source_sha


def _write_manifest(dataset: Dataset) -> None:
    dataset.root.mkdir(parents=True, exist_ok=True)
    manifest = dataset.root / MANIFEST_NAME
    text = json.dumps(dataset.to_dict(), indent=2, sort_keys=False, ensure_ascii=False)
    manifest.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Building


def build_dataset(rows: list[dict], audio_for: Callable[[str], Path], output: Path, *, source: dict) -> Dataset:
    """Turn AudioFolder rows into a dataset folder at ``output``.

    ``audio_for(file_name)`` must return the local path of the FLAC referenced by a row.
    """
    output = Path(output)
    if not rows:
        raise DatasetError("no rows to build a dataset from")
    _validate_rows(rows)

    sessions: dict[str, list[dict]] = {}
    for row in rows:
        sessions.setdefault(str(row["session_id"]), []).append(row)

    conversations: list[Conversation] = []
    for session_id in sorted(sessions):
        session_rows = sessions[session_id]
        captures = [capture_ms_from_name(str(row["file_name"])) for row in session_rows]
        known = [value for value in captures if value is not None]
        base_ms = min(known) if known else None

        order = sorted(
            range(len(session_rows)),
            key=lambda i: (
                captures[i] is None,
                captures[i] if captures[i] is not None else 0,
                str(session_rows[i]["speaker_id"]),
            ),
        )

        tracks: list[Track] = []
        for channel, i in enumerate(order, start=1):
            row = session_rows[i]
            capture_ms = captures[i]
            offset_s = 0.0 if capture_ms is None or base_ms is None else (capture_ms - base_ms) / 1000.0
            file_name = str(row["file_name"])
            track_id = f"ch{channel}"
            rel_path = f"audio/{session_id}/{track_id}.flac"
            destination = output / "audio" / session_id / f"{track_id}.flac"

            sha256 = _copy_audio(Path(audio_for(file_name)), destination)
            duration_s, sample_rate = _audio_info(destination)

            tracks.append(
                Track(
                    id=track_id,
                    speaker_id=str(row["speaker_id"]),
                    path=rel_path,
                    sha256=sha256,
                    duration_s=round(duration_s, 6),
                    sample_rate=sample_rate,
                    offset_s=round(offset_s, 3),
                    capture_ms=capture_ms,
                    source_file=file_name,
                    reference=_reference_text(row),
                    words=_words(row, offset_s),
                    metadata={key: row[key] for key in _METADATA_KEYS if key in row},
                )
            )

        first = session_rows[order[0]]
        conversations.append(
            Conversation(
                id=session_id,
                language=str(first.get("language") or "en"),
                locale=first.get("language_locale"),
                duration_s=round(max(track.offset_s + track.duration_s for track in tracks), 6),
                tracks=tuple(tracks),
            )
        )

    dataset = Dataset(
        root=output,
        source=dict(source),
        created_at=_utc_now(),
        conversations=tuple(conversations),
    )
    _write_manifest(dataset)
    return dataset


def pull_dataset(
    output: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    token: str | None = None,
) -> Dataset:
    """Download an Ocular AI AudioFolder dataset from the Hugging Face Hub and build it at ``output``."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    def download(filename: str) -> Path:
        print(f"pull {repo_id}@{revision[:12]}: {filename}", file=sys.stderr)
        try:
            local = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, token=token)
        except EntryNotFoundError:
            raise
        except Exception as exc:
            raise DatasetError(f"failed to download {filename} from {repo_id}@{revision}: {exc}") from exc
        return Path(local)

    metadata_path = "train/metadata.jsonl"
    try:
        metadata_local = download(metadata_path)
    except EntryNotFoundError:
        metadata_path = "metadata.jsonl"
        try:
            metadata_local = download(metadata_path)
        except EntryNotFoundError as exc:
            raise DatasetError(f"{repo_id}@{revision} has neither train/metadata.jsonl nor metadata.jsonl") from exc

    prefix = metadata_path[: -len("metadata.jsonl")]
    rows = read_jsonl(metadata_local)
    source = {
        "kind": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_file(metadata_local),
    }

    def audio_for(file_name: str) -> Path:
        try:
            return download(prefix + file_name)
        except EntryNotFoundError as exc:
            raise DatasetError(f"{repo_id}@{revision}: audio file {prefix + file_name!r} not found") from exc

    return build_dataset(rows, audio_for, Path(output), source=source)


def build_from_folder(source_dir: Path, output: Path) -> Dataset:
    """Build a dataset from a local AudioFolder (``metadata.jsonl`` at the root or under ``train/``)."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise DatasetError(f"source folder not found: {source_dir}")
    candidates = (source_dir / "metadata.jsonl", source_dir / "train" / "metadata.jsonl")
    metadata = next((path for path in candidates if path.is_file()), None)
    if metadata is None:
        raise DatasetError(f"{source_dir} has neither metadata.jsonl nor train/metadata.jsonl")

    rows = read_jsonl(metadata)
    base = metadata.parent
    source = {
        "kind": "folder",
        "path": str(source_dir.resolve()),
        "metadata_path": metadata.relative_to(source_dir).as_posix(),
        "metadata_sha256": sha256_file(metadata),
    }
    return build_dataset(rows, lambda file_name: base / Path(*file_name.split("/")), Path(output), source=source)


# ---------------------------------------------------------------------------
# Loading and checking


def load_dataset(root: Path) -> Dataset:
    """Read ``manifest.json`` from a dataset folder."""
    root = Path(root)
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        raise DatasetError(f"no {MANIFEST_NAME} in {root}; run the pull command first")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{manifest}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetError(f"{manifest}: expected a JSON object")
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        raise DatasetError(f"{manifest}: unsupported schema {schema!r} (expected {SCHEMA_VERSION})")
    try:
        conversations = tuple(
            sorted(
                (Conversation.from_dict(item) for item in data.get("conversations", [])),
                key=lambda conversation: conversation.id,
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"{manifest}: malformed conversation entry: {exc}") from exc
    return Dataset(
        root=root,
        source=dict(data.get("source") or {}),
        created_at=str(data.get("created_at", "")),
        conversations=conversations,
    )


def verify_dataset(dataset: Dataset) -> list[str]:
    """Check that every track's audio exists, matches its checksum and has the recorded duration."""
    problems: list[str] = []
    for conversation, track in dataset.tracks:
        label = f"{conversation.id}/{track.id}"
        path = track.audio_path(dataset.root)
        if not path.is_file():
            problems.append(f"{label}: missing audio file {path}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != track.sha256:
            problems.append(f"{label}: sha256 mismatch (manifest {track.sha256[:12]}..., file {actual_sha[:12]}...)")
        try:
            duration_s, sample_rate = _audio_info(path)
        except DatasetError as exc:
            problems.append(f"{label}: {exc}")
            continue
        if abs(duration_s - track.duration_s) > _DURATION_TOLERANCE_S:
            problems.append(f"{label}: duration mismatch (manifest {track.duration_s:.3f} s, file {duration_s:.3f} s)")
        if sample_rate != track.sample_rate:
            problems.append(f"{label}: sample rate mismatch (manifest {track.sample_rate}, file {sample_rate})")
    return problems


__all__ = [
    "DEFAULT_REPO_ID",
    "DEFAULT_REVISION",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "Conversation",
    "Dataset",
    "DatasetError",
    "Track",
    "Word",
    "build_dataset",
    "build_from_folder",
    "capture_ms_from_name",
    "load_dataset",
    "pull_dataset",
    "read_jsonl",
    "sha256_file",
    "verify_dataset",
]
