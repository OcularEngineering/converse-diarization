from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from converse_diarization.dataset import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    DatasetError,
    build_dataset,
    build_from_folder,
    capture_ms_from_name,
    load_dataset,
    read_jsonl,
    verify_dataset,
)

SAMPLE_RATE = 16_000
BASE_MS = 1_757_534_367_463


def write_flac(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.zeros(int(SAMPLE_RATE * seconds), dtype="float32"), SAMPLE_RATE, format="FLAC")
    return path


def track_name(capture_ms: int, speaker: str) -> str:
    return f"{capture_ms}-{speaker}-hifi-audio-{capture_ms}.flac"


def make_row(session_id: str, speaker: str, capture_ms: int, **extra) -> dict:
    row = {
        "session_id": session_id,
        "speaker_id": speaker,
        "file_name": track_name(capture_ms, speaker),
        "language": "en",
        "language_locale": "en-US",
        "accent_self_reported": "General American",
        "age_range": "25-34",
        "gender_self_reported": "female",
        "native_language": "en",
        "take_id": f"{session_id}-{speaker}",
        "datetime_utc": "2025-09-10T20:39:27Z",
        "duration_seconds": 0.5,
        "transcript": f"hello <breath/> from {speaker}",
        "segments": [
            {
                "index": 0,
                "start": 0.0,
                "end": 0.4,
                "text": f"hello from {speaker}",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.2},
                    {"word": "from", "start": 0.2, "end": 0.3},
                    {"word": speaker, "start": 0.3, "end": 0.4},
                ],
            }
        ],
    }
    row.update(extra)
    return row


@pytest.fixture
def two_sessions(tmp_path: Path) -> tuple[list[dict], Path, Path]:
    """Two sessions: session-a has two tracks 1.5 s apart, session-b has one track."""
    audio_dir = tmp_path / "src"
    rows = [
        make_row("session-b", "zed", BASE_MS + 10_000),
        make_row(
            "session-a",
            "bob",
            BASE_MS + 1_500,
            transcript="",
            segments=[
                {
                    "index": 1,
                    "start": 1.0,
                    "end": 1.5,
                    "text": "second segment",
                    "words": [
                        {"word": "second", "start": 1.0, "end": 1.2},
                        {"word": "segment", "start": 1.2, "end": 1.5},
                    ],
                },
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 0.6,
                    "text": "first <laugh/> segment",
                    "words": [
                        {"word": "first", "start": 0.0, "end": 0.2},
                        {"word": "", "start": 0.2, "end": 0.3},
                        {"word": "ghost", "start": 0.3, "end": 0.3},
                        {"word": "segment", "start": 0.4, "end": 0.6},
                    ],
                },
            ],
        ),
        make_row("session-a", "amy", BASE_MS),
    ]
    for row in rows:
        write_flac(audio_dir / row["file_name"])
    return rows, audio_dir, tmp_path / "out"


def test_capture_ms_from_name_parses_trailing_ms():
    assert capture_ms_from_name(track_name(BASE_MS, "ab12cd")) == BASE_MS
    assert capture_ms_from_name(f"train/{track_name(BASE_MS, 'ab12cd')}") == BASE_MS
    assert capture_ms_from_name("something-else.flac") is None
    assert capture_ms_from_name("1757534367463.flac") is None
    assert capture_ms_from_name("noext") is None


def test_build_dataset_tracks_offsets_words_and_manifest(two_sessions):
    rows, audio_dir, output = two_sessions
    dataset = build_dataset(rows, lambda name: audio_dir / name, output, source={"kind": "test"})

    assert [conversation.id for conversation in dataset.conversations] == ["session-a", "session-b"]
    assert dataset.source == {"kind": "test"}
    assert dataset.created_at.endswith("Z")

    session_a = dataset.conversation("session-a")
    assert session_a.language == "en"
    assert session_a.locale == "en-US"
    assert [track.id for track in session_a.tracks] == ["ch1", "ch2"]
    ch1, ch2 = session_a.tracks

    assert ch1.speaker_id == "amy"
    assert ch1.offset_s == 0.0
    assert ch1.capture_ms == BASE_MS
    assert ch2.speaker_id == "bob"
    assert ch2.offset_s == 1.5
    assert ch2.capture_ms == BASE_MS + 1_500
    assert ch1.path == "audio/session-a/ch1.flac"
    assert ch2.path == "audio/session-a/ch2.flac"
    assert ch1.audio_path(output) == output / "audio" / "session-a" / "ch1.flac"
    assert ch1.audio_path(output).is_file() and ch2.audio_path(output).is_file()

    assert ch1.duration_s == pytest.approx(0.5)
    assert ch1.sample_rate == SAMPLE_RATE
    assert session_a.duration_s == pytest.approx(2.0)
    assert ch1.source_file == rows[2]["file_name"]
    assert ch1.metadata == {
        "accent_self_reported": "General American",
        "age_range": "25-34",
        "gender_self_reported": "female",
        "native_language": "en",
        "take_id": "session-a-amy",
        "datetime_utc": "2025-09-10T20:39:27Z",
    }

    # Transcript kept verbatim (tags included) when present.
    assert ch1.reference == "hello <breath/> from amy"
    # Falls back to segment text joined in index order when transcript is empty.
    assert ch2.reference == "first <laugh/> segment second segment"

    # Words are shifted by the offset, zero-duration and empty words are dropped, and sorted by start.
    assert [word.text for word in ch2.words] == ["first", "segment", "second", "segment"]
    assert [word.start_s for word in ch2.words] == pytest.approx([1.5, 1.9, 2.5, 2.7])
    assert [word.end_s for word in ch2.words] == pytest.approx([1.7, 2.1, 2.7, 3.0])
    assert [word.start_s for word in ch1.words] == pytest.approx([0.0, 0.2, 0.3])

    session_b = dataset.conversation("session-b")
    assert [track.id for track in session_b.tracks] == ["ch1"]
    assert len(dataset.tracks) == 3

    manifest = output / MANIFEST_NAME
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA_VERSION
    assert data == dataset.to_dict()

    with pytest.raises(DatasetError):
        dataset.conversation("nope")
    with pytest.raises(DatasetError):
        session_a.track("ch9")


def test_load_dataset_round_trips_and_digest_is_stable(two_sessions):
    rows, audio_dir, output = two_sessions
    built = build_dataset(rows, lambda name: audio_dir / name, output, source={"kind": "test"})
    loaded = load_dataset(output)

    assert loaded.to_dict() == built.to_dict()
    assert loaded.root == output
    assert loaded.digest() == built.digest()
    assert len(built.digest()) == 16
    assert int(built.digest(), 16) >= 0

    # Rebuilding into the same folder skips the copy (same sha) and gives the same digest.
    rebuilt = build_dataset(rows, lambda name: audio_dir / name, output, source={"kind": "test"})
    assert rebuilt.digest() == built.digest()


def test_verify_dataset_detects_tampered_audio(two_sessions):
    rows, audio_dir, output = two_sessions
    dataset = build_dataset(rows, lambda name: audio_dir / name, output, source={"kind": "test"})
    assert verify_dataset(dataset) == []

    tampered = dataset.conversation("session-a").track("ch2").audio_path(output)
    write_flac(tampered, seconds=1.0)
    problems = verify_dataset(load_dataset(output))
    assert len(problems) == 2
    assert all(problem.startswith("session-a/ch2:") for problem in problems)
    assert any("sha256 mismatch" in problem for problem in problems)
    assert any("duration mismatch" in problem for problem in problems)

    tampered.unlink()
    problems = verify_dataset(load_dataset(output))
    assert len(problems) == 1
    assert "missing audio file" in problems[0]


def test_build_dataset_rejects_row_without_session_id(tmp_path: Path):
    audio_dir = tmp_path / "src"
    good = make_row("session-a", "amy", BASE_MS)
    bad = make_row("session-a", "bob", BASE_MS + 1)
    del bad["session_id"]
    for row in (good, bad):
        write_flac(audio_dir / row["file_name"])

    with pytest.raises(DatasetError, match="row 1"):
        build_dataset([good, bad], lambda name: audio_dir / name, tmp_path / "out", source={"kind": "test"})
    with pytest.raises(DatasetError):
        build_dataset([], lambda name: audio_dir / name, tmp_path / "out", source={"kind": "test"})


def test_unparsable_capture_ms_gives_zero_offset(tmp_path: Path):
    audio_dir = tmp_path / "src"
    row = make_row("session-a", "amy", BASE_MS, file_name="weird-name.flac")
    write_flac(audio_dir / row["file_name"])
    dataset = build_dataset([row], lambda name: audio_dir / name, tmp_path / "out", source={"kind": "test"})
    track = dataset.conversation("session-a").tracks[0]
    assert track.capture_ms is None
    assert track.offset_s == 0.0
    assert dataset.to_dict()["conversations"][0]["tracks"][0]["capture_ms"] is None


@pytest.mark.parametrize("subdir", ["", "train"])
def test_build_from_folder(tmp_path: Path, subdir: str):
    source_dir = tmp_path / "folder"
    base = source_dir / subdir if subdir else source_dir
    rows = [make_row("session-a", "amy", BASE_MS), make_row("session-a", "bob", BASE_MS + 250)]
    for row in rows:
        write_flac(base / row["file_name"])
    base.mkdir(parents=True, exist_ok=True)
    with (base / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.write("\n")

    dataset = build_from_folder(source_dir, tmp_path / "out")
    assert dataset.source["kind"] == "folder"
    assert dataset.source["metadata_path"] == f"{subdir}/metadata.jsonl".lstrip("/")
    tracks = dataset.conversation("session-a").tracks
    assert [track.speaker_id for track in tracks] == ["amy", "bob"]
    assert tracks[1].offset_s == 0.25
    assert verify_dataset(dataset) == []


def test_build_from_folder_without_metadata(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="metadata.jsonl"):
        build_from_folder(tmp_path / "empty", tmp_path / "out")


def test_read_jsonl_tolerates_blank_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]
    with pytest.raises(DatasetError):
        read_jsonl(tmp_path / "missing.jsonl")


def test_load_dataset_errors(tmp_path: Path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"schema": 99, "conversations": []}), encoding="utf-8")
    with pytest.raises(DatasetError, match="schema"):
        load_dataset(tmp_path)
