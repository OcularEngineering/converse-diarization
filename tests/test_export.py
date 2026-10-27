from __future__ import annotations

import json
from pathlib import Path

import pytest

from converse_diarization.cli import main
from converse_diarization.export import (
    build_export,
    harness_commit,
    harness_version,
    is_prerelease,
    load_board,
    write_export,
)
from converse_diarization.registry import Registry
from converse_diarization.report import ReportError, load_summary
from tests.test_report import _conversation, _write_run

BOARD = {
    "slug": "converse-diarization",
    "version": "1",
    "openWeights": ["model-b"],
    "displayNames": {"model-a": "Model A (site casing)"},
    "windows": {"high": {"label": "High overlap", "conversation": "conv-2", "start_s": 4.0, "duration_s": 8.0}},
}


def _vendor(run_dir: Path, vendor: str, **extra) -> None:
    """Give a fixture run a real vendor (for logos) and anything else the test needs on its model."""
    summary = load_summary(run_dir)
    summary["model"]["vendor"] = vendor
    summary["model"].update(extra)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({"schema": 1, "harness": {"name": "converse-diarization", "version": "0.1.0"}}), encoding="utf-8"
    )


@pytest.fixture
def export_runs(tmp_path):
    root = tmp_path / "runs"
    good = _write_run(
        root,
        "model-a",
        [
            _conversation("conv-1", missed=1.0, false_alarm=0.5, confusion=0.5, extra=False),
            _conversation("conv-2", missed=2.0, false_alarm=1.0, confusion=1.0, extra=False),
        ],
    )
    worse = _write_run(
        root,
        "model-b",
        [
            _conversation("conv-1", missed=3.0, false_alarm=2.0, confusion=1.0, extra=True),
            _conversation("conv-2", missed=4.0, false_alarm=2.0, confusion=2.0, extra=False),
        ],
    )
    _vendor(good, "ElevenLabs", provider="openrouter")
    _vendor(worse, "pyannote", provider="pyannote", href="https://huggingface.co/pyannote/x")
    return root, good, worse


def _registry(tmp_path: Path, text: str = "") -> Registry:
    path = tmp_path / "models.toml"
    path.write_text(text, encoding="utf-8")
    return Registry.load(path)


def test_export_writes_board_timelines_windows_and_logos(export_runs, tmp_path):
    root, good, worse = export_runs
    export = build_export([good, worse], board=BOARD, registry=_registry(tmp_path), generated="Oct 14, 2026")
    board = export["board"]

    assert board["slug"] == "converse-diarization" and board["version"] == "1"
    assert board["generated"] == "Oct 14, 2026"
    assert board["harness"] == {
        "name": "converse-diarization",
        "version": harness_version(),
        "commit": harness_commit(),
        "href": "https://github.com/OcularEngineering/converse-diarization",
    }
    assert board["conversations"] == 2 and board["dataset"]["digest"] == "dataset-digest"
    assert board["timelinesPath"] == "/data/converse-diarization/v1/timelines.json"

    assert [run["id"] for run in board["runs"]] == ["model-a", "model-b"], "best DER first"
    a, b = board["runs"]
    assert a["displayName"] == "Model A (site casing)", "board file casing wins over the run's display name"
    assert b["displayName"] == "Model B"
    assert (a["access"], a["route"]) == ("proprietary", "openrouter")
    assert (b["access"], b["route"]) == ("open", "local")
    assert a["prerelease"] is False and b["prerelease"] is False
    assert a["harness"] == {"name": "converse-diarization", "version": "0.1.0"}
    assert set(a) >= {
        "der",
        "derCollar",
        "jer",
        "breakdown",
        "breakdownCollar",
        "referenceS",
        "speakerCount",
        "tiers",
        "conversationsScored",
        "conversationsTotal",
        "failures",
        "complete",
        "scoredAt",
        "modelDigest",
        "conversations",
    }
    assert set(a["conversations"][0]) == {
        "id",
        "duration_s",
        "overlap_ratio",
        "tier",
        "der",
        "der_collar",
        "jer",
        "counts",
        "speakers",
    }

    timelines = export["timelines"]
    assert set(timelines["reference"]) == {"conv-1", "conv-2"}
    reference = timelines["reference"]["conv-1"]
    assert reference["speakers"] == ["ch1", "ch2"] and reference["duration_s"] == 16.0
    assert reference["segments"][:2] == [[0, 5000, 0], [4500, 10500, 1]], "milliseconds, speaker index"
    hypothesis = timelines["runs"]["model-b"]["conv-1"]
    assert hypothesis["assignment"] == [["ch1", "SPEAKER_00"], ["ch2", "SPEAKER_01"]]
    assert hypothesis["speakers"] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    assert hypothesis["segments"][-1] == [14500, 16000, 2]

    window = export["windows"]["high"]
    assert window["label"] == "High overlap" and window["conversation"] == "conv-2"
    assert window["tier"] == "25.00% overlap on the conversation"
    assert window["start_s"] == 4.0 and window["duration_s"] == 8.0
    # Reference turns ch1 0-5 and ch2 4.5-10.5 overlap for 0.5 s inside the 4-12 s window.
    assert window["overlap_s"] == 0.5
    assert window["reference"] == [[0.0, 1.0, 0], [0.5, 6.5, 1], [7.0, 8.0, 0]], "clipped to the window"
    assert set(window["runs"]) == {"model-a", "model-b"}
    assert window["runs"]["model-a"]["segments"] == [[0.0, 1.1, 0], [1.0, 6.8, 1], [7.0, 8.0, 0]]

    assert board["logos"]["ElevenLabs"] == {
        "light": "/data/converse-diarization/v1/logos/elevenlabs.svg",
        "dark": "/data/converse-diarization/v1/logos/elevenlabs-dark.svg",
    }
    assert board["logos"]["pyannote"] == "/data/converse-diarization/v1/logos/pyannote.png"

    out = write_export(export, tmp_path / "export")
    assert out == tmp_path / "export" / "converse-diarization" / "v1"
    written = json.loads((out / "board.json").read_text(encoding="utf-8"))
    assert written == board
    assert json.loads((out / "timelines.json").read_text(encoding="utf-8")) == timelines
    assert json.loads((out / "windows.json").read_text(encoding="utf-8")) == export["windows"]
    logos = {path.name for path in (out / "logos").iterdir()}
    assert logos == {"elevenlabs.svg", "elevenlabs-dark.svg", "pyannote.png"}


def test_prerelease_runs_are_excluded_unless_asked_for(export_runs, tmp_path):
    root, good, worse = export_runs
    registry = _registry(tmp_path, "[model-b]\nprovider = 'pyannote'\nmodel = 'x'\nprerelease = true\n")
    assert is_prerelease(load_summary(worse), registry) is True
    assert is_prerelease(load_summary(good), registry) is False

    public = build_export([good, worse], board=BOARD, registry=registry)
    assert [run["id"] for run in public["board"]["runs"]] == ["model-a"]
    assert public["excluded"] == ["model-b"]
    assert set(public["timelines"]["runs"]) == {"model-a"}
    assert set(public["windows"]["high"]["runs"]) == {"model-a"}

    internal = build_export([good, worse], board=BOARD, registry=registry, include_prerelease=True)
    assert {run["id"]: run["prerelease"] for run in internal["board"]["runs"]} == {"model-a": False, "model-b": True}
    assert internal["excluded"] == []

    # The run's own frozen config can also say so (the model was pre-release when it was scored).
    _vendor(good, "ElevenLabs", provider="openrouter", prerelease=True)
    assert is_prerelease(load_summary(good), _registry(tmp_path)) is True

    # Nothing left to publish is an error, not an empty board.
    with pytest.raises(ReportError, match="pre-release"):
        build_export([good, worse], board=BOARD, registry=registry)


def test_load_board_builtin_and_path(tmp_path):
    builtin = load_board(None)
    assert builtin["slug"] == "converse-diarization" and builtin["version"] == "1"
    assert set(builtin["windows"]) == {"medium", "high"}
    path = tmp_path / "board.json"
    path.write_text(json.dumps(BOARD), encoding="utf-8")
    assert load_board(path)["openWeights"] == ["model-b"]
    with pytest.raises(ReportError, match="unknown board"):
        load_board("nope")


def test_export_command(export_runs, tmp_path, capsys):
    root, good, worse = export_runs
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps(BOARD), encoding="utf-8")
    registry = tmp_path / "models.toml"
    registry.write_text("", encoding="utf-8")
    code = main(
        [
            "--models-file",
            str(registry),
            "export",
            "--run",
            str(good),
            "--run",
            str(worse),
            "--board",
            str(board_path),
            "--out",
            str(tmp_path / "out"),
            "--generated",
            "Oct 14, 2026",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "2 board runs" in out and "Model A (site casing) (ElevenLabs)" in out
    assert (tmp_path / "out" / "converse-diarization" / "v1" / "board.json").is_file()
