from __future__ import annotations

import os

import converse_diarization.providers as providers
from converse_diarization.cli import main
from tests.fixtures import make_dataset
from tests.test_runner import FakeProvider


def test_dataset_show_mix_and_rttm(tmp_path, capsys, monkeypatch):
    dataset = make_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["dataset", "show", str(dataset.root)]) == 0
    out = capsys.readouterr().out
    assert "overlap" in out and "reference speaker time" in out and "(high)" in out and "(low)" in out

    assert main(["dataset", "mix", str(dataset.root)]) == 0
    out = capsys.readouterr().out
    assert "2 mixes ready" in out and (dataset.root / "mix" / f"{dataset.conversations[0].id}.wav").is_file()

    assert main(["rttm", "--dataset", str(dataset.root), "--output", str(tmp_path / "ref")]) == 0
    rttm = tmp_path / "ref" / f"{dataset.conversations[1].id}.rttm"
    assert rttm.is_file() and rttm.read_text(encoding="utf-8").count("SPEAKER ") == 2
    assert str(rttm) in capsys.readouterr().out

    # Comparing the reference with itself is a perfect score.
    assert main(["compare", str(rttm), str(rttm)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("DER 0.00%") and "JER 0.00%" in out and "speakers 2 reference / 2 hypothesis" in out


def test_run_and_report_end_to_end(tmp_path, capsys, monkeypatch):
    dataset = make_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(providers.PROVIDERS, "fake", FakeProvider)
    config = tmp_path / "fake.toml"
    config.write_text(
        'name = "fake-diarizer"\nprovider = "fake"\nmodel = "fake-1"\nvendor = "Tests"\n', encoding="utf-8"
    )

    code = main(
        ["run", "--model", str(config), "--dataset", str(dataset.root), "--output", str(tmp_path / "runs/fake")]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "DER" in captured.out and "run directory" in captured.out
    assert "2 diarized" in captured.err

    assert main(["score", "--run", str(tmp_path / "runs/fake"), "--json"]) == 0
    assert '"der"' in capsys.readouterr().out

    report = tmp_path / "report.html"
    assert main(["--runs-dir", str(tmp_path / "runs"), "report", "--output", str(report)]) == 0
    html = report.read_text(encoding="utf-8")
    assert "Converse-Diarization" in html and "fake-diarizer" in html and "__PAYLOAD__" not in html


def test_models_list_and_check_use_registry(tmp_path, capsys, monkeypatch):
    registry = tmp_path / "models.toml"
    registry.write_text('[acme-diarize]\nurl = "https://api.acme.example/v1"\nkey = "ACME_API_KEY"\n', encoding="utf-8")
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    assert main(["--models-file", str(registry), "models"]) == 0
    out = capsys.readouterr().out
    assert "acme-diarize" in out and "openai-compatible" in out and "pyannote-community-1" in out
    assert main(["--models-file", str(registry), "check"]) == 1
    assert "ACME_API_KEY is not set" in capsys.readouterr().out
    os.environ["ACME_API_KEY"] = "x"
    try:
        assert main(["--models-file", str(registry), "check", "acme-diarize"]) == 0
    finally:
        del os.environ["ACME_API_KEY"]


def test_models_add_keyless_providers(tmp_path, capsys):
    registry = tmp_path / "models.toml"
    rttm_dir = tmp_path / "hyp"
    rttm_dir.mkdir()
    code = main(
        ["--models-file", str(registry), "models", "add", "external", "--provider", "rttm", "--dir", str(rttm_dir)]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ready (rttm:external)" in out
    assert (
        main(["--models-file", str(registry), "models", "add", "x", "--provider", "deepgram", "--key", "K_API_KEY"])
        == 2
    )
    assert "--url is required" in capsys.readouterr().err
