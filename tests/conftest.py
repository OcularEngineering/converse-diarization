from __future__ import annotations

import pytest

from converse_diarization.registry import REGISTRY_ENV


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Keep tests away from the project's real models.toml unless a test opts in."""
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "models.toml"))
    return tmp_path / "models.toml"
