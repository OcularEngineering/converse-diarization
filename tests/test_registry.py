from __future__ import annotations

import os

import pytest

from converse_diarization.cli import main
from converse_diarization.config import ConfigError, all_model_names, load_model_config
from converse_diarization.env import parse_dotenv
from converse_diarization.registry import Registry, normalize_url, suggest_env_name, upsert_dotenv

MINIMAL = """
# my models
[acme-stt-1]
url = "https://api.acme.example/v1/audio/transcriptions/"
key = "ACME_API_KEY"

[acme-stt]
url = "https://api.acme.ai/v1"
key = "ACME_API_KEY"
model = "acme-stt-2"
vendor = "Acme"
display_name = "Acme STT 2"
upload = "mp3"
"""


def test_two_line_entry_expands_to_openai_compatible_config(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MINIMAL, encoding="utf-8")
    registry = Registry.load(path)
    assert registry.names() == ["acme-stt-1", "acme-stt"]

    config = registry.config("acme-stt-1")
    assert config.provider == "openai-compatible"
    assert config.model == "acme-stt-1"
    assert config.env == ("ACME_API_KEY",)
    assert config.options == {"base_url": "https://api.acme.example/v1", "api_key_env": "ACME_API_KEY"}

    acme = registry.config("acme-stt")
    assert acme.model == "acme-stt-2" and acme.vendor == "Acme" and acme.display_name == "Acme STT 2"
    assert acme.options["upload"] == "mp3"


def test_registry_shadows_builtins_and_lists_first(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('[pyannote-community-1]\nprovider = "pyannote"\nvendor = "pyannote"\n', encoding="utf-8")
    config = load_model_config("pyannote-community-1", registry=path)
    assert config.source.endswith(":[pyannote-community-1]")
    names = all_model_names(registry=path)
    assert names[0] == "pyannote-community-1" and names.count("pyannote-community-1") == 1
    assert "gpt-4o-transcribe-diarize" in names


def test_secret_in_key_field_is_rejected(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('[leaky]\nurl = "https://x.ai/v1"\nkey = "sk-live-abcdef"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="NAME of an environment variable"):
        Registry.load(path).config("leaky")


def test_transport_options_do_not_change_the_digest(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[a]\nurl = "https://one.example/v1"\nkey = "A_KEY"\n\n'
        '[b]\nurl = "https://two.example/v1"\nkey = "B_KEY"\nmodel = "a"\n\n'
        '[c]\nurl = "https://one.example/v1"\nkey = "A_KEY"\nmodel = "a"\nupload = "mp3"\n',
        encoding="utf-8",
    )
    registry = Registry.load(path)
    assert registry.config("a").digest() == registry.config("b").digest()
    assert registry.config("a").digest() != registry.config("c").digest()


def test_add_appends_and_keeps_comments_remove_rewrites(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MINIMAL, encoding="utf-8")
    registry = Registry.load(path)
    registry.add("new-one", url="https://api.new.example/v1/", key_env="NEW_API_KEY", vendor="New")
    registry.save()
    text = path.read_text(encoding="utf-8")
    assert "# my models" in text and text.rstrip().endswith('vendor = "New"')
    assert Registry.load(path).names() == ["acme-stt-1", "acme-stt", "new-one"]

    with pytest.raises(ConfigError, match="already exists"):
        registry.add("new-one", url="https://api.new.example/v1", key_env="NEW_API_KEY")

    registry.remove("acme-stt")
    registry.save()
    reloaded = Registry.load(path)
    assert reloaded.names() == ["acme-stt-1", "new-one"]
    assert reloaded.config("new-one").options["base_url"] == "https://api.new.example/v1"


def test_helpers():
    assert normalize_url("https://api.x.com/v1/audio/transcriptions") == "https://api.x.com/v1"
    assert normalize_url("https://api.x.com/v1/") == "https://api.x.com/v1"
    assert suggest_env_name("acme-stt-1") == "ACME_API_KEY"
    assert suggest_env_name("gpt-4o") == "GPT_API_KEY"


def test_upsert_dotenv_replaces_or_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nB=2\n", encoding="utf-8")
    upsert_dotenv(env, "B", "two")
    upsert_dotenv(env, "C", "th ree")
    assert parse_dotenv(env.read_text(encoding="utf-8")) == {"A": "1", "B": "two", "C": "th ree"}


def test_cli_models_add_with_literal_key_writes_dotenv(tmp_path, capsys, monkeypatch):
    registry = tmp_path / "models.toml"
    env = tmp_path / ".env"
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    code = main(
        [
            "--models-file",
            str(registry),
            "--env-file",
            str(env),
            "models",
            "add",
            "acme-stt-1",
            "--url",
            "https://api.acme.example/v1",
            "--key",
            "acme_secret_value_1234",
            "--model",
            "stt-1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ready" in out and "acme_secret_value_1234" not in out
    assert parse_dotenv(env.read_text(encoding="utf-8")) == {"ACME_API_KEY": "acme_secret_value_1234"}
    assert "acme_secret_value_1234" not in registry.read_text(encoding="utf-8")
    assert os.environ["ACME_API_KEY"] == "acme_secret_value_1234"

    assert main(["--models-file", str(registry), "models", "show", "acme-stt-1"]) == 0
    assert '"model": "stt-1"' in capsys.readouterr().out

    # Default `check` covers every registered model; --model works too.
    assert main(["--models-file", str(registry), "--env-file", str(env), "check"]) == 0
    assert main(["--models-file", str(registry), "--env-file", str(env), "check", "--model", "acme-stt-1"]) == 0


def test_cli_models_add_openrouter_route(tmp_path, capsys, monkeypatch):
    import tomllib

    registry = tmp_path / "models.toml"
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    args = ["--models-file", str(registry), "models", "add"]
    assert main([*args, "nova", "--openrouter", "--model", "deepgram/nova-3", "--upload", "flac"]) == 0
    out = capsys.readouterr().out
    assert "nova: ready (openrouter:deepgram/nova-3)" in out
    data = tomllib.loads(registry.read_text(encoding="utf-8"))
    assert data["nova"]["provider"] == "openrouter" and data["nova"]["key"] == "OPENROUTER_API_KEY"
    assert data["nova"]["notes"] == "via OpenRouter" and data["nova"]["href"] == "https://openrouter.ai/deepgram/nova-3"
    assert "url" not in data["nova"]

    # Nested vendor options arrive as JSON on the shell and round-trip through the file as inline tables.
    code = main(
        [
            *args,
            "acme",
            "--openrouter",
            "--model",
            "acme/listen-1",
            "--upload",
            "flac",
            "--option",
            'provider_options={"acme": {"speakers": "auto"}}',
        ]
    )
    assert code == 0, capsys.readouterr()
    data = tomllib.loads(registry.read_text(encoding="utf-8"))
    assert data["acme"]["options"]["provider_options"] == {"acme": {"speakers": "auto"}}

    # The direct route is untouched and lives next to the routed entries.
    assert (
        main(
            [
                *args,
                "nova-direct",
                "--provider",
                "deepgram",
                "--url",
                "https://api.deepgram.com/v1",
                "--key",
                "OPENROUTER_API_KEY",
                "--model",
                "nova-3",
            ]
        )
        == 0
    )
    assert "nova-direct" in registry.read_text(encoding="utf-8")

    assert main([*args, "x", "--openrouter"]) == 2
    assert "needs the OpenRouter model id" in capsys.readouterr().err
    assert main([*args, "x", "--openrouter", "--model", "a/b", "--url", "https://y"]) == 2
    assert main([*args, "x", "--openrouter", "--model", "a/b", "--provider", "deepgram"]) == 2
    assert main([*args, "x", "--openrouter", "--model", "a/b", "--option", "provider_options={oops"]) == 2
    assert "does not parse" in capsys.readouterr().err


def test_cli_models_add_requires_url_and_key(tmp_path, capsys):
    registry = tmp_path / "models.toml"
    assert main(["--models-file", str(registry), "models", "add", "x", "--key", "X_API_KEY"]) == 2
    assert "--url is required" in capsys.readouterr().err
    assert main(["--models-file", str(registry), "models", "add", "x", "--url", "https://x/v1"]) == 2
    assert "--key is required" in capsys.readouterr().err
    assert not registry.exists()


def test_local_overlay_merges_over_the_public_file_and_is_never_written(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MINIMAL, encoding="utf-8")
    overlay = tmp_path / "models.local.toml"
    overlay.write_text(
        '[secret-diarizer]\nurl = "https://private.example/v1"\nkey = "SECRET_KEY"\nprerelease = true\n\n'
        '[acme-stt]\nurl = "https://api.acme.ai/v1"\nkey = "ACME_API_KEY"\nnotes = "overridden"\n',
        encoding="utf-8",
    )
    registry = Registry.load(path)
    assert registry.names() == ["acme-stt-1", "acme-stt", "secret-diarizer"], "public order first, overlay after"
    assert registry.local == {"secret-diarizer", "acme-stt"}
    assert registry.config("secret-diarizer").prerelease is True
    assert registry.config("acme-stt").notes == "overridden", "an overlay table replaces the public one"
    assert registry.config("secret-diarizer").source.startswith(str(overlay))
    assert registry.entries["secret-diarizer"].get("prerelease") is True, "export's is_prerelease reads entries"
    assert registry.config("acme-stt-1").prerelease is False

    with pytest.raises(ConfigError, match="models.local.toml"):
        registry.remove("secret-diarizer")
    with pytest.raises(ConfigError, match="models.local.toml"):
        registry.add("secret-diarizer", url="https://x.example/v1", key_env="X_KEY", replace=True)

    registry.add("new-one", url="https://api.new.example/v1", key_env="NEW_API_KEY")
    registry.remove("acme-stt-1")
    registry.save()
    public = path.read_text(encoding="utf-8")
    assert "secret-diarizer" not in public and "private.example" not in public, "overlay never reaches models.toml"
    assert "[new-one]" in public and "[acme-stt-1]" not in public
    assert overlay.read_text(encoding="utf-8").count("[") == 2, "the overlay is left alone"
    assert Registry.load(path).names() == ["acme-stt", "new-one", "secret-diarizer"]
