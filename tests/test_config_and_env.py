from __future__ import annotations

import pytest

from converse_diarization.config import ConfigError, builtin_model_names, load_model_config, parse_model_config
from converse_diarization.env import load_dotenv, parse_dotenv, redact

GOOD = """
name = "demo-model"
display_name = "Demo"
vendor = "Acme"
provider = "openai-compatible"
model = "demo-1"
env = ["DEMO_KEY"]

[options]
response_format = "json"
"""


def test_parse_model_config_round_trip():
    config = parse_model_config(GOOD, source="test")
    assert config.name == "demo-model"
    assert config.env == ("DEMO_KEY",)
    assert config.options == {"response_format": "json"}
    assert config.to_dict()["display_name"] == "Demo"


def test_digest_ignores_display_fields_but_not_options():
    base = parse_model_config(GOOD)
    renamed = parse_model_config(GOOD.replace('display_name = "Demo"', 'display_name = "Other"'))
    changed = parse_model_config(GOOD.replace('response_format = "json"', 'response_format = "text"'))
    assert base.digest() == renamed.digest()
    assert base.digest() != changed.digest()


@pytest.mark.parametrize(
    "old, new, match",
    [
        ('name = "demo-model"\n', "", "missing required"),
        ('name = "demo-model"', 'name = "Bad Name"', "must be lowercase"),
        ("[options]", '[options]\napi_key = "x"', "credential"),
        ('env = ["DEMO_KEY"]', "env = 5", "list of variable names"),
        ("[options]", "[options", "invalid TOML"),
    ],
)
def test_parse_model_config_rejects_bad_input(old, new, match):
    with pytest.raises(ConfigError, match=match):
        parse_model_config(GOOD.replace(old, new))


def test_load_model_config_by_path_and_builtin(tmp_path):
    path = tmp_path / "demo.toml"
    path.write_text(GOOD, encoding="utf-8")
    assert load_model_config(path).source == str(path)
    names = builtin_model_names()
    assert names, "expected built-in model configs"
    assert load_model_config(names[0]).name == names[0]
    with pytest.raises(ConfigError):
        load_model_config("does-not-exist")


def test_parse_dotenv_handles_quotes_comments_and_export():
    values = parse_dotenv("# c\nexport A=\"x y\"\nB='z'\nC=plain # note\nD=\n")
    assert values == {"A": "x y", "B": "z", "C": "plain", "D": ""}


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CD_TEST_A=one\nCD_TEST_B=two\n", encoding="utf-8")
    monkeypatch.setenv("CD_TEST_A", "keep")
    monkeypatch.delenv("CD_TEST_B", raising=False)
    assert load_dotenv(env) == ["CD_TEST_B"]
    assert load_dotenv(tmp_path / "missing.env") == []


def test_redact_masks_known_key_shapes(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "customsecretvalue123")
    text = "Bearer sk-abcdefghijklmnopqrstuvwxyz and omi_sk_abcdefgh123 and customsecretvalue123"
    assert "sk-abc" not in redact(text)
    assert "omi_sk_" not in redact(text)
    assert "customsecretvalue123" not in redact(text)
