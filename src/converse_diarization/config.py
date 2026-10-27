"""Model configuration files.

A model config is a small TOML document that says which provider to call,
which model to ask for and any decoding options. Built-in configs live in
``converse_diarization/models``; you can also pass a path to your own file.

Example::

    name = "gpt-4o-transcribe-diarize"
    display_name = "GPT-4o Transcribe Diarize"
    vendor = "OpenAI"
    provider = "openai-compatible"
    model = "gpt-4o-transcribe-diarize"
    env = ["OPENAI_API_KEY"]

    [options]
    response_format = "diarized_json"
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SECRET_KEY = re.compile(r"(key|token|secret|password)", re.IGNORECASE)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Options that decide *where* and *how patiently* to call, not *what* comes back.
TRANSPORT_OPTIONS = frozenset({"base_url", "api_key_env", "timeout_s", "poll_wait_s", "deadline_s"})


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    vendor: str
    provider: str
    model: str
    env: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)
    href: str | None = None
    notes: str | None = None
    prerelease: bool = False  # pre-release deployment: `export` leaves the run out unless asked to include it
    source: str = "<memory>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "provider": self.provider,
            "model": self.model,
            "env": list(self.env),
            "options": dict(self.options),
            "href": self.href,
            "notes": self.notes,
            "prerelease": self.prerelease,
        }

    def digest(self) -> str:
        """Stable hash of everything that changes the hypotheses (not display fields or transport settings)."""
        options = {key: value for key, value in self.options.items() if key not in TRANSPORT_OPTIONS}
        payload = {"provider": self.provider, "model": self.model, "options": options}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _models_root():
    return resources.files("converse_diarization").joinpath("models")


def builtin_model_names() -> list[str]:
    names = []
    for entry in _models_root().iterdir():
        if entry.name.endswith(".toml"):
            names.append(entry.name[: -len(".toml")])
    return sorted(names)


def _reject_secret_keys(source: str, key: str, value: Any) -> None:
    """Options (and nested option tables such as ``provider_options``) must not carry credentials."""
    if _SECRET_KEY.search(key) and not key.endswith("_env"):
        raise ConfigError(f"{source}: option {key!r} looks like a credential; put secrets in .env instead")
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            _reject_secret_keys(source, f"{key}.{nested_key}", nested_value)


def parse_model_config(text: str, *, source: str = "<memory>") -> ModelConfig:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: invalid TOML: {exc}") from exc

    missing = [key for key in ("name", "provider", "model") if not data.get(key)]
    if missing:
        raise ConfigError(f"{source}: missing required field(s): {', '.join(missing)}")

    name = str(data["name"])
    if not _NAME.match(name):
        raise ConfigError(f"{source}: name {name!r} must be lowercase letters, digits, '.', '_' or '-'")

    options = data.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"{source}: [options] must be a table")
    for key, value in options.items():
        if key.endswith("_env"):
            if not isinstance(value, str) or not _ENV_NAME.match(value):
                raise ConfigError(
                    f"{source}: option {key!r} must name an environment variable such as VENDOR_API_KEY, "
                    f"not hold the secret itself"
                )
            continue
        _reject_secret_keys(source, key, value)

    env = data.get("env", [])
    if isinstance(env, str):
        env = [env]
    if not isinstance(env, list) or not all(isinstance(item, str) for item in env):
        raise ConfigError(f"{source}: env must be a list of variable names")
    prerelease = data.get("prerelease", False)
    if not isinstance(prerelease, bool):
        raise ConfigError(f"{source}: prerelease must be true or false")

    return ModelConfig(
        name=name,
        display_name=str(data.get("display_name") or name),
        vendor=str(data.get("vendor") or data["provider"]),
        provider=str(data["provider"]),
        model=str(data["model"]),
        env=tuple(env),
        options=dict(options),
        href=data.get("href"),
        notes=data.get("notes"),
        prerelease=prerelease,
        source=source,
    )


def load_model_config(value: str | Path, *, registry: Path | None = None) -> ModelConfig:
    """Load a model by name (``models.toml`` first, then built-ins) or any TOML file by path."""
    from converse_diarization.registry import Registry  # local import: registry depends on this module

    path = Path(value)
    if path.suffix == ".toml" or path.exists():
        if not path.is_file():
            raise ConfigError(f"model config not found: {path}")
        return parse_model_config(path.read_text(encoding="utf-8"), source=str(path))
    name = str(value)
    project = Registry.load(registry)
    if name in project:
        return project.config(name)
    resource = _models_root().joinpath(f"{name}.toml")
    if not resource.is_file():
        registered = ", ".join(project.names()) or "none"
        builtin = ", ".join(builtin_model_names())
        raise ConfigError(
            f"unknown model {name!r}; models in {project.path}: {registered}; built-in models: {builtin}. "
            f"Add one with: converse-diarization models add {name} --url <base url> --key <ENV_NAME>"
        )
    return parse_model_config(resource.read_text(encoding="utf-8"), source=f"builtin:{name}")


def all_model_names(*, registry: Path | None = None) -> list[str]:
    """Registered models first (file order), then built-ins that are not shadowed."""
    from converse_diarization.registry import Registry

    project = Registry.load(registry)
    names = project.names()
    names.extend(name for name in builtin_model_names() if name not in project)
    return names
