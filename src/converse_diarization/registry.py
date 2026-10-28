"""The project model registry: ``models.toml``.

One file lists every model the harness runs. Adding a hosted model that returns
speaker-labelled segments from the OpenAI ``/audio/transcriptions`` protocol
takes two lines::

    [gpt-4o-transcribe-diarize]
    url = "https://api.openai.com/v1"
    key = "OPENAI_API_KEY"            # name of the variable in .env

Everything else is optional: ``model`` (defaults to the table name), ``vendor``,
``display_name``, ``href``, ``notes``, ``provider`` (defaults to
``openai-compatible``) and an ``[name.options]`` table for provider options.
Secrets never live in this file; ``key`` names the environment variable.

A second file next to it, ``models.local.toml``, is read the same way and merged over
``models.toml``: its tables add to or replace the public ones. It is git-ignored, so
models that cannot be named in the repository yet (a pre-release deployment, a private
endpoint) live there, usually with ``prerelease = true``. ``models add`` / ``models remove``
only ever write ``models.toml``.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from converse_diarization.config import ConfigError, ModelConfig, parse_model_config

DEFAULT_REGISTRY = Path("models.toml")
REGISTRY_ENV = "CONVERSE_DIARIZATION_MODELS_FILE"
LOCAL_SUFFIX = ".local"  # models.toml -> models.local.toml
DEFAULT_PROVIDER = "openai-compatible"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TRANSCRIPTIONS_SUFFIX = "/audio/transcriptions"

_FIELD_ALIASES = {"url": "base_url", "key": "api_key_env"}
_TOP_LEVEL = {"name", "display_name", "vendor", "provider", "model", "env", "href", "notes", "options", "prerelease"}


def registry_path(path: Path | None = None) -> Path:
    """``models.toml`` next to the working directory unless overridden."""
    if path is not None:
        return Path(path)
    override = os.environ.get(REGISTRY_ENV)
    return Path(override) if override else DEFAULT_REGISTRY


def local_registry_path(path: Path) -> Path:
    """The git-ignored overlay beside a registry file: ``models.toml`` -> ``models.local.toml``."""
    return path.with_name(f"{path.stem}{LOCAL_SUFFIX}{path.suffix}")


def _read_tables(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    entries: dict[str, dict[str, Any]] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: top-level key {name!r} must be a table, one per model")
        entries[name] = entry
    return entries


def normalize_url(url: str) -> str:
    """Accept a base URL or the full transcription endpoint and return the base URL."""
    url = str(url).strip().rstrip("/")
    if url.endswith(_TRANSCRIPTIONS_SUFFIX):
        url = url[: -len(_TRANSCRIPTIONS_SUFFIX)]
    return url.rstrip("/")


def _entry_to_config_dict(name: str, entry: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Expand the registry shorthand (``url``/``key``) into a full model config dict."""
    if not isinstance(entry, dict):
        raise ConfigError(f"{source}: [{name}] must be a table")
    options: dict[str, Any] = dict(entry.get("options") or {})
    data: dict[str, Any] = {"name": name}
    for field, value in entry.items():
        if field == "options":
            continue
        if field in _FIELD_ALIASES:
            options[_FIELD_ALIASES[field]] = value
        elif field in _TOP_LEVEL:
            data[field] = value
        else:
            # Any other scalar is a provider option; this keeps simple entries flat.
            options[field] = value

    if "api_key_env" in options:
        key = str(options["api_key_env"])
        if not _ENV_NAME.match(key):
            raise ConfigError(
                f"{source}: [{name}] key must be the NAME of an environment variable (for example "
                f"OPENAI_API_KEY), not the secret itself. Put the secret in .env."
            )
        env = list(data.get("env") or [])
        if key not in env:
            env.append(key)
        data["env"] = env
    if "base_url" in options:
        options["base_url"] = normalize_url(options["base_url"])

    data.setdefault("provider", DEFAULT_PROVIDER)
    data.setdefault("model", name)
    if options:
        data["options"] = options
    return data


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else _toml_scalar(key)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    if isinstance(value, dict):  # inline table, e.g. provider_options = { deepgram = { diarize = true } }
        items = ", ".join(f"{_toml_key(key)} = {_toml_scalar(item)}" for key, item in value.items())
        return "{ " + items + " }" if items else "{}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _config_dict_to_toml(data: dict[str, Any]) -> str:
    """Render one config dict as TOML text (used both for parsing and for writing the file back)."""
    lines = []
    for field in ("name", "display_name", "vendor", "provider", "model", "env", "href", "notes"):
        if field in data and data[field] not in (None, "", []):
            lines.append(f"{field} = {_toml_scalar(data[field])}")
    if data.get("prerelease") is True:
        lines.append("prerelease = true")
    options = data.get("options") or {}
    if options:
        lines.append("")
        lines.append("[options]")
        for key in sorted(options):
            lines.append(f"{key} = {_toml_scalar(options[key])}")
    return "\n".join(lines) + "\n"


class Registry:
    """Parsed ``models.toml`` plus its ``models.local.toml`` overlay. Entries are kept in file order,
    public file first; ``local`` names the entries that came from (or were replaced by) the overlay."""

    def __init__(
        self,
        path: Path,
        entries: dict[str, dict[str, Any]],
        overlay: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.local_path = local_registry_path(path)
        # ``public`` is what models.toml holds and what save() writes; ``entries`` is the merged view.
        self.public = dict(entries)
        self.entries = dict(entries)
        self.local: set[str] = set()
        for name, entry in (overlay or {}).items():
            self.entries[name] = entry
            self.local.add(name)
        self._appended: list[str] = []
        self._rewrite = False

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        resolved = registry_path(path)
        entries = _read_tables(resolved) if resolved.is_file() else {}
        overlay_path = local_registry_path(resolved)
        overlay = _read_tables(overlay_path) if overlay_path.is_file() else {}
        return cls(resolved, entries, overlay)

    def names(self) -> list[str]:
        return list(self.entries)

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def source_of(self, name: str) -> Path:
        return self.local_path if name in self.local else self.path

    def config(self, name: str) -> ModelConfig:
        if name not in self.entries:
            raise ConfigError(f"model {name!r} is not in {self.path}")
        source = f"{self.source_of(name)}:[{name}]"
        data = _entry_to_config_dict(name, self.entries[name], source=source)
        return parse_model_config(_config_dict_to_toml(data), source=source)

    def configs(self) -> list[ModelConfig]:
        return [self.config(name) for name in self.entries]

    def add(
        self,
        name: str,
        *,
        url: str | None,
        key_env: str | None,
        model: str | None = None,
        provider: str = DEFAULT_PROVIDER,
        vendor: str | None = None,
        display_name: str | None = None,
        href: str | None = None,
        notes: str | None = None,
        options: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> ModelConfig:
        if name in self.local:
            raise ConfigError(f"model {name!r} is defined in {self.local_path}; edit that file by hand")
        if name in self.entries and not replace:
            raise ConfigError(f"model {name!r} already exists in {self.path}; pass --replace to overwrite it")
        entry: dict[str, Any] = {}
        if url:
            entry["url"] = normalize_url(url)
        if key_env:
            entry["key"] = key_env
        if model and model != name:
            entry["model"] = model
        if provider != DEFAULT_PROVIDER:
            entry["provider"] = provider
        for field, value in (("vendor", vendor), ("display_name", display_name), ("href", href), ("notes", notes)):
            if value:
                entry[field] = value
        if options:
            entry["options"] = dict(options)
        # Validate before touching the file.
        source = f"{self.path}:[{name}]"
        expanded = _entry_to_config_dict(name, entry, source=source)
        config = parse_model_config(_config_dict_to_toml(expanded), source=source)
        if name in self.public:
            self._rewrite = True  # replacing: the old block has to go
        else:
            self._appended.append(name)
        self.public[name] = entry
        self.entries[name] = entry
        return config

    def remove(self, name: str) -> None:
        if name in self.local:
            raise ConfigError(f"model {name!r} is defined in {self.local_path}; edit that file by hand")
        if name not in self.public:
            raise ConfigError(f"model {name!r} is not in {self.path}")
        del self.public[name]
        del self.entries[name]
        self._rewrite = True

    @staticmethod
    def _render_entry(name: str, entry: dict[str, Any]) -> str:
        lines = [f"[{name}]"]
        options = entry.get("options") or {}
        for field, value in entry.items():
            if field == "options":
                continue
            lines.append(f"{field} = {_toml_scalar(value)}")
        if options:
            lines.append("")
            lines.append(f"[{name}.options]")
            for key in sorted(options):
                lines.append(f"{key} = {_toml_scalar(options[key])}")
        return "\n".join(lines) + "\n"

    def render(self) -> str:
        header = (
            "# Models the Ocular AI Converse-Diarization harness runs. One table per model.\n"
            '# Through OpenRouter: provider = "openrouter", model = "<vendor>/<id>", key = "OPENROUTER_API_KEY".\n'
            "# Direct: url + key (the NAME of the variable in .env) for an OpenAI-compatible diarizing endpoint.\n"
            "# Optional: model (defaults to the table name), vendor, display_name, href, notes, options, provider\n"
            "# (openai-compatible | openrouter | deepgram | assemblyai | azure | diarizen | elevenlabs | meta | nemo\n"
            "#  | nemotron-transcribe | xai | pyannote | qwen-asr | rttm).\n"
            "# Shell: converse-diarization models add <name> --openrouter --model <vendor/id>\n"
            "#    or: converse-diarization models add <name> --url <base url> --key <secret or ENV>\n"
        )
        blocks = [header] + [self._render_entry(name, entry) for name, entry in self.public.items()]
        return "\n".join(blocks)

    def save(self) -> Path:
        """Append new entries to an existing file (keeps your comments); rewrite only after remove/replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and not self._rewrite:
            existing = self.path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            blocks = [self._render_entry(name, self.public[name]) for name in self._appended]
            self.path.write_text(existing + "".join(f"\n{block}" for block in blocks), encoding="utf-8")
        else:
            self.path.write_text(self.render(), encoding="utf-8")
        self._appended.clear()
        self._rewrite = False
        return self.path


def suggest_env_name(name: str) -> str:
    """``<vendor>-<model>`` -> ``<VENDOR>_API_KEY``: the first word of the name, so one vendor shares one key."""
    head = re.split(r"[^a-z0-9]+", name.lower().strip())[0] or "model"
    return f"{head.upper()}_API_KEY"


def looks_like_env_name(value: str) -> bool:
    return bool(_ENV_NAME.match(value.strip()))


def upsert_dotenv(path: Path, key: str, value: str) -> Path:
    """Set ``KEY=value`` in a dotenv file, replacing an existing line or appending a new one."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=")
    rendered = f'{key}="{value}"' if any(ch in value for ch in " #'\"") else f"{key}={value}"
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = rendered
            break
    else:
        lines.append(rendered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_REGISTRY",
    "LOCAL_SUFFIX",
    "REGISTRY_ENV",
    "Registry",
    "local_registry_path",
    "looks_like_env_name",
    "normalize_url",
    "registry_path",
    "suggest_env_name",
    "upsert_dotenv",
]
