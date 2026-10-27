"""Export scored runs as the JSON the Converse-Diarization website consumes.

The website never reads a run directory. ``converse-diarization export`` turns ``runs/`` into::

    <out>/<slug>/v<version>/board.json       every run with pooled seconds per tier and conversation
    <out>/<slug>/v<version>/timelines.json   reference and hypothesis speaker segments, in milliseconds
    <out>/<slug>/v<version>/windows.json     the 15 s report excerpts: reference turns and every run's segments
    <out>/<slug>/v<version>/logos/<file>     the vendor marks the board refers to

``slug``, ``version``, the open-weight list and the report windows come from the board file
(``boards/converse-diarization.json``). Models marked ``prerelease = true`` in the registry (usually
in the git-ignored ``models.local.toml``) are left out unless ``include_prerelease`` is set, in
which case their rows carry ``prerelease: true`` so a site can show them as previews.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from converse_diarization.registry import Registry
from converse_diarization.report import ReportError, load_summary, run_payload
from converse_diarization.runner import RUN_FILE

HARNESS_NAME = "converse-diarization"
HARNESS_HREF = "https://github.com/OcularEngineering/converse-diarization"
DEFAULT_BOARD = "converse-diarization"
DEFAULT_EXPORT = Path("export")
BOARD_FILE = "board.json"
TIMELINES_FILE = "timelines.json"
WINDOWS_FILE = "windows.json"
LOGOS_DIR = "logos"
# Providers that run a model from published weights on the harness machine.
LOCAL_PROVIDERS = frozenset({"pyannote", "diarizen", "nemo", "qwen-asr"})
# Window slices drop segments shorter than this, which would render as hairlines.
MIN_WINDOW_SEGMENT_S = 0.025
_HUGGINGFACE = "https://huggingface.co/"

# Vendor marks shipped with the package (``templates/logos``), keyed by the vendor name in models.toml.
VENDOR_LOGOS = {
    "Alibaba": "qwen.png",
    "AssemblyAI": "assemblyai.png",
    "BUT": "but.png",
    "Deepgram": "deepgram.svg",
    "ElevenLabs": "elevenlabs.svg",
    "Google": "google.png",
    "Meta": "meta.svg",
    "Microsoft": "microsoft.svg",
    "Microsoft AI": "microsoft.svg",
    "Mistral": "mistral.svg",
    "Mistral AI": "mistral.svg",
    "NVIDIA": "nvidia.svg",
    "OpenAI": "openai.svg",
    "pyannote": "pyannote.png",
    "xAI": "xai.png",
}
# Marks that are dark on a transparent background and need a white copy on dark surfaces.
VENDOR_LOGOS_DARK = {
    "ElevenLabs": "elevenlabs-dark.svg",
    "Mistral": "mistral-dark.svg",
    "Mistral AI": "mistral-dark.svg",
}


def harness_version() -> str:
    try:
        return metadata.version(HARNESS_NAME)
    except metadata.PackageNotFoundError:  # pragma: no cover - only when run from an unpacked checkout
        return "0.1.0"


def harness_commit() -> str | None:
    """The git commit of the harness checkout that produced the export, so a published board names the
    exact scoring code and not just a package version. ``None`` when not run from a git checkout."""
    root = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if len(sha) == 40 else None


def default_generated() -> str:
    return datetime.now(UTC).strftime("%b %-d, %Y")


def load_board(name_or_path: str | Path | None) -> dict[str, Any]:
    """A board file: a built-in name such as ``converse-diarization`` or a JSON path."""
    name_or_path = name_or_path or DEFAULT_BOARD
    path = Path(name_or_path)
    if path.suffix == ".json" and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        resource = resources.files("converse_diarization").joinpath("boards", f"{name_or_path}.json")
        if not resource.is_file():
            raise ReportError(f"unknown board {name_or_path!r}")
        data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("slug") or not data.get("version"):
        raise ReportError("a board file needs `slug` and `version`")
    return data


def is_prerelease(summary: dict[str, Any], registry: Registry) -> bool:
    """Pre-release models stay off the export: the run's own config says so, or the current registry does."""
    model = summary["model"]
    if model.get("prerelease"):
        return True
    entry = registry.entries.get(str(model.get("name") or ""), {})
    return bool(entry.get("prerelease"))


def _route(model: dict[str, Any]) -> str:
    provider = model.get("provider")
    if provider in LOCAL_PROVIDERS:
        return "local"
    if provider == "openrouter":
        return "openrouter"
    base_url = str((model.get("options") or {}).get("base_url") or "")
    return "openrouter" if "openrouter" in base_url else "direct"


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _conversation_row(conversation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": conversation["id"],
        "duration_s": conversation["duration_s"],
        "overlap_ratio": conversation["overlap_ratio"],
        "tier": conversation["tier"],
        "der": conversation["der"],
        "der_collar": conversation["der_collar"],
        "jer": conversation["jer"],
        "counts": conversation["counts"],
        "speakers": {key: conversation["speakers"][key] for key in ("reference", "hypothesis", "correct")},
    }


def _timeline(conversation: dict[str, Any], segments_key: str, *, with_assignment: bool) -> dict[str, Any]:
    """One conversation's segments as ``[start_ms, end_ms, speaker_index]`` over a speaker list."""
    if with_assignment:
        speakers = list(conversation["speaking_time"]["hypothesis"].keys())
    else:
        speakers = sorted({segment["speaker"] for segment in conversation[segments_key]})
    index = {speaker: i for i, speaker in enumerate(speakers)}
    out: dict[str, Any] = {}
    if with_assignment:
        out["assignment"] = conversation["assignment"]
    else:
        out["duration_s"] = conversation["duration_s"]
    out["speakers"] = speakers
    out["segments"] = [
        [_ms(segment["start_s"]), _ms(segment["end_s"]), index[segment["speaker"]]]
        for segment in conversation[segments_key]
    ]
    return out


def _window_slice(segments: list[dict[str, Any]], speakers: list[str], start: float, duration: float) -> list:
    index = {speaker: i for i, speaker in enumerate(speakers)}
    out = []
    for segment in segments:
        a, b = segment["start_s"] - start, segment["end_s"] - start
        if b <= 0 or a >= duration or (b - a) < MIN_WINDOW_SEGMENT_S:
            continue
        out.append([round(max(a, 0.0), 3), round(min(b, duration), 3), index[segment["speaker"]]])
    return out


def _overlap_seconds(segments: list[dict[str, Any]], start: float, duration: float) -> float:
    """Seconds inside the window where two or more reference speakers talk at once."""
    events: list[tuple[float, int]] = []
    for segment in segments:
        a, b = max(segment["start_s"], start), min(segment["end_s"], start + duration)
        if b > a:
            events += [(a, 1), (b, -1)]
    events.sort()
    active, overlap, last = 0, 0.0, 0.0
    for time, delta in events:
        if active >= 2:
            overlap += time - last
        active += delta
        last = time
    return round(overlap, 2)


def _find_conversation(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    for conversation in payload["conversations"]:
        if conversation["id"].startswith(prefix):
            return conversation
    raise ReportError(f"{payload['id']}: no conversation starting with {prefix!r} for a report window")


def convert_run(run_dir: Path, *, board: dict[str, Any], registry: Registry) -> dict[str, Any]:
    """One run directory as a board row plus its timelines and window slices."""
    run_dir = Path(run_dir)
    summary = load_summary(run_dir)
    record = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    payload = run_payload(run_dir)
    model = summary["model"]
    name = str(model["name"])
    entry = registry.entries.get(name, {})
    href = payload["href"] or entry.get("href") or ""
    open_weights = set(board.get("openWeights") or [])
    is_open = name in open_weights or model.get("provider") in LOCAL_PROVIDERS or href.startswith(_HUGGINGFACE)
    display_names = board.get("displayNames") or {}

    run = {
        "id": payload["id"],
        "displayName": display_names.get(name) or payload["display_name"],
        "vendor": payload["vendor"],
        "model": payload["model"],
        "href": href,
        "provider": payload["provider"],
        "route": _route(model),
        "access": "open" if is_open else "proprietary",
        # Scored on a pre-release deployment. Only ever true in an --include-prerelease export; the
        # site shows a Preview chip and no model link for such a row.
        "prerelease": is_prerelease(summary, registry),
        "notes": payload["notes"] or None,
        "der": payload["der"],
        "derCollar": payload["der_collar"],
        "jer": payload["jer"],
        "breakdown": payload["breakdown"],
        "breakdownCollar": payload["breakdown_collar"],
        "referenceS": payload["reference_s"],
        "speakerCount": payload["speaker_count"],
        "tiers": payload["tiers"],
        "conversationsScored": payload["conversations_scored"],
        "conversationsTotal": payload["conversations_total"],
        "failures": payload["failures"],
        "complete": payload["complete"],
        "scoredAt": payload["scored_at"],
        "modelDigest": payload["model_digest"],
        "harness": record.get("harness"),
        "conversations": [_conversation_row(c) for c in payload["conversations"]],
    }
    timelines = {c["id"]: _timeline(c, "hypothesis_segments", with_assignment=True) for c in payload["conversations"]}
    windows = {}
    for tier, window in (board.get("windows") or {}).items():
        conversation = _find_conversation(payload, window["conversation"])
        speakers = sorted(conversation["speaking_time"]["hypothesis"].keys())
        windows[tier] = {
            "assignment": conversation["assignment"],
            "speakers": speakers,
            "segments": _window_slice(
                conversation["hypothesis_segments"], speakers, window["start_s"], window["duration_s"]
            ),
        }
    return {"run": run, "timelines": timelines, "windows": windows, "payload": payload}


def _reference(payload: dict[str, Any], board: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reference timelines and window headers, read off one run's scored rows (every run scores the
    same reference)."""
    timelines = {c["id"]: _timeline(c, "reference_segments", with_assignment=False) for c in payload["conversations"]}
    windows = {}
    for tier, window in (board.get("windows") or {}).items():
        conversation = _find_conversation(payload, window["conversation"])
        speakers = sorted({segment["speaker"] for segment in conversation["reference_segments"]})
        windows[tier] = {
            "label": window["label"],
            "conversation": window["conversation"],
            "tier": f"{conversation['overlap_ratio']:.2f}% overlap on the conversation",
            "start_s": window["start_s"],
            "duration_s": window["duration_s"],
            "overlap_s": _overlap_seconds(conversation["reference_segments"], window["start_s"], window["duration_s"]),
            "reference": _window_slice(
                conversation["reference_segments"], speakers, window["start_s"], window["duration_s"]
            ),
            "runs": {},
        }
    return timelines, windows


def _logo_resource(file_name: str):
    return resources.files("converse_diarization").joinpath("templates", LOGOS_DIR, file_name)


def _logo_files(vendor: str) -> tuple[str, str | None] | None:
    """``(light, dark)`` file names shipped for a vendor; ``dark`` is ``None`` when the mark reads on both
    surfaces, the whole thing ``None`` when the package has no mark for the vendor."""
    light = VENDOR_LOGOS.get(vendor)
    if not light or not _logo_resource(light).is_file():
        return None
    dark = VENDOR_LOGOS_DARK.get(vendor)
    return light, (dark if dark and _logo_resource(dark).is_file() else None)


def build_export(
    run_dirs: list[Path],
    *,
    board: dict[str, Any],
    registry: Registry,
    generated: str | None = None,
    include_prerelease: bool = False,
) -> dict[str, Any]:
    """Everything ``write_export`` writes: ``board``, ``timelines``, ``windows`` and the ``excluded`` names."""
    slug = str(board["slug"])
    version = str(board["version"])
    data_path = f"/data/{slug}/v{version}"

    excluded: list[str] = []
    kept: list[Path] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not include_prerelease and is_prerelease(load_summary(run_dir), registry):
            excluded.append(run_dir.name)
            continue
        kept.append(run_dir)
    if not kept:
        raise ReportError("no runs to export" + (f" ({len(excluded)} pre-release run(s) excluded)" if excluded else ""))

    converted = [convert_run(run_dir, board=board, registry=registry) for run_dir in kept]
    # Best DER first, incomplete runs last, ties by id; the same order the HTML report uses.
    converted.sort(key=lambda item: (0 if item["run"]["complete"] else 1, item["run"]["der"], item["run"]["id"]))
    seen: set[str] = set()
    for item in converted:
        base = item["run"]["id"]
        suffix = 2
        while item["run"]["id"] in seen:
            item["run"]["id"] = f"{base}-{suffix}"
            suffix += 1
        seen.add(item["run"]["id"])

    runs = [item["run"] for item in converted]
    reference_timelines, windows = _reference(converted[0]["payload"], board)
    timelines = {"reference": reference_timelines, "runs": {item["run"]["id"]: item["timelines"] for item in converted}}
    for tier in windows:
        windows[tier]["runs"] = {item["run"]["id"]: item["windows"][tier] for item in converted}

    first = load_summary(kept[0])
    source = (first.get("dataset") or {}).get("source") or {}
    vendors = {str(run["vendor"]) for run in runs} | {
        str(entry.get("vendor") or "") for entry in registry.public.values()
    }
    vendors.discard("")
    logos: dict[str, Any] = {}
    for vendor in sorted(vendors):
        files = _logo_files(vendor)
        if not files:
            continue
        light, dark = files
        path = f"{data_path}/{LOGOS_DIR}/{light}"
        logos[vendor] = {"light": path, "dark": f"{data_path}/{LOGOS_DIR}/{dark}"} if dark else path

    board_json = {
        "slug": slug,
        "version": version,
        "generated": generated or default_generated(),
        "harness": {
            "name": HARNESS_NAME,
            "version": harness_version(),
            "commit": harness_commit(),
            "href": HARNESS_HREF,
        },
        "dataset": {
            "repo_id": source.get("repo_id"),
            "revision": source.get("revision"),
            "digest": (first.get("dataset") or {}).get("digest"),
            "conversations": int((first.get("conversations") or {}).get("total") or len(runs[0]["conversations"])),
        },
        "conversations": int((first.get("conversations") or {}).get("total") or len(runs[0]["conversations"])),
        "collar_s": first.get("collar_s"),
        "merge_gap_s": first.get("merge_gap_s"),
        "timelinesPath": f"{data_path}/{TIMELINES_FILE}",
        "runs": runs,
        "logos": logos,
    }
    return {
        "slug": slug,
        "version": version,
        "board": board_json,
        "timelines": timelines,
        "windows": windows,
        "excluded": excluded,
    }


def _clear(directory: Path, suffixes: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.iterdir():
        if old.is_file() and old.suffix in suffixes:
            old.unlink()


def write_export(export: dict[str, Any], out: Path) -> Path:
    """Write ``board.json``, ``timelines.json``, ``windows.json`` and the vendor marks under ``out``."""
    root = Path(out) / export["slug"] / f"v{export['version']}"
    root.mkdir(parents=True, exist_ok=True)
    board = export["board"]
    (root / BOARD_FILE).write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / TIMELINES_FILE).write_text(
        json.dumps(export["timelines"], separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    (root / WINDOWS_FILE).write_text(
        json.dumps(export["windows"], separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    logos_dir = root / LOGOS_DIR
    _clear(logos_dir, (".svg", ".png"))
    for vendor in board["logos"]:
        light, dark = _logo_files(vendor) or (None, None)
        for file_name in (light, dark):
            if file_name:
                with resources.as_file(_logo_resource(file_name)) as source:
                    shutil.copyfile(source, logos_dir / file_name)
    return root


def format_table(export: dict[str, Any]) -> str:
    board = export["board"]
    lines = [f"{len(board['runs'])} board runs, {len(board['logos'])} vendor marks, {len(export['windows'])} windows"]
    for run in board["runs"]:
        der = "  n/a" if run["der"] is None else f"{run['der']:5.2f}"
        jer = "  n/a" if run["jer"] is None else f"{run['jer']:5.2f}"
        flag = "  [prerelease]" if run["prerelease"] else ""
        lines.append(
            f"  {der}  {jer}  {run['access']:11}  {run['route']:10}  {run['displayName']} ({run['vendor']})  "
            f"fail={run['failures']}{flag}"
        )
    return "\n".join(lines)


__all__ = [
    "BOARD_FILE",
    "DEFAULT_BOARD",
    "DEFAULT_EXPORT",
    "HARNESS_HREF",
    "LOCAL_PROVIDERS",
    "LOGOS_DIR",
    "TIMELINES_FILE",
    "VENDOR_LOGOS",
    "VENDOR_LOGOS_DARK",
    "WINDOWS_FILE",
    "build_export",
    "convert_run",
    "default_generated",
    "format_table",
    "harness_commit",
    "harness_version",
    "is_prerelease",
    "load_board",
    "write_export",
]
