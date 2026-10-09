"""Command line entry point: ``converse-diarization``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from converse_diarization import __version__
from converse_diarization.config import ConfigError, builtin_model_names, load_model_config
from converse_diarization.env import load_dotenv
from converse_diarization.registry import (
    DEFAULT_PROVIDER,
    DEFAULT_REGISTRY,
    Registry,
    looks_like_env_name,
    suggest_env_name,
    upsert_dotenv,
)

DEFAULT_DATASET = Path("data/converse-en-sample")
DEFAULT_RUNS = Path("runs")
DEFAULT_REPORT = Path("reports/converse-diarization.html")
PROVIDER_CHOICES = ("openai-compatible", "deepgram", "assemblyai", "elevenlabs", "pyannote", "rttm")
URL_PROVIDERS = {"openai-compatible", "deepgram", "assemblyai", "elevenlabs"}
KEYLESS_PROVIDERS = {"pyannote", "rttm"}


def _fail(message: str) -> int:
    print(f"converse-diarization: error: {message}", file=sys.stderr)
    return 2


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# --- dataset -----------------------------------------------------------------


def cmd_dataset_pull(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, pull_dataset

    try:
        dataset = pull_dataset(args.output, repo_id=args.repo, revision=args.revision, token=args.token)
    except DatasetError as exc:
        return _fail(str(exc))
    print(f"{len(dataset.conversations)} conversations, {len(dataset.tracks)} tracks -> {dataset.root}")
    return 0


def cmd_dataset_build(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, build_from_folder

    try:
        dataset = build_from_folder(args.source, args.output)
    except DatasetError as exc:
        return _fail(str(exc))
    print(f"{len(dataset.conversations)} conversations, {len(dataset.tracks)} tracks -> {dataset.root}")
    return 0


def cmd_dataset_verify(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, load_dataset, verify_dataset

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        return _fail(str(exc))
    problems = verify_dataset(dataset)
    for problem in problems:
        print(f"  - {problem}")
    if problems:
        return 1
    print(f"ok: {len(dataset.tracks)} tracks match the manifest (digest {dataset.digest()})")
    return 0


def cmd_dataset_show(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, load_dataset
    from converse_diarization.metrics import overlap_ratio, overlap_tier, speaking_time
    from converse_diarization.reference import reference_timeline

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        return _fail(str(exc))
    source = dataset.source
    print(f"{dataset.root}  digest {dataset.digest()}")
    origin = source.get("repo_id") or source.get("path") or ""
    print(f"source: {source.get('kind')} {origin} {source.get('revision') or ''}".rstrip())
    total_speech = 0.0
    for conversation in dataset.conversations:
        reference = reference_timeline(conversation)
        ratio = overlap_ratio(reference)
        per_speaker = speaking_time(reference)
        total_speech += sum(per_speaker.values())
        print(
            f"{conversation.id}  {conversation.language}  {conversation.duration_s / 60:.1f} min  "
            f"overlap {ratio * 100:.1f}% ({overlap_tier(ratio)})"
        )
        for track in conversation.tracks:
            print(
                f"  {track.id}  speaker {track.speaker_id}  offset {track.offset_s:+.3f}s  "
                f"{per_speaker.get(track.id, 0.0):.1f}s speech in {len(track.words):,} words"
            )
    print(f"{len(dataset.conversations)} conversations, {total_speech / 60:.1f} min of reference speaker time")
    return 0


def cmd_dataset_mix(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, load_dataset
    from converse_diarization.mix import MixError, mix_dataset

    try:
        dataset = load_dataset(args.dataset)
        results = mix_dataset(dataset, rebuild=args.rebuild)
    except (DatasetError, MixError) as exc:
        return _fail(str(exc))
    for result in results:
        print(f"{result.path}  {result.duration_s:.1f}s  sha256 {result.sha256[:16]}")
    print(f"{len(results)} mixes ready")
    return 0


# --- models ------------------------------------------------------------------


def cmd_models_list(args: argparse.Namespace) -> int:
    registry = Registry.load(args.models_file)
    rows = []
    for name in registry.names():
        try:
            config = registry.config(name)
        except ConfigError as exc:
            rows.append((name, "invalid", str(exc), "", ""))
            continue
        rows.append((name, "models.toml", config.provider, config.model, config.options.get("base_url", "")))
    for name in builtin_model_names():
        if name in registry:
            continue
        config = load_model_config(name, registry=args.models_file)
        rows.append((name, "built-in", config.provider, config.model, config.options.get("base_url", "")))
    if not rows:
        print("no models yet. Add one: converse-diarization models add <name> --url <base url> --key <ENV_NAME>")
        return 0
    width = max(len(row[0]) for row in rows)
    for name, source, provider, model, base_url in rows:
        tail = f"  {base_url}" if base_url else ""
        print(f"{name:{width}}  {source:11} {provider:18} {model}{tail}")
    if not registry.path.is_file():
        print(f"\n({registry.path} does not exist yet; `models add` creates it)")
    return 0


def _parse_options(items: list[str] | None) -> dict:
    options: dict = {}
    for item in items or ():
        if "=" not in item:
            raise ConfigError(f"--option expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            options[key] = lowered == "true"
            continue
        try:
            options[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            options[key] = float(raw)
            continue
        except ValueError:
            pass
        options[key] = raw
    return options


def cmd_models_add(args: argparse.Namespace) -> int:
    from converse_diarization.providers import PROVIDERS, create_provider

    registry = Registry.load(args.models_file)
    provider_key = args.provider or DEFAULT_PROVIDER
    if provider_key not in PROVIDERS:
        return _fail(f"unknown provider {provider_key!r}; available: {', '.join(sorted(PROVIDERS))}")
    if provider_key in URL_PROVIDERS and not args.url:
        return _fail("--url is required: the base URL of the API, for example https://api.vendor.com/v1")

    key_env: str | None = None
    secret: str | None = None
    if args.key == "-":
        args.key = sys.stdin.readline().strip()
    if args.key:
        if looks_like_env_name(args.key):
            key_env = args.key.strip()
        else:
            secret = args.key.strip()
            key_env = args.key_env or suggest_env_name(args.name)
    elif args.key_env:
        key_env = args.key_env
    elif provider_key not in KEYLESS_PROVIDERS:
        return _fail("--key is required: either the secret itself or the NAME of the variable that holds it in .env")

    try:
        options = _parse_options(args.option)
    except ConfigError as exc:
        return _fail(str(exc))
    if args.upload:
        options["upload"] = args.upload
    if args.response_format:
        options["response_format"] = args.response_format
    if args.dir:
        options["dir"] = args.dir

    try:
        config = registry.add(
            args.name,
            url=args.url,
            key_env=key_env,
            model=args.model,
            provider=provider_key,
            vendor=args.vendor,
            display_name=args.display_name,
            href=args.href,
            notes=args.notes,
            options=options or None,
            replace=args.replace,
        )
    except ConfigError as exc:
        return _fail(str(exc))

    if secret and key_env:
        upsert_dotenv(args.env_file, key_env, secret)
        os.environ[key_env] = secret
        print(f"stored the key in {args.env_file} as {key_env}")
    registry.save()
    print(f"added [{config.name}] to {registry.path}")

    problems = create_provider(config).check()
    if problems:
        print(f"{config.name}: not ready")
        for problem in problems:
            print(f"  - {problem}")
        if key_env and any(key_env in problem for problem in problems):
            print(f"  add a line to {args.env_file}:  {key_env}=<your key>")
        return 1
    print(
        f"{config.name}: ready ({config.provider}:{config.model}). "
        f"Run it: converse-diarization run --model {config.name}"
    )
    return 0


def cmd_models_show(args: argparse.Namespace) -> int:
    try:
        config = load_model_config(args.name, registry=args.models_file)
    except ConfigError as exc:
        return _fail(str(exc))
    data = config.to_dict()
    data["source"] = config.source
    _print_json(data)
    return 0


def cmd_models_remove(args: argparse.Namespace) -> int:
    registry = Registry.load(args.models_file)
    try:
        registry.remove(args.name)
    except ConfigError as exc:
        return _fail(str(exc))
    registry.save()
    print(f"removed [{args.name}] from {registry.path} (runs/{args.name} is untouched)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from converse_diarization.providers import create_provider

    names = list(args.model or []) + list(args.model_flag or [])
    if not names:
        names = Registry.load(args.models_file).names()
        if not names:
            return _fail("no models to check; name one or add some to models.toml")
    status = 0
    for value in names:
        try:
            config = load_model_config(value, registry=args.models_file)
            provider = create_provider(config)
        except ConfigError as exc:
            print(f"{value}: {exc}")
            status = 1
            continue
        problems = provider.check()
        if problems:
            status = 1
            print(f"{config.name}: not ready")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"{config.name}: ready ({config.provider}:{config.model})")
    return status


# --- run / score / report -------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, load_dataset
    from converse_diarization.runner import RunError, RunOptions, diarize_dataset, format_summary, score_run

    try:
        config = load_model_config(args.model, registry=args.models_file)
        dataset = load_dataset(args.dataset)
    except (ConfigError, DatasetError) as exc:
        return _fail(str(exc))
    output = args.output or args.runs_dir / config.name
    try:
        options = RunOptions(
            resume=not args.fresh,
            conversation_ids=tuple(args.conversation or ()),
            max_consecutive_failures=args.max_failures,
            collar_s=args.collar,
        )
        counts = diarize_dataset(dataset, config, output, options=options)
        print(
            f"{counts['diarized']} diarized, {counts['skipped']} already done, "
            f"{counts['failed']} failed of {counts['selected']} selected conversations",
            file=sys.stderr,
        )
        if counts["diarized"] + counts["skipped"] == 0:
            return 1
        summary = score_run(output, dataset=dataset, collar_s=args.collar)
    except RunError as exc:
        return _fail(str(exc))
    print(format_summary(summary))
    print(f"run directory: {output}")
    return 0 if counts["failed"] == 0 else 1


def cmd_score(args: argparse.Namespace) -> int:
    from converse_diarization.dataset import DatasetError, load_dataset
    from converse_diarization.runner import RunError, format_summary, score_run

    try:
        dataset = load_dataset(args.dataset) if args.dataset else None
        summary = score_run(args.run, dataset=dataset, collar_s=args.collar)
    except (RunError, DatasetError) as exc:
        return _fail(str(exc))
    if args.json:
        _print_json(summary)
    else:
        print(format_summary(summary))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from converse_diarization.report import ReportError, discover_runs, write_report

    run_dirs = list(args.run or [])
    if not run_dirs:
        run_dirs = discover_runs(args.runs_dir)
        if not run_dirs:
            return _fail(f"no scored runs under {args.runs_dir}; run a model first or pass --run")
    try:
        output = write_report(run_dirs, args.output, sample_label=args.label)
    except ReportError as exc:
        return _fail(str(exc))
    print(f"{output}  ({len(run_dirs)} run{'s' if len(run_dirs) != 1 else ''})")
    return 0


# --- rttm ------------------------------------------------------------------------


def cmd_rttm(args: argparse.Namespace) -> int:
    """Write the reference timelines as RTTM files so other scorers can be compared against this one."""
    from converse_diarization.dataset import DatasetError, load_dataset
    from converse_diarization.metrics import write_rttm
    from converse_diarization.reference import reference_timeline

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        return _fail(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    for conversation in dataset.conversations:
        path = args.output / f"{conversation.id}.rttm"
        path.write_text(write_rttm(reference_timeline(conversation), conversation.id), encoding="utf-8")
        print(path)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Score one hypothesis RTTM against one reference RTTM."""
    from converse_diarization.metrics import der, jer, percent, read_rttm, speaker_count

    reference = read_rttm(args.reference.read_text(encoding="utf-8"))
    hypothesis = read_rttm(args.hypothesis.read_text(encoding="utf-8"))
    plain = der(reference, hypothesis, collar_s=0.0)
    collared = der(reference, hypothesis, collar_s=args.collar)
    print(f"DER {percent(plain.rate)}  {plain.counts.to_dict()}")
    print(f"DER ±{args.collar:.2f}s {percent(collared.rate)}")
    print(f"JER {percent(jer(reference, hypothesis).rate)}")
    count = speaker_count(reference, hypothesis)
    print(f"speakers {count.reference} reference / {count.hypothesis} hypothesis")
    return 0


# --- parser --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from converse_diarization.metrics import DEFAULT_COLLAR_S

    parser = argparse.ArgumentParser(
        prog="converse-diarization",
        description=(
            "Ocular AI Converse-Diarization harness: mix the two speaker tracks of each Ocular AI conversation, "
            "diarize the mix with a model and score DER / JER / speaker count."
        ),
    )
    parser.add_argument("--version", action="version", version=f"converse-diarization {__version__}")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="dotenv file to load (default: ./.env)")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help=f"model registry (default: ./{DEFAULT_REGISTRY} or $CONVERSE_DIARIZATION_MODELS_FILE)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS, help="where run directories live (default: runs/)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser("dataset", help="download, build, mix, inspect or verify a dataset").add_subparsers(
        dest="dataset_command", required=True
    )
    pull = dataset.add_parser("pull", help="download an Ocular AI dataset from Hugging Face")
    from converse_diarization.dataset import DEFAULT_REPO_ID, DEFAULT_REVISION

    pull.add_argument("--repo", default=DEFAULT_REPO_ID)
    pull.add_argument("--revision", default=DEFAULT_REVISION)
    pull.add_argument("--token", default=None, help="Hugging Face token (defaults to HF_TOKEN / cached login)")
    pull.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    pull.set_defaults(func=cmd_dataset_pull)
    build = dataset.add_parser("build", help="build a dataset from a local AudioFolder export")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(func=cmd_dataset_build)
    verify = dataset.add_parser("verify", help="check audio files against the manifest")
    verify.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    verify.set_defaults(func=cmd_dataset_verify)
    show = dataset.add_parser("show", help="list conversations, speaker time and overlap")
    show.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    show.set_defaults(func=cmd_dataset_show)
    mix = dataset.add_parser("mix", help="build the 16 kHz mono mixes the models hear (run does this on demand)")
    mix.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    mix.add_argument("--rebuild", action="store_true", help="rebuild even when a verified mix exists")
    mix.set_defaults(func=cmd_dataset_mix)

    models = sub.add_parser("models", help="list, add, show or remove models in models.toml")
    models.set_defaults(func=cmd_models_list)
    models_sub = models.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list registered and built-in models").set_defaults(func=cmd_models_list)
    add = models_sub.add_parser(
        "add",
        help="register a model: a name, the API base URL and the key are enough",
        description=(
            "Register a model in models.toml. For an OpenAI-compatible diarizing endpoint only --url and --key are "
            "needed. --key accepts either the secret itself (stored in .env, never in models.toml) or the NAME of an "
            "environment variable that already holds it. Use `--key -` to read the secret from stdin and keep it "
            "out of your shell history."
        ),
    )
    add.add_argument("name", help="short id, e.g. gpt-4o-transcribe-diarize (also the default model id and run dir)")
    add.add_argument("--url", help="API base URL, e.g. https://api.vendor.com/v1")
    add.add_argument("--key", help="the API key, or the NAME of the .env variable holding it, or '-' for stdin")
    add.add_argument("--key-env", help="variable name to store a literal --key under (default: <VENDOR>_API_KEY)")
    add.add_argument("--model", help="model id sent to the API when it differs from the name")
    add.add_argument("--provider", choices=PROVIDER_CHOICES, default=None)
    add.add_argument("--vendor", help="shown on the leaderboard, e.g. OpenAI")
    add.add_argument("--display-name", help="shown on the leaderboard, e.g. 'GPT-4o Transcribe Diarize'")
    add.add_argument("--href", help="link for the leaderboard row")
    add.add_argument("--notes")
    add.add_argument("--upload", choices=("flac", "wav16k", "mp3"), help="audio format to send (provider default)")
    add.add_argument("--response-format", help="for openai-compatible endpoints (default diarized_json)")
    add.add_argument("--dir", help="for --provider rttm: directory holding <conversation>.rttm files")
    add.add_argument("--option", action="append", metavar="KEY=VALUE", help="extra provider option (repeatable)")
    add.add_argument("--replace", action="store_true", help="overwrite an existing entry with the same name")
    add.set_defaults(func=cmd_models_add)
    show = models_sub.add_parser("show", help="print the resolved config for one model")
    show.add_argument("name")
    show.set_defaults(func=cmd_models_show)
    remove = models_sub.add_parser("remove", help="delete a model from models.toml")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_models_remove)

    check = sub.add_parser("check", help="verify credentials and dependencies (default: every model in models.toml)")
    check.add_argument("model", nargs="*", help="model name or path to a TOML config")
    check.add_argument("--model", dest="model_flag", action="append", help="same as the positional form")
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="diarize the mixed conversations with a model and score it")
    run.add_argument("--model", required=True, help="model name (models.toml or built-in) or path to a TOML config")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--output", type=Path, default=None, help="run directory (default: runs/<model name>)")
    run.add_argument("--conversation", action="append", help="only these conversation ids (repeatable)")
    run.add_argument("--fresh", action="store_true", help="discard previous hypotheses in the output directory")
    run.add_argument("--max-failures", type=int, default=3, help="stop after this many consecutive failures")
    run.add_argument("--collar", type=float, default=DEFAULT_COLLAR_S, help="collar for the secondary DER (s)")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="rescore an existing run")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--dataset", type=Path, default=None, help="override the dataset recorded in run.json")
    score.add_argument("--collar", type=float, default=None, help="collar for the secondary DER (s)")
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=cmd_score)

    report = sub.add_parser("report", help="build the leaderboard HTML from scored runs (default: every run in runs/)")
    report.add_argument(
        "--run", type=Path, action="append", help="run directory (repeatable); default: every scored run"
    )
    report.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    report.add_argument("--label", default="This sample", help="label for harness rows")
    report.set_defaults(func=cmd_report)

    rttm = sub.add_parser("rttm", help="export the reference speaker timelines as RTTM files")
    rttm.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    rttm.add_argument("--output", type=Path, default=Path("reference-rttm"))
    rttm.set_defaults(func=cmd_rttm)

    compare = sub.add_parser("compare", help="score one hypothesis RTTM against one reference RTTM")
    compare.add_argument("reference", type=Path)
    compare.add_argument("hypothesis", type=Path)
    compare.add_argument("--collar", type=float, default=DEFAULT_COLLAR_S)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(args.env_file)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
