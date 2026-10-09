# Converse-Diarization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ocular AI's open-source evaluation harness for **Converse-Diarization**, the speaker-diarization benchmark in
the [Ocular AI Converse](https://www.useocular.com/converse) family, released under the
[MIT License](LICENSE). It mixes the two separately recorded speaker tracks of an Ocular AI conversation into
the single channel a diarization model hears, asks the model who spoke when, and reports **diarization error
rate (DER)**, **Jaccard error rate (JER)** and **speaker-count accuracy** against the word-level reference.

Converse-Diarization is run independently by Ocular AI: every number on the board is produced by the code in
this repository, so anyone can reproduce a score or add a model. Its sibling,
[Converse-STT](https://github.com/OcularEngineering/converse-stt), scores transcription on the same
recordings.

## About the benchmark

**Converse** is Ocular AI's family of benchmarks for AI on real-world, two-person conversation.
**Converse-Diarization** measures how accurately speaker-diarization models tell two people apart in real,
unscripted American English conversation when both voices arrive on one channel, as they do in phone calls,
meetings and field recordings. A lower DER is a better segmentation.

**The recordings.** Ocular AI's Hi-Fi conversational dataset: four American English conversations of about
15 minutes each, roughly 60 minutes of speech across eight speaker tracks. Each speaker is recorded on a
separate microphone at 48 kHz and delivered as 24-bit mono FLAC with verbatim, time-aligned transcripts. The
recordings keep the pauses, fillers, repetitions and false starts of ordinary talk, including the moments
when both people speak at once. Because each voice sits on its own track, the reference says exactly who was
speaking when; the harness sums the tracks to produce the mixed input.

**The score.** DER is the share of reference speaker time that a model gets wrong: speech it missed, speech
it invented (false alarm) and speech it gave to the wrong person (confusion), all measured in seconds and
divided by the total time anyone was speaking. Overlapping speech counts for every speaker involved and is
never excluded. Alongside DER the harness reports DER with a ±0.25 s boundary collar, JER (a per-speaker
Jaccard error that does not let a talkative speaker dominate) and whether the model found the right number of
speakers. Every conversation is also tagged with its share of overlapping speech, so results can be read for
low-, medium- and high-overlap conversations separately.

A public sample with recordings and reference transcripts is on Hugging Face; it is a sample of a larger
collection that Ocular AI licenses.

## Quick start

```bash
cd "Converse Diarization"
uv sync --group dev                 # hosted providers only
uv sync --group dev --extra local   # also installs pyannote.audio + torch for local models
cp .env.example .env                # add HF_TOKEN and OPENROUTER_API_KEY (plus any direct vendor keys you use)

uv run converse-diarization dataset pull        # Ocular AI English sample from Hugging Face -> data/converse-en-sample
uv run converse-diarization dataset show        # conversations, speaker time, overlap share
uv run converse-diarization check               # credentials + dependencies for every model in models.toml
uv run converse-diarization run --model deepgram-nova-3   # mix + diarize + score -> runs/deepgram-nova-3
uv run converse-diarization report              # every scored run in runs/ -> reports/converse-diarization.html
open reports/converse-diarization.html
```

`.env` is loaded automatically from the working directory and never committed.

## Adding a model

Models live in `models.toml` at the project root. There are two ways to reach a hosted model; both end up as
the same kind of entry, and you can mix them freely.

### Through OpenRouter (one key, many models)

[OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/stt) fronts a catalog of speech-to-text
vendors behind one `/audio/transcriptions` endpoint and one key. Diarization is not a normalised parameter
there: each vendor is switched on through its own option (Deepgram `diarize`, Azure `diarization.enabled`,
ElevenLabs `diarize`, AssemblyAI `speaker_labels`), which the `openrouter` provider sends for you. The entries
in `models.toml` marked `via OpenRouter` need nothing more than `OPENROUTER_API_KEY` in `.env`. To add another
model from the catalog, pass its OpenRouter id:

```bash
uv run converse-diarization models add <model-name> --openrouter --model <vendor>/<model-id> --vendor <Vendor>
uv run converse-diarization run --model <model-name>
```

For a vendor whose diarization switch the harness does not know yet, give it under its OpenRouter endpoint
slug (the `tag` in `GET https://openrouter.ai/api/v1/models/<vendor>/<model-id>/endpoints`):

```bash
uv run converse-diarization models add <model-name> --openrouter --model <vendor>/<model-id> \
    --option 'provider_options={"<slug>": {"<diarization option>": true}}'
```

Two caveats come with the proxy. OpenRouter times out upstream requests after about a minute of processing,
so slow vendors can fail on the 15-minute conversations (failures land in `runs/<name>/failures.jsonl` and
are resumable), and the route adds a hop to latency, so every routed entry carries `notes = "via OpenRouter"`
and the leaderboard can say so. A vendor that returns no speaker labels through OpenRouter is reported as
such rather than scored.

### Directly, with the vendor's URL and key

For any API that returns speaker-labelled segments from the OpenAI `POST /audio/transcriptions` protocol
(OpenAI's `diarized_json`, or a compatible server), a model is a URL and a key:

```bash
uv run converse-diarization models add <model-name> --url https://api.<provider>.com/v1 --key <your-api-key>
#   stored the key in .env as <PROVIDER>_API_KEY
#   added [<model-name>] to models.toml
#   <model-name>: ready (openai-compatible:<model-name>). Run it: converse-diarization run --model <model-name>
uv run converse-diarization run --model <model-name>
uv run converse-diarization report
```

For example, OpenAI's diarizing transcription model:

```bash
uv run converse-diarization models add gpt-4o-transcribe-diarize --url https://api.openai.com/v1 --key sk-... --vendor OpenAI
```

`--key` takes either the secret itself (written to `.env`, never to `models.toml`) or the NAME of a variable
that already holds it (`--key OPENAI_API_KEY`). Use `--key -` to paste the secret on stdin and keep it out of
shell history. Useful extras: `--model <api-model-id>` when the id the API expects differs from the name,
`--vendor` and `--display-name` for the leaderboard, `--provider` for APIs with their own protocol
(`deepgram`, `assemblyai`, `elevenlabs`), `--provider pyannote` for a local pyannote pipeline, and
`--provider rttm --dir <folder>` to score RTTM files produced anywhere else (a GPU box, another toolkit),
`--option key=value` for anything else (JSON values such as `{"a": 1}` become nested tables). To run the same
model both ways, give the direct entry its own name, as `models.toml` does with `deepgram-nova-3` and
`deepgram-nova-3-direct`.

Editing the file by hand is equally fine:

```toml
[<model-name>]                           # direct
url = "https://api.<provider>.com/v1"    # base URL
key = "<PROVIDER>_API_KEY"               # NAME of the variable in .env
model = "<api-model-id>"                 # optional, defaults to the table name
vendor = "<Provider>"                    # optional, leaderboard label

[<model-name>-openrouter]                # the same model through OpenRouter
provider = "openrouter"
model = "<vendor>/<model-id>"
key = "OPENROUTER_API_KEY"
vendor = "<Provider>"
notes = "via OpenRouter"
```

`converse-diarization models` lists everything, `models show <name>` prints the resolved config,
`models remove <name>` drops an entry (its `runs/<name>` directory is kept). Names in `models.toml` shadow the
built-in configs in `src/converse_diarization/models/`.

## How a score is produced

1. **Dataset.** `dataset pull` downloads the Hugging Face AudioFolder at a pinned revision and writes
   `manifest.json`: one conversation per session, one track per speaker microphone, with the original
   transcript, word timings shifted onto the shared conversation timeline, the recording offset between the
   two microphones, and a SHA-256 of every FLAC. `dataset verify` re-checks the audio against the manifest.
2. **Mix.** For each conversation the two tracks are decoded, resampled to 16 kHz, delayed by their recording
   offsets so they line up, summed sample-wise (never averaged or ducked, so relative levels survive), scaled
   down only if the sum would clip, and written as mono PCM16 WAV under `data/<dataset>/mix/`. The policy id
   (`mono-mix-v1`) and a SHA-256 of every mix are recorded in `mix.json` and in each run. `run` builds mixes
   on demand; `dataset mix` builds them ahead of time.
3. **Reference.** Each track's reference words become speech regions for that speaker: consecutive words
   separated by at most 0.2 s are merged into one region, longer gaps stay silence. The two tracks' regions
   together form the reference timeline; where they overlap, both speakers are active.
4. **Diarization.** `run` sends each mix to the provider named in the model config and appends the speaker
   segments it returns to `hypotheses.jsonl`. Models that emit speaker turns (dedicated diarization models,
   segment-timestamp transcription models) are scored exactly as returned. Models that emit speaker-labelled
   words have their words merged with the same 0.2 s rule as the reference. Runs are resumable: conversations
   whose mix hash already has a hypothesis for the same model config are skipped. Failures land in
   `failures.jsonl` with the API key redacted; a run directory is locked while in use.
5. **Scoring.** Reference and hypothesis are compared on exact intervals (no frame quantization). Speaker
   labels are matched one-to-one by the assignment that maximizes correctly attributed time. DER is reported
   with no collar (primary) and with a ±0.25 s no-score collar around reference boundaries; JER and speaker
   count come from the same timelines. Per-conversation rows go to `scores.jsonl` (with both timelines), the
   hypothesis timeline is also written as `rttm/<conversation>.rttm`, and `summary.json` holds the corpus
   figures. Corpus DER is **pooled**: total error seconds over total reference speaker seconds, never an
   average of percentages. Tier figures (low < 10 %, medium 10–20 %, high ≥ 20 % overlapping speech) are
   pooled the same way.
6. **Report.** `report` renders a self-contained HTML leaderboard from every scored run in `runs/` (or the
   `--run` directories you name): ranked DER bars with the missed / false-alarm / confusion split under each,
   toggles for collared DER and JER, an overlap-tier filter, and for every conversation a timeline that draws
   the reference speakers beside the model's prediction with overlapping speech marked.

`rttm` exports the reference timelines as RTTM files and `compare` scores one RTTM against another, so the
harness's numbers can be checked against any other scorer.

## Model configs

`models.toml` is the short form described above. Under the hood every entry becomes a full model config;
built-in examples live in `src/converse_diarization/models/*.toml`, and any TOML file with that shape works
with `--model path/to/file.toml`:

```toml
name = "gpt-4o-transcribe-diarize"
display_name = "GPT-4o Transcribe Diarize"
vendor = "OpenAI"
provider = "openai-compatible"   # openai-compatible | openrouter | deepgram | assemblyai | elevenlabs | pyannote | rttm
model = "gpt-4o-transcribe-diarize"
env = ["OPENAI_API_KEY"]
href = "https://platform.openai.com/docs/guides/speech-to-text"

[options]
response_format = "diarized_json"
chunking_strategy = "auto"
upload = "mp3"                   # 25 MB upload cap
```

| provider            | what it calls                                                           | output scored as |
| ------------------- | ----------------------------------------------------------------------- | ---------------- |
| `openai-compatible` | `POST {base_url}/audio/transcriptions` with a diarizing response format | segments (or words when that is all the API labels) |
| `openrouter`        | `POST https://openrouter.ai/api/v1/audio/transcriptions` (JSON, `verbose_json`) with the serving vendor's diarization option under `provider.options` | words, merged with the 0.2 s rule (segments as fallback) |
| `deepgram`          | `POST {base_url}/listen?diarize=true&utterances=true`                   | words, merged with the 0.2 s rule |
| `assemblyai`        | upload, create transcript with `speaker_labels`, poll                   | words, merged with the 0.2 s rule |
| `elevenlabs`        | `POST {base_url}/speech-to-text` with `diarize=true`                    | words, merged with the 0.2 s rule |
| `pyannote`          | local `pyannote.audio` pipeline (`uv sync --extra local`, `HF_TOKEN`)   | segments |
| `rttm`              | reads `<dir>/<conversation>.rttm` produced elsewhere                     | segments |

Options never contain credentials; keys are read from the environment variables listed in `env`.

## Layout

```
models.toml       the models this harness runs (url + key per model)
src/converse_diarization/
  dataset.py      Hugging Face / local AudioFolder -> manifest.json
  mix.py          two speaker tracks -> one 16 kHz mono WAV (mono-mix-v1)
  reference.py    word timings -> speaker timelines (0.2 s merge)
  metrics.py      exact-interval DER / JER / speaker count, overlap share, RTTM
  config.py       model TOML loading
  registry.py     models.toml parsing / `models add`
  providers/      openai_compatible, openrouter, deepgram, assemblyai, elevenlabs, pyannote, rttm (+ retrying HTTP, audio prep)
  runner.py       resumable diarization + scoring of a run directory
  report.py       leaderboard HTML
  models/         built-in model configs
  templates/      leaderboard.html
tests/            pytest, no network
data/             datasets and mixes (gitignored)
runs/             run directories (gitignored)
reports/          generated HTML (gitignored)
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Roadmap

- **More models.** Keep adding hosted and local diarization models to `models.toml` so the board covers every
  system we are asked about; contributions of new entries are welcome.
- **More providers.** Native adapters for diarization APIs and toolkits that do not fit the current six
  backends (streaming / WebSocket vendors, NeMo-style local pipelines).
- **Speaker-attributed transcription.** Score transcript text per speaker (cpWER) on the same mixes, bridging
  Converse-Diarization and Converse-STT.
- **More datasets and languages.** Run the same harness over additional Ocular AI conversation sets as they
  are published, including non-English ones and conversations with more than two speakers.
- **Latency and cost.** Record time-to-result and per-minute pricing next to accuracy.
- **Published board sync.** Push verified harness results straight to the public
  [Converse](https://www.useocular.com/converse) leaderboards.

## License

Copyright (c) 2026 Ocular AI. This project is released under the [MIT License](LICENSE); you are free to
use, modify and redistribute it, including commercially, as long as the copyright and permission notice are
kept. Contributions are accepted under the same license.

The harness code is MIT; the Ocular AI conversation datasets it downloads from Hugging Face carry their own
terms, stated on each dataset card. Model outputs and scores you generate belong to you.

## Contact

Questions about the benchmark, a model you would like on the board, or the Hi-Fi conversational datasets:
[research@useocular.com](mailto:research@useocular.com). Bugs and contributions to the harness are welcome as
GitHub issues and pull requests.
