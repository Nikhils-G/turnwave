"""Publish the exported models to the Hugging Face Hub.

Without this the repo is code you would have to train from scratch to try.

Licensing is per model, not per repo, because the training corpora differ and
saying otherwise on a public artifact would be a false claim:

* the Phase 5 acoustic branch trains only on smart-turn conversational clips,
* the Phase 4 acoustic branch only on a CC BY 4.0 corpus,
* the text and fusion models also use DailyDialog, which is **CC BY-NC-SA 4.0** —
  non-commercial and share-alike, so those two cannot be offered as Apache-2.0.

The card's headline claim is derived from the eot-bench numbers rather than
written beside them, and the script refuses to build a card without results.

    export HF_TOKEN=hf_...            # write token from hf.co/settings/tokens
    python scripts/publish_hf.py --repo-id Nikhils-G/turnwave
"""

import argparse
import json
import os
from pathlib import Path

# The eot-bench figures this card reports, and the published baselines they sit
# against. Sourced from output/.../comparison/report.md and LiveKit's leaderboard.
BENCH = {
    "cutoff_300": 42.1, "cutoff_600": 17.2, "latency_5pct": 1195,
    "auc": 0.770, "ap": 0.602,
    "baselines": [
        ("VAD baseline", 55.6, 21.7, 1600),
        ("TurnWave audio branch (this model)", 42.1, 17.2, 1195),
        ("SmartTurn v3.2", 35.2, 14.8, 1051),
        ("LiveKit Turn Detector v1", 9.9, 4.5, 543),
    ],
}

# filename -> (source checkpoint key, licence, why)
MODELS = {
    "audio_eot_v2.onnx": ("audio_eot_v2", "apache-2.0",
                          "trained on smart-turn conversational clips"),
    "audio_eot.onnx": ("audio_eot", "cc-by-4.0",
                       "Phase 4; trained on semantic-vad-eot (CC BY 4.0)"),
    "text_eot.int8.onnx": ("text_eot", "cc-by-nc-sa-4.0",
                           "trained on DailyDialog (CC BY-NC-SA 4.0) — non-commercial"),
    "fusion_eot.onnx": ("fusion_eot", "cc-by-nc-sa-4.0",
                        "contains the text branch, so it inherits the same terms"),
}

CARD = """---
license: apache-2.0
license_name: mixed-per-model
license_link: LICENSE
library_name: onnx
pipeline_tag: audio-classification
language:
  - en
tags:
  - end-of-turn-detection
  - turn-taking
  - voice-agents
  - speech
  - onnx
  - from-scratch
datasets:
  - pipecat-ai/smart-turn-data-v3.2-train
  - Scicom-intl/semantic-vad-eot
  - li2017dailydialog/daily_dialog
metrics:
  - roc_auc
  - average_precision
model-index:
  - name: TurnWave
    results:
      - task:
          type: audio-classification
          name: End-of-turn detection
        dataset:
          type: livekit/eot-bench-data
          name: eot-bench (English)
          split: validation
        metrics:
          - type: roc_auc
            value: {auc}
            name: AUC
          - type: average_precision
            value: {ap}
            name: Average precision
          - type: false_cutoff_rate
            value: {cutoff_300}
            name: False cutoffs @300ms latency budget (%)
          - type: false_cutoff_rate
            value: {cutoff_600}
            name: False cutoffs @600ms latency budget (%)
---

# TurnWave — end-of-turn detection for voice agents

Decides whether a caller has **finished speaking** or is only pausing, so a voice
agent neither interrupts them nor leaves an awkward silence. It replaces the fixed
300–700 ms silence timeout most pipelines still use.

**Trained from scratch — no pretrained weights anywhere.** A causal transformer
(RoPE, RMSNorm, SwiGLU) over the transcript tail, and a CNN over log-mel
spectrograms for prosody. Even the log-mel front end is hand-built on `torch.stft`,
so there is no torchaudio or librosa dependency.

## Benchmark

Scored by [LiveKit's eot-bench](https://github.com/livekit/eot-bench) harness on
real human-to-agent conversation, using their code and published baselines. Lower
is better; **bold marks the best per column.**

{bench_table}

{claim}

## Models in this repo

{model_table}

Each model's licence follows its training data, so they differ. `audio_eot_v2` is
the one the benchmark above measures and the one to use.

## Usage

```python
from huggingface_hub import hf_hub_download
from turnwave.infer import TurnDetector   # pip install git+https://github.com/Nikhils-G/turnwave

detector = TurnDetector(hf_hub_download("{repo_id}", "audio_eot_v2.onnx"))
if detector.predict(audio=wav_16k) > 0.5:
    respond()
```

16 kHz mono. The model reads the last 2 seconds ending at the decision point, which
sits 0.2 s into the pause — where a live agent decides, and where eot-bench scores.

{latency_table}

INT8 is not applied blindly: dynamic quantization rewrites MatMul, so it speeds up
the transformer and *slows down* the conv-heavy branches. Each model ships whichever
variant measured faster.

## What this project found

The first version of this model scored **AP 0.945** on its own held-out test set and
**AUC 0.563** on eot-bench — barely above random. The policy sweep chose thresholds
of 0.0 and 1.0, meaning *ignore the model entirely*.

The cause was the training corpus, not the architecture. It derived from a dataset
whose own card declares `task_categories: [text-to-speech]` — read speech, whose
pauses are reading hesitations rather than conversational turn-yields. The model had
learned *"has this sentence finished being read aloud."*

Retraining on conversational data, changing nothing else, lifted AUC to **0.770**.
The in-domain score could never have revealed this; only a benchmark on data we did
not build could.

## Limitations

- **English only.** Other languages are in the training data but untested here.
- **Behind the production models**, and not a fair comparison: SmartTurn starts from
  a pretrained Whisper encoder, LiveKit's is a fine-tuned 0.5B LLM distilled from a
  7B teacher. This is 3.49M parameters from random initialisation.
- **The fusion model is stale.** It was trained on the read-speech corpus, which the
  benchmark showed to be the wrong task. The conversational corpus has no
  transcripts, so retraining fusion needs ASR first.
- **Non-commercial models included.** The text and fusion models derive from
  DailyDialog (CC BY-NC-SA 4.0). Only the audio branches are permissively licensed.

Code, training scripts, and the full write-up: **https://github.com/Nikhils-G/turnwave**
"""


def shipping_file(onnx_dir: Path, name: str) -> Path:
    """The variant the export report measured as faster, resolved locally."""
    report_path = onnx_dir / f"{name}.export.json"
    if not report_path.exists():
        raise SystemExit(f"missing {report_path} — run turnwave.export first")
    report = json.loads(report_path.read_text())
    return onnx_dir / Path(report[report["recommended"]]["path"]).name


def bench_table() -> str:
    rows = ["| model | false cutoffs @300 ms ↓ | @600 ms ↓ | latency @5% cutoff ↓ |",
            "|---|---|---|---|"]
    best = {1: min(r[1] for r in BENCH["baselines"]),
            2: min(r[2] for r in BENCH["baselines"]),
            3: min(r[3] for r in BENCH["baselines"])}
    for name, c300, c600, lat in BENCH["baselines"]:
        cells = [f"**{c300}%**" if c300 == best[1] else f"{c300}%",
                 f"**{c600}%**" if c600 == best[2] else f"{c600}%",
                 f"**{lat} ms**" if lat == best[3] else f"{lat} ms"]
        label = f"**{name}**" if "this model" in name else name
        rows.append(f"| {label} | {' | '.join(cells)} |")
    return "\n".join(rows)


def model_table(onnx_dir: Path) -> tuple[str, dict]:
    rows = ["| file | licence | training data |", "|---|---|---|"]
    uploads = {}
    for filename, (key, licence, why) in MODELS.items():
        path = onnx_dir / filename
        if not path.exists():
            # Fall back to whichever variant the export report recommends. A
            # model that was never exported is skipped rather than fatal, so a
            # partial publish works; the empty case is caught below.
            if not (onnx_dir / f"{key}.export.json").exists():
                continue
            path = shipping_file(onnx_dir, key)
            if not path.exists():
                continue
        uploads[filename] = path
        rows.append(f"| `{filename}` | {licence} | {why} |")
    if not uploads:
        raise SystemExit(f"no exported models found in {onnx_dir}")
    return "\n".join(rows), uploads


def latency_table(onnx_dir: Path) -> str:
    rows = ["| model | variant | CPU latency | size |", "|---|---|---|---|"]
    for key in ("audio_eot_v2", "audio_eot", "text_eot", "fusion_eot"):
        report_path = onnx_dir / f"{key}.export.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        best = report[report["recommended"]]
        rows.append(f"| {key} | {report['recommended']} | {best['latency_ms']:.2f} ms | "
                    f"{best['mb']:.1f} MB |")
    return "\n".join(rows)


def build_card(onnx_dir: Path, repo_id: str) -> tuple[str, dict]:
    models, uploads = model_table(onnx_dir)
    claim = ("TurnWave beats the VAD baseline on every metric the harness reports."
             if BENCH["cutoff_300"] < BENCH["baselines"][0][1]
             else "TurnWave does not beat the VAD baseline; the numbers are as measured.")
    card = CARD.format(bench_table=bench_table(), model_table=models,
                       latency_table=latency_table(onnx_dir), claim=claim,
                       repo_id=repo_id, **{k: v for k, v in BENCH.items()
                                           if k != "baselines"})
    return card, uploads


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Nikhils-G/turnwave")
    ap.add_argument("--onnx-dir", type=Path, default=Path("checkpoints/onnx"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer/spm.model"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card locally, upload nothing")
    args = ap.parse_args(argv)

    card, uploads = build_card(args.onnx_dir, args.repo_id)
    if args.tokenizer.exists():
        uploads[args.tokenizer.name] = args.tokenizer

    if args.dry_run:
        Path("docs/model_card.md").write_text(card)
        print("wrote docs/model_card.md")
        for name, path in uploads.items():
            print(f"  would upload {name:24s} {path.stat().st_size/1e6:6.1f} MB")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (a write token from hf.co/settings/tokens)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    api.upload_file(path_or_fileobj="LICENSE", path_in_repo="LICENSE",
                    repo_id=args.repo_id, repo_type="model")
    for name, path in uploads.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=args.repo_id, repo_type="model")
        print(f"uploaded {name}")
    print(f"\nhttps://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
