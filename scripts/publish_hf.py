"""Publish the exported models to the Hugging Face Hub.

Without this the repo is code you would have to train from scratch to try. It
uploads the *shipping* variant of each branch — chosen by the measured latency in
each export report, not by assuming INT8 is always better — plus the tokenizer,
the configs, and a model card carrying the results and the limitations.

    export HF_TOKEN=hf_...            # write token from hf.co/settings/tokens
    python scripts/publish_hf.py --repo-id Nikhils-G/turnwave
"""

import argparse
import json
import os
from pathlib import Path

CARD = """---
license: apache-2.0
library_name: onnx
pipeline_tag: audio-classification
tags:
  - end-of-turn-detection
  - turn-taking
  - voice-agents
  - speech
  - onnx
---

# TurnWave — end-of-turn detection for voice agents

Decides whether a caller has **finished speaking** or is merely pausing, so a
voice agent neither interrupts them nor leaves an awkward silence. It replaces
the fixed 300–700 ms silence timeout that most pipelines still use.

Two branches, both **trained from scratch** — no pretrained weights anywhere:

- **Text**: a 6.92M-param causal transformer (RoPE, RMSNorm, SwiGLU, causal
  self-attention) over the transcript tail. Judges semantic completeness.
- **Audio**: a 3.49M-param CNN over log-mel spectrograms. Hears prosody — a
  speaker who intends to continue holds pitch level and trails energy into the
  pause; a finished speaker drops pitch and closes.
- **Fused**: a head over both. Each branch alone sees half the evidence.

The log-mel front end is also hand-written (`torch.stft` plus a mel filterbank),
so there is no torchaudio or librosa dependency.

## Results

{results_table}

{results_claim}

## Latency (one CPU thread)

{latency_table}

INT8 is not applied blindly. Dynamic quantization rewrites MatMul, so it speeds
up the transformer and *slows down* the conv-heavy branches; the shipping variant
for each model is whichever measured faster.

## Usage

```python
from huggingface_hub import hf_hub_download
from turnwave.infer import TurnDetector   # pip install git+https://github.com/Nikhils-G/turnwave

detector = TurnDetector(
    hf_hub_download("{repo_id}", "fusion_eot.onnx"),
    tokenizer=hf_hub_download("{repo_id}", "spm.model"),
)
probability = detector.predict(audio=wav_16k, text="i want a large pepperoni and")
if probability > 0.5:
    respond()
```

Audio is 16 kHz mono; the model reads the last 2 seconds ending at the decision
point, which sits 0.2 s into the pause — where a real agent decides, and where
LiveKit's eot-bench scores.

## Limitations

- **English only.** Trained on English; other languages are untested.
- **Trained on isolated utterances.** The corpus has no dialogue context, so the
  text branch runs without a previous turn and degrades relative to its
  DailyDialog performance (AP 0.888 there vs {text_ap} here). A text-only
  detector needs conversational context; this is one reason the fused model
  matters.
- **The acoustic branch is small and trained from scratch**, against competitors
  that start from pretrained speech encoders. The gap is expected and reported
  rather than hidden.

## Training data

- [Scicom-intl/semantic-vad-eot](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot) (CC-BY-4.0) — audio with word-level timings and pause spans
- [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) — text branch pretraining pairs

Code, training scripts and the full write-up: **https://github.com/Nikhils-G/turnwave**
"""

BRANCHES = ("text_eot", "audio_eot", "fusion_eot")


def shipping_file(onnx_dir: Path, name: str) -> Path:
    """The variant the export report measured as faster."""
    report_path = onnx_dir / f"{name}.export.json"
    if not report_path.exists():
        raise SystemExit(f"missing {report_path} — run turnwave.export first")
    report = json.loads(report_path.read_text())
    return Path(report[report["recommended"]]["path"])


FUSION_WINS = ("Fusion beats both branches. That is the claim the project was built "
               "to test:\nprosody carries end-of-turn information the transcript does not.")
FUSION_LOSES = ("On this evaluation the fused model did **not** beat the best single "
                "branch.\nThe number is reported as measured.")


def build_tables(onnx_dir: Path, ablation: Path | None) -> tuple[str, str, str, str]:
    latency_rows = ["| model | variant | latency | size |", "|---|---|---|---|"]
    for name in BRANCHES:
        report = json.loads((onnx_dir / f"{name}.export.json").read_text())
        best = report[report["recommended"]]
        latency_rows.append(f"| {name.replace('_eot', '')} | {report['recommended']} | "
                            f"{best['latency_ms']:.2f} ms | {best['mb']:.1f} MB |")

    if not (ablation and ablation.exists()):
        raise SystemExit(
            f"no ablation results at {ablation}. A model card whose Results section "
            f"says 'not available' while the text claims fusion wins is worse than no "
            f"card at all.\nRun: python -m turnwave.ablate ... --out {ablation}")

    data = json.loads(ablation.read_text())
    if True:
        rows = ["| model | acc | precision | recall | F1 | AP |", "|---|---|---|---|---|---|"]
        text_ap = "n/a"
        for label, m in data["results"].items():
            emphasis = "**" if label.startswith("fused") else ""
            rows.append(f"| {emphasis}{label}{emphasis} | {m['accuracy']:.3f} | "
                        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                        f"{emphasis}{m['ap']:.3f}{emphasis} |")
            if label == "text only":
                text_ap = f"{m['ap']:.3f}"
        results = f"Held-out test set, {data['examples']:,} examples:\n\n" + "\n".join(rows)

    # The claim is derived from the numbers, never asserted alongside them.
    scores = {label: m["ap"] for label, m in data["results"].items()}
    fused = max((ap for label, ap in scores.items() if label.startswith("fused")), default=None)
    best_single = max((ap for label, ap in scores.items()
                       if label in ("text only", "audio only")), default=None)
    claim = FUSION_WINS if (fused is not None and best_single is not None
                            and fused > best_single) else FUSION_LOSES
    return results, "\n".join(latency_rows), text_ap, claim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Nikhils-G/turnwave")
    ap.add_argument("--onnx-dir", type=Path, default=Path("checkpoints/onnx"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer/spm.model"))
    ap.add_argument("--ablation", type=Path, default=Path("docs/ablation.json"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card locally, upload nothing")
    args = ap.parse_args(argv)

    results_table, latency_table, text_ap, claim = build_tables(args.onnx_dir, args.ablation)
    card = CARD.format(results_table=results_table, latency_table=latency_table,
                       results_claim=claim, repo_id=args.repo_id, text_ap=text_ap)

    uploads = {shipping_file(args.onnx_dir, name).name: shipping_file(args.onnx_dir, name)
               for name in BRANCHES}
    uploads[args.tokenizer.name] = args.tokenizer

    if args.dry_run:
        Path("docs/model_card.md").write_text(card)
        print("wrote docs/model_card.md")
        for name, path in uploads.items():
            print(f"  would upload {name:24s} <- {path} ({path.stat().st_size/1e6:.1f} MB)")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (a write token from hf.co/settings/tokens)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    for name, path in uploads.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=args.repo_id, repo_type="model")
        print(f"uploaded {name}")
    print(f"\nhttps://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
