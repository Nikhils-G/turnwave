"""The ablation: text alone, audio alone, and the two together.

This is the table the whole project builds toward. It is only meaningful if every
row is scored on **the same examples**, so all three models are evaluated over one
split of the audio cache — the only dataset that carries aligned audio and
transcripts. The text branch therefore sees exactly the utterances the CNN hears,
and any difference between rows is the modality, not the data.

    python -m turnwave.ablate --cache data/audio \
        --tokenizer checkpoints/tokenizer/spm.model \
        --text-ckpt checkpoints/text_eot/best.pt \
        --audio-ckpt checkpoints/audio_eot/best.pt \
        --fusion-ckpt checkpoints/fusion_eot/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.audio_loader import EOTAudioDataset, audio_collate, make_fusion_collate
from .data.loader import make_collate
from .evaluate import INCOMPLETE_LAST_WORDS
from .metrics import average_precision, binary_metrics
from .models.fusion import load_fusion
from .tokenizer import Tokenizer
from .train import evaluate


def heuristic_row(rows: list[dict]) -> dict:
    """The cue-word baseline from Phase 1, rerun here so the table has a floor."""
    probs, labels = [], []
    for row in rows:
        words = row["text"].split()
        probs.append(0.0 if words and words[-1] in INCOMPLETE_LAST_WORDS else 1.0)
        labels.append(float(row["label"]))
    out = binary_metrics(probs, labels)
    out["ap"] = average_precision(probs, labels)
    return out


def majority_row(rows: list[dict]) -> dict:
    labels = [float(r["label"]) for r in rows]
    majority = 1.0 if sum(labels) >= len(labels) / 2 else 0.0
    out = binary_metrics([majority] * len(labels), labels)
    out["ap"] = average_precision([majority] * len(labels), labels)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--text-ckpt", type=Path, required=True)
    ap.add_argument("--audio-ckpt", type=Path, required=True)
    ap.add_argument("--fusion-ckpt", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="write the table as JSON")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    tok = Tokenizer(args.tokenizer)

    from .export import load_any

    text_model, _ = load_any(args.text_ckpt, device)
    audio_model, _ = load_any(args.audio_ckpt, device)

    dataset = EOTAudioDataset(args.cache, args.split, tokenizer=tok)
    rows = dataset.rows

    def loader(collate):
        return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    results = [
        ("majority class", majority_row(rows)),
        ("cue-word heuristic", heuristic_row(rows)),
        ("text only", evaluate(text_model, loader(make_collate(tok.pad_id)), device)),
        ("audio only", evaluate(audio_model, loader(audio_collate), device)),
    ]
    if args.fusion_ckpt:
        fused = load_fusion(args.fusion_ckpt, device)
        results.append(("fused (text + audio)",
                        evaluate(fused, loader(make_fusion_collate(tok.pad_id)), device)))

    positives = sum(r["label"] for r in rows)
    print(f"\n{args.cache}/{args.split}: {len(rows):,} examples "
          f"({positives:,} turn-final / {len(rows) - positives:,} mid-turn)\n")
    print(f"{'':24s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'ap':>7s}")
    for name, m in results:
        print(f"{name:24s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f} {m['ap']:7.3f}")

    best_single = max(results[2], results[3], key=lambda r: r[1]["ap"])
    if args.fusion_ckpt:
        gain = results[-1][1]["ap"] - best_single[1]["ap"]
        verdict = ("fusion helps" if gain > 0 else
                   "fusion does NOT beat the best single branch")
        print(f"\n{verdict}: AP {results[-1][1]['ap']:.3f} vs "
              f"{best_single[1]['ap']:.3f} ({best_single[0]}), delta {gain:+.3f}")

    if args.out:
        args.out.write_text(json.dumps(
            {"split": args.split, "examples": len(rows),
             "results": {name: m for name, m in results}}, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
