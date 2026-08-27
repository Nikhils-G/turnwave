"""Evaluate a trained text EOT checkpoint against honest baselines.

Baselines matter: our negatives are generated truncations, so a model must beat
(a) the majority class and (b) a hand-written cue-word heuristic before the
learned approach has earned anything.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.loader import EOTTextDataset, make_collate
from .metrics import average_precision, binary_metrics
from .models.text_transformer import TextEOTConfig, TextEOTModel
from .tokenizer import Tokenizer
from .train import evaluate

# Words that rarely end a complete utterance: conjunctions, prepositions,
# articles, pronouns-in-flight, auxiliaries, fillers.
INCOMPLETE_LAST_WORDS = {
    "and", "or", "but", "so", "because", "if", "when", "while", "that", "which",
    "the", "a", "an", "to", "of", "in", "on", "at", "with", "for", "from", "by",
    "about", "into", "than", "as", "my", "your", "his", "her", "our", "their",
    "i", "i'm", "i'll", "i'd", "we", "he", "she", "they", "it's", "is", "are",
    "was", "were", "am", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "very", "really", "quite", "um",
    "uh", "like", "just", "not", "no", "how", "what", "where", "who", "why",
}


def heuristic_baseline(rows: list[dict]) -> dict:
    probs = [0.0 if row["text"].split()[-1] in INCOMPLETE_LAST_WORDS else 1.0 for row in rows]
    labels = [float(row["label"]) for row in rows]
    out = binary_metrics(probs, labels)
    out["ap"] = average_precision(probs, labels)
    return out


def majority_baseline(rows: list[dict]) -> dict:
    labels = [float(row["label"]) for row in rows]
    majority = 1.0 if sum(labels) >= len(labels) / 2 else 0.0
    probs = [majority] * len(labels)
    out = binary_metrics(probs, labels)
    out["ap"] = average_precision(probs, labels)
    return out


def load_model(ckpt_path: Path, device: torch.device) -> TextEOTModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TextEOTModel(TextEOTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    tok = Tokenizer(args.tokenizer)
    model = load_model(args.ckpt, device)

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    ds = EOTTextDataset(args.data, tok)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=make_collate(tok.pad_id))
    model_metrics = evaluate(model, loader, device)

    print(f"\n{args.data} ({len(rows)} examples)\n")
    header = f"{'':24s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'ap':>7s}"
    print(header)
    for name, m in [("majority class", majority_baseline(rows)),
                    ("cue-word heuristic", heuristic_baseline(rows)),
                    ("turnwave text model", model_metrics)]:
        print(f"{name:24s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f} {m['ap']:7.3f}")


if __name__ == "__main__":
    main()
