"""Build the text EOT dataset from DailyDialog.

Writes data/text/{train,validation,test}.jsonl plus a tokenizer corpus of the
complete training utterances. Splits follow DailyDialog's own dialogue-level
splits, so no utterance leaks across train/val/test.

    python scripts/build_text_dataset.py --out data/text
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from turnwave.data.text_pairs import iter_examples

# per-split seed offsets keep negative sampling deterministic and independent
SPLIT_SEEDS = {"train": 0, "validation": 1, "test": 2}


def load_dailydialog(hf_name: str):
    # The repo's own loading script is unsupported by datasets>=4, so we read
    # HF's auto-converted parquet branch; plain load is the fallback for repos
    # that already ship parquet.
    try:
        return load_dataset(hf_name, revision="refs/convert/parquet")
    except Exception as first_error:
        try:
            return load_dataset(hf_name)
        except Exception:
            raise SystemExit(
                f"Could not load DailyDialog from '{hf_name}'.\nOriginal error: {first_error}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/text"))
    ap.add_argument("--hf-name", default="li2017dailydialog/daily_dialog")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negatives", type=int, default=1, help="truncation negatives per utterance")
    args = ap.parse_args()

    ds = load_dailydialog(args.hf_name)
    args.out.mkdir(parents=True, exist_ok=True)

    corpus_lines: list[str] = []
    for split, seed_offset in SPLIT_SEEDS.items():
        dialogues = [example["dialog"] for example in ds[split]]
        rows = list(iter_examples(dialogues, seed=args.seed + seed_offset,
                                  negatives_per_positive=args.negatives))
        path = args.out / f"{split}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        n_pos = sum(row["label"] for row in rows)
        print(f"{split:12s} {len(dialogues):6d} dialogues -> {len(rows):7d} examples "
              f"({n_pos} complete / {len(rows) - n_pos} truncated)")
        if split == "train":
            corpus_lines = [row["text"] for row in rows if row["label"] == 1]

    corpus_path = args.out / "corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines) + "\n")
    print(f"tokenizer corpus: {corpus_path} ({len(corpus_lines)} utterances)")


if __name__ == "__main__":
    main()
