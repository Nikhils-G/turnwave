"""Torch Dataset + collate for the text EOT task."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EOTTextDataset(Dataset):
    """Reads a JSONL of {context, text, label}. `tokenizer` is any object with
    encode_example(context, text, max_len) and a pad_id attribute."""

    def __init__(self, path: str | Path, tokenizer, max_len: int = 128, limit: int | None = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        self.rows = rows[:limit] if limit else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        ids = self.tokenizer.encode_example(row.get("context", ""), row["text"], self.max_len)
        return {"ids": ids, "label": float(row["label"])}


def make_collate(pad_id: int):
    def collate(batch: list[dict]):
        max_len = max(len(b["ids"]) for b in batch)
        idx = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            idx[i, : len(b["ids"])] = torch.tensor(b["ids"], dtype=torch.long)
        lengths = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        return idx, lengths, labels

    return collate
