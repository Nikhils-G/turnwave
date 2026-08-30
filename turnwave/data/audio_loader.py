"""Datasets over the precomputed log-mel cache.

Three tasks share one collate contract: every collate returns
`(inputs_tuple, labels)` and every model is called as `model(*inputs)`. Text
yields `((idx, lengths), y)`, audio `((mel,), y)`, fusion `((idx, lengths, mel), y)`
— so `train.py` stays a single loop instead of growing a task abstraction.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EOTAudioDataset(Dataset):
    """Reads the memmapped feature cache written by scripts/build_audio_dataset.py.

    Set `tokenizer` to also emit token ids for fusion training.
    """

    def __init__(self, cache_dir: str | Path, split: str, tokenizer=None,
                 max_len: int = 128, limit: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(self.cache_dir / f"{split}.jsonl") as f:
            rows = [json.loads(line) for line in f]
        self.rows = rows[:limit] if limit else rows
        self._features: np.ndarray | None = None  # opened lazily, see below

    @property
    def features(self) -> np.ndarray:
        # Opened on first access rather than in __init__ so each DataLoader worker
        # gets its own handle instead of inheriting one across the fork.
        if self._features is None:
            self._features = np.load(self.cache_dir / f"{self.split}.f16.npy", mmap_mode="r")
        return self._features

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        item = {
            "mel": torch.from_numpy(np.asarray(self.features[row["i"]], dtype=np.float32)),
            "label": float(row["label"]),
        }
        if self.tokenizer is not None:
            item["ids"] = self.tokenizer.encode_example("", row["text"], self.max_len)
        return item


def _stack_ids(batch: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(b["ids"]) for b in batch)
    idx = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        idx[i, : len(b["ids"])] = torch.tensor(b["ids"], dtype=torch.long)
    lengths = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
    return idx, lengths


def _labels(batch: list[dict]) -> torch.Tensor:
    return torch.tensor([b["label"] for b in batch], dtype=torch.float)


def audio_collate(batch: list[dict]):
    return (torch.stack([b["mel"] for b in batch]),), _labels(batch)


def make_fusion_collate(pad_id: int):
    def collate(batch: list[dict]):
        idx, lengths = _stack_ids(batch, pad_id)
        mel = torch.stack([b["mel"] for b in batch])
        return (idx, lengths, mel), _labels(batch)

    return collate


def positive_weight(dataset: EOTAudioDataset) -> float:
    """pos_weight for BCE, so a skewed cache does not bias the model toward the
    majority class. Returns 1.0 for a balanced split."""
    positives = sum(r["label"] for r in dataset.rows)
    negatives = len(dataset.rows) - positives
    return (negatives / positives) if positives else 1.0
