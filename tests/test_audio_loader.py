"""Loader tests, including the collate contract shared by all three tasks."""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from turnwave.data.audio_loader import (
    EOTAudioDataset,
    audio_collate,
    make_fusion_collate,
    positive_weight,
)
from turnwave.data.loader import EOTTextDataset, make_collate
from turnwave.models.audio_cnn import AudioEOTConfig, AudioEOTModel
from turnwave.train import configure_optimizer

N_MELS, N_FRAMES = 16, 32


class FakeTokenizer:
    pad_id = 0

    def encode_example(self, context, text, max_len=128):
        return [1] + [ord(c) % 20 + 2 for c in text[:8]]


def make_cache(tmp_path, n=24, separable=False, positive_rate=0.5):
    """Writes a cache in the on-disk format the builder produces."""
    rng = np.random.default_rng(0)
    feats = np.lib.format.open_memmap(
        tmp_path / "train.f16.npy", mode="w+", dtype=np.float16, shape=(n, N_MELS, N_FRAMES))
    with open(tmp_path / "train.jsonl", "w") as f:
        for i in range(n):
            label = int(i < round(n * positive_rate))
            x = rng.normal(size=(N_MELS, N_FRAMES))
            if separable:
                # positives get energy in the low bins: a signal a CNN can find
                x[:4] += 4.0 if label else -4.0
            feats[i] = x.astype(np.float16)
            f.write(json.dumps({"i": i, "text": "hello there", "label": label}) + "\n")
    feats.flush()
    (tmp_path / "manifest.json").write_text(json.dumps({"shape": [N_MELS, N_FRAMES]}))
    return tmp_path


def test_dataset_reads_cache(tmp_path):
    ds = EOTAudioDataset(make_cache(tmp_path), "train")
    assert len(ds) == 24
    item = ds[0]
    assert item["mel"].shape == (N_MELS, N_FRAMES)
    assert item["mel"].dtype == torch.float32  # cached as f16, served as f32
    assert item["label"] in (0.0, 1.0)


def test_dataset_limit(tmp_path):
    assert len(EOTAudioDataset(make_cache(tmp_path), "train", limit=5)) == 5


def test_audio_collate_contract(tmp_path):
    """Every task's collate returns (inputs_tuple, labels)."""
    ds = EOTAudioDataset(make_cache(tmp_path), "train")
    inputs, labels = audio_collate([ds[0], ds[1], ds[2]])
    assert isinstance(inputs, tuple) and len(inputs) == 1
    assert inputs[0].shape == (3, N_MELS, N_FRAMES)
    assert labels.shape == (3,)


def test_fusion_collate_contract(tmp_path):
    ds = EOTAudioDataset(make_cache(tmp_path), "train", tokenizer=FakeTokenizer())
    (idx, lengths, mel), labels = make_fusion_collate(0)([ds[0], ds[1]])
    assert idx.shape[0] == 2 and lengths.shape == (2,)
    assert mel.shape == (2, N_MELS, N_FRAMES)
    assert labels.shape == (2,)


def test_text_collate_uses_the_same_contract(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text('{"context": "", "text": "hi there", "label": 1}\n' * 3)
    ds = EOTTextDataset(path, FakeTokenizer())
    inputs, labels = make_collate(0)([ds[0], ds[1]])
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert labels.shape == (2,)


def test_positive_weight_balances_a_skewed_cache(tmp_path):
    ds = EOTAudioDataset(make_cache(tmp_path, n=20, positive_rate=0.25), "train")
    assert positive_weight(ds) == pytest.approx(15 / 5)  # negatives / positives


def test_positive_weight_is_one_when_balanced(tmp_path):
    ds = EOTAudioDataset(make_cache(tmp_path, n=20, positive_rate=0.5), "train")
    assert positive_weight(ds) == pytest.approx(1.0)


def test_audio_model_learns_a_separable_signal(tmp_path):
    """CPU smoke test: the loss must fall on a task the CNN can actually solve."""
    torch.manual_seed(0)
    ds = EOTAudioDataset(make_cache(tmp_path, n=64, separable=True), "train")
    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True, collate_fn=audio_collate)
    model = AudioEOTModel(AudioEOTConfig(
        n_mels=N_MELS, n_frames=N_FRAMES, channels=(8, 16), embed_dim=16,
        dropout=0.0, freq_mask=0, time_mask=0)).train()
    optimizer = configure_optimizer(model, lr=3e-3, weight_decay=0.0)

    losses = []
    for _ in range(12):
        for inputs, y in loader:
            loss = F.binary_cross_entropy_with_logits(model(*inputs), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    first, last = sum(losses[:4]) / 4, sum(losses[-4:]) / 4
    assert last < first * 0.6, f"loss did not fall: {first:.3f} -> {last:.3f}"
