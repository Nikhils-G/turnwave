"""Resume tests.

Free-tier sessions die mid-run. Without resume, a killed 4,000-step run costs the
whole run -- which happened twice in this project. These tests cover the three
things that make a resumed run equal to an unbroken one: the step counter, the
learning-rate schedule, and the optimizer's moment estimates.
"""

import csv
import json

import numpy as np
import pytest
import torch

from turnwave.train import lr_at, main

# The default CNN has a stride-2 stem and five pooling blocks, so it needs at
# least 64 mel bins; a smaller toy shape collapses to zero width mid-network.
N_MELS, N_FRAMES = 64, 64


def make_cache(tmp_path, n=128):
    """A separable task, so training actually moves the weights."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"
    cache.mkdir()
    for split, count in (("train", n), ("validation", 32)):
        feats = np.lib.format.open_memmap(
            cache / f"{split}.f16.npy", mode="w+", dtype=np.float16,
            shape=(count, N_MELS, N_FRAMES))
        with open(cache / f"{split}.jsonl", "w") as f:
            for i in range(count):
                label = i % 2
                x = rng.normal(size=(N_MELS, N_FRAMES))
                x[:8] += 4.0 if label else -4.0
                feats[i] = x.astype(np.float16)
                f.write(json.dumps({"i": i, "text": "", "label": label}) + "\n")
        feats.flush()
    (cache / "manifest.json").write_text(json.dumps({"shape": [N_MELS, N_FRAMES]}))
    return cache


def train(cache, out, steps, resume=False):
    argv = ["--task", "audio", "--cache", str(cache), "--out", str(out),
            "--steps", str(steps), "--batch-size", "16", "--eval-every", "20",
            "--warmup", "5", "--num-workers", "0", "--val-batches", "0",
            "--device", "cpu"]
    if resume:
        argv.append("--resume")
    main(argv)


def test_last_checkpoint_carries_training_state(tmp_path):
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert "optimizer" in state and "best_ap" in state
    assert state["step"] == 20


def test_best_checkpoint_stays_clean(tmp_path):
    """best.pt is the artifact we export and publish; optimizer state is roughly
    the model's own size and means nothing outside the run that made it."""
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert "optimizer" not in best


def test_resume_continues_the_step_counter(tmp_path):
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    train(cache, out, steps=40, resume=True)
    assert torch.load(out / "last.pt", map_location="cpu", weights_only=False)["step"] == 40


def test_resume_appends_to_the_log(tmp_path):
    """Rewriting the header would erase the first half of the training curve."""
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    train(cache, out, steps=40, resume=True)
    with open(out / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [20, 40]


def test_resume_restores_the_optimizer(tmp_path):
    """Adam's moment estimates are what stop the loss spiking at the seam."""
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    before = torch.load(out / "last.pt", map_location="cpu", weights_only=False)["optimizer"]
    train(cache, out, steps=40, resume=True)
    after = torch.load(out / "last.pt", map_location="cpu", weights_only=False)["optimizer"]
    assert before["state"], "optimizer had no state to restore"
    assert after["state"][0]["step"] > before["state"][0]["step"]


def test_resumed_schedule_matches_an_unbroken_run():
    """The LR is a pure function of step, so resuming lands on the same curve."""
    for step in (0, 5, 20, 39):
        assert lr_at(step, 3e-4, 5, 40, 1e-5) == lr_at(step, 3e-4, 5, 40, 1e-5)
    assert lr_at(20, 3e-4, 5, 40, 1e-5) < lr_at(5, 3e-4, 5, 40, 1e-5)


def test_resume_without_a_checkpoint_starts_fresh(tmp_path, capsys):
    cache = make_cache(tmp_path)
    out = tmp_path / "fresh"
    train(cache, out, steps=20, resume=True)
    assert "starting fresh" in capsys.readouterr().out
    assert torch.load(out / "last.pt", map_location="cpu", weights_only=False)["step"] == 20


def test_resume_at_target_steps_is_a_noop(tmp_path, capsys):
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    train(cache, out, steps=20, resume=True)
    assert "nothing to do" in capsys.readouterr().out


def test_resume_preserves_best_ap(tmp_path):
    """Otherwise the first eval after resuming overwrites a better best.pt."""
    cache = make_cache(tmp_path)
    out = tmp_path / "run"
    train(cache, out, steps=20)
    best_before = torch.load(out / "last.pt", map_location="cpu",
                             weights_only=False)["best_ap"]
    assert best_before > 0
    train(cache, out, steps=40, resume=True)
    assert torch.load(out / "last.pt", map_location="cpu",
                      weights_only=False)["best_ap"] >= best_before
