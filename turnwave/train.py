"""Train an EOT branch: text, audio, or the fused model.

Text (Phase 1):
    python -m turnwave.train --task text \
        --train data/text/train.jsonl --val data/text/validation.jsonl \
        --tokenizer checkpoints/tokenizer/spm.model --out checkpoints/text_eot

Audio (Phase 2):
    python -m turnwave.train --task audio \
        --cache data/audio --out checkpoints/audio_eot

One loop serves every task because all three collates return
`(inputs_tuple, labels)` and every model is called as `model(*inputs)`.

Add --steps 300 --limit 20000 for a CPU smoke run; real training is a Colab T4
session (see notebooks/colab_train.ipynb).
"""

import argparse
import contextlib
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.audio_loader import (
    EOTAudioDataset,
    audio_collate,
    make_fusion_collate,
    positive_weight,
)
from .data.loader import EOTTextDataset, make_collate
from .metrics import average_precision, binary_metrics
from .models.audio_cnn import AudioEOTConfig, AudioEOTModel
from .models.fusion import FusionConfig, build_fusion
from .models.text_transformer import TextEOTConfig, TextEOTModel
from .tokenizer import Tokenizer


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Only parameters with requires_grad are optimized, so a fusion model with
    frozen branches trains just its head."""
    # decay matrices only; norms, biases, and the head bias stay undecayed
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None,
             pos_weight: torch.Tensor | None = None) -> dict:
    """Works for any task: loaders yield (inputs_tuple, labels)."""
    model.eval()
    losses, probs, labels = [], [], []
    for i, (inputs, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = tuple(t.to(device) for t in inputs)
        y = y.to(device)
        logits = model(*inputs)
        losses.append(F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight).item())
        probs.extend(torch.sigmoid(logits).tolist())
        labels.extend(y.tolist())
    model.train()
    out = binary_metrics(probs, labels)
    out["loss"] = sum(losses) / len(losses)
    out["ap"] = average_precision(probs, labels)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["text", "audio", "fusion"], default="text")
    ap.add_argument("--train", type=Path, help="text task: train jsonl")
    ap.add_argument("--val", type=Path, help="text task: validation jsonl")
    ap.add_argument("--tokenizer", type=Path, help="text task: sentencepiece model")
    ap.add_argument("--cache", type=Path, help="audio/fusion task: feature cache directory")
    ap.add_argument("--text-ckpt", type=Path, help="fusion task: trained text branch")
    ap.add_argument("--audio-ckpt", type=Path, help="fusion task: trained audio branch")
    ap.add_argument("--unfreeze", action="store_true",
                    help="fusion task: fine-tune the branches too (use a low --lr)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=40, help="cap val batches per eval; 0 = full val set")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="subsample training rows (smoke tests)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          args.device if args.device != "auto" else "cpu")

    if args.task == "text":
        if not (args.train and args.val and args.tokenizer):
            ap.error("--task text needs --train, --val and --tokenizer")
        tok = Tokenizer(args.tokenizer)
        train_ds = EOTTextDataset(args.train, tok, args.max_len, limit=args.limit)
        val_ds = EOTTextDataset(args.val, tok, args.max_len)
        collate = make_collate(tok.pad_id)
        cfg = TextEOTConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                            n_layers=args.n_layers, n_heads=args.n_heads,
                            max_seq_len=args.max_len, dropout=args.dropout)
        model = TextEOTModel(cfg).to(device)
    elif args.task == "audio":
        if not args.cache:
            ap.error("--task audio needs --cache")
        manifest = json.loads((args.cache / "manifest.json").read_text())
        n_mels, n_frames = manifest["shape"]
        train_ds = EOTAudioDataset(args.cache, "train", limit=args.limit)
        val_ds = EOTAudioDataset(args.cache, "validation")
        collate = audio_collate
        cfg = AudioEOTConfig(n_mels=n_mels, n_frames=n_frames, dropout=args.dropout)
        model = AudioEOTModel(cfg).to(device)
    else:
        if not (args.cache and args.tokenizer and args.text_ckpt and args.audio_ckpt):
            ap.error("--task fusion needs --cache, --tokenizer, --text-ckpt and --audio-ckpt")
        tok = Tokenizer(args.tokenizer)
        train_ds = EOTAudioDataset(args.cache, "train", tokenizer=tok,
                                   max_len=args.max_len, limit=args.limit)
        val_ds = EOTAudioDataset(args.cache, "validation", tokenizer=tok, max_len=args.max_len)
        collate = make_fusion_collate(tok.pad_id)
        cfg = FusionConfig(dropout=args.dropout, freeze_branches=not args.unfreeze)
        model = build_fusion(args.text_ckpt, args.audio_ckpt, cfg, device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate,
                            num_workers=args.num_workers)

    # Rebalance the loss if the cache is skewed, so the model cannot win by
    # guessing the majority class.
    pos_weight = None
    if args.task in ("audio", "fusion"):
        weight = positive_weight(train_ds)
        if abs(weight - 1.0) > 0.05:
            pos_weight = torch.tensor(weight, device=device)

    trainable = getattr(model, "num_trainable_params", model.num_params)
    print(f"device={device} task={args.task} params={model.num_params/1e6:.2f}M "
          f"(trainable {trainable/1e6:.2f}M) train={len(train_ds)} val={len(val_ds)}"
          + (f" pos_weight={pos_weight.item():.2f}" if pos_weight is not None else ""))

    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else contextlib.nullcontext())

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "lr", "train_loss", "val_loss", "val_acc", "val_f1", "val_ap"])

    def save(name: str, step: int, val: dict):
        payload = {"model": model.state_dict(), "config": asdict(cfg),
                   "task": args.task, "step": step, "val": val}
        if args.task == "fusion":
            # FusionConfig alone cannot rebuild the model: the branch shapes live
            # in their own configs, and the weights in this state_dict are useless
            # without them.
            payload["text_config"] = asdict(model.text.cfg)
            payload["audio_config"] = asdict(model.audio.cfg)
        torch.save(payload, args.out / name)

    best_ap = -1.0
    step = 0
    running_loss = []
    model.train()
    while step < args.steps:
        for inputs, y in train_loader:
            if step >= args.steps:
                break
            lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            inputs = tuple(t.to(device) for t in inputs)
            y = y.to(device)
            with autocast:
                loss = F.binary_cross_entropy_with_logits(model(*inputs), y,
                                                          pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            running_loss.append(loss.item())
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                val = evaluate(model, val_loader, device,
                               max_batches=args.val_batches or None,
                               pos_weight=pos_weight)
                train_loss = sum(running_loss) / len(running_loss)
                running_loss = []
                print(f"step {step:6d}  lr {lr:.2e}  train {train_loss:.4f}  "
                      f"val {val['loss']:.4f}  acc {val['accuracy']:.3f}  "
                      f"f1 {val['f1']:.3f}  ap {val['ap']:.3f}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{lr:.6g}", f"{train_loss:.5f}", f"{val['loss']:.5f}",
                                            f"{val['accuracy']:.4f}", f"{val['f1']:.4f}", f"{val['ap']:.4f}"])
                save("last.pt", step, val)
                if val["ap"] > best_ap:
                    best_ap = val["ap"]
                    save("best.pt", step, val)

    print(f"done. best val AP {best_ap:.3f}  checkpoints in {args.out}")


if __name__ == "__main__":
    main()
