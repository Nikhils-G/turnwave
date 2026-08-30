"""Prosody branch: a CNN over log-mel spectrograms, trained from scratch.

This is the half of TurnWave that hears what the transcript cannot. Phase 1's
text model is capped by truncations that are accidentally complete phrases —
"yes i agree completely" cut to "yes i agree" is lexically a finished turn. What
separates them is delivery: a speaker who intends to continue holds pitch level
and trails energy into the pause; a finished speaker drops pitch and closes.

Deliberately random-initialised. Pipecat's smart-turn uses a pretrained
Whisper-tiny encoder and will very likely score better for it; the trade is that
every layer here is one the author chose and can defend, and the benchmark
reports the gap rather than hiding it.

SpecAugment is applied inside the model so it follows train()/eval() automatically
— a masking bug that leaked into evaluation would quietly corrupt every number the
project reports.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AudioEOTConfig:
    n_mels: int = 64
    n_frames: int = 201
    channels: tuple[int, ...] = (64, 128, 256, 384)
    embed_dim: int = 256
    dropout: float = 0.2
    # SpecAugment: max width of each mask, in bins/frames. Two masks per axis.
    freq_mask: int = 8
    time_mask: int = 20
    n_masks: int = 2


class SpecAugment(nn.Module):
    """Zeroes random frequency bands and time spans during training only.

    The standard regulariser for spectrogram models, and the direct answer to the
    overfitting Phase 1 measured: masking forces the model to spread its evidence
    over the whole window instead of memorising one cue.
    """

    def __init__(self, cfg: AudioEOTConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, n_mels, n_frames = x.shape
        x = x.clone()
        for size, axis_len, axis in ((self.cfg.freq_mask, n_mels, 1),
                                     (self.cfg.time_mask, n_frames, 2)):
            if size <= 0:
                continue
            for _ in range(self.cfg.n_masks):
                width = torch.randint(0, size + 1, (B,), device=x.device)
                start = (torch.rand(B, device=x.device) *
                         (axis_len - width).clamp(min=1).float()).long()
                ramp = torch.arange(axis_len, device=x.device)
                mask = (ramp >= start[:, None]) & (ramp < (start + width)[:, None])
                x = x.masked_fill(mask[:, :, None] if axis == 1 else mask[:, None, :], 0.0)
        return x


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, then halve both axes.

    Two convs rather than one: a single 3x3 per resolution gives a receptive field
    too small to see a pitch contour, and stacking them is cheaper than widening
    to the same capacity.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return self.pool(x)


class AudioEOTModel(nn.Module):
    def __init__(self, cfg: AudioEOTConfig):
        super().__init__()
        self.cfg = cfg
        self.augment = SpecAugment(cfg)
        blocks, in_ch = [], 1
        for out_ch in cfg.channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Linear(cfg.channels[-1], cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.embed_dim, 1)

    def embed(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, n_mels, n_frames] -> [B, embed_dim]. Shared with the fusion model."""
        x = self.augment(mel)
        # Per-example standardization: absolute loudness varies with the recording,
        # the prosodic contour is what carries the turn-taking signal.
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        x = self.blocks(x.unsqueeze(1))  # [B, C, mels', frames']
        x = x.mean(dim=(2, 3))  # global average pool
        return self.dropout(self.act(self.norm(self.project(x))))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(mel)).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
