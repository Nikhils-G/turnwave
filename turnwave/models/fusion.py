"""The fused model: prosody and semantics deciding together.

Each branch sees half the evidence. The text transformer cannot tell "yes i
agree" (truncated) from "yes i agree" (finished) — lexically they are identical.
The CNN hears the difference in delivery but has no idea what was said, so it
cannot use the fact that "i want to order a" is syntactically incomplete. Fusion
is where a turn detector should stop being either.

Branches are loaded from their standalone checkpoints and frozen by default, so
the ablation is honest: the fused model demonstrably starts from the two trained
branches rather than being a third model that happens to see both inputs.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .audio_cnn import AudioEOTConfig, AudioEOTModel
from .text_transformer import TextEOTConfig, TextEOTModel


@dataclass
class FusionConfig:
    hidden_dim: int = 256
    dropout: float = 0.2
    freeze_branches: bool = True


class FusionEOTModel(nn.Module):
    def __init__(self, text: TextEOTModel, audio: AudioEOTModel, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.text = text
        self.audio = audio
        if cfg.freeze_branches:
            self.freeze_branches()

        in_dim = text.cfg.d_model + audio.cfg.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def freeze_branches(self) -> None:
        for p in self.text.parameters():
            p.requires_grad = False
        for p in self.audio.parameters():
            p.requires_grad = False

    def unfreeze_branches(self) -> None:
        """For an optional second pass at a low learning rate."""
        for p in self.text.parameters():
            p.requires_grad = True
        for p in self.audio.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep frozen branches in eval mode.

        Without this, BatchNorm in the CNN would keep updating its running
        statistics and dropout would keep firing inside branches that are not
        being trained — the fused model would drift away from the branches the
        ablation claims it started from.
        """
        super().train(mode)
        if self.cfg.freeze_branches:
            self.text.eval()
            self.audio.eval()
        return self

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor,
                mel: torch.Tensor) -> torch.Tensor:
        if self.cfg.freeze_branches:
            with torch.no_grad():
                text_embed = self.text.embed(idx, lengths)
                audio_embed = self.audio.embed(mel)
        else:
            text_embed = self.text.embed(idx, lengths)
            audio_embed = self.audio.embed(mel)
        return self.head(torch.cat([text_embed, audio_embed], dim=-1)).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_branch(ckpt_path: str | Path, kind: str, device: torch.device):
    """Rebuild a branch from a training checkpoint written by turnwave.train."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if kind == "text":
        model = TextEOTModel(TextEOTConfig(**ckpt["config"]))
    elif kind == "audio":
        model = AudioEOTModel(AudioEOTConfig(**ckpt["config"]))
    else:
        raise ValueError(f"unknown branch kind: {kind}")
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def build_fusion(text_ckpt: str | Path, audio_ckpt: str | Path, cfg: FusionConfig,
                 device: torch.device) -> FusionEOTModel:
    text = load_branch(text_ckpt, "text", device)
    audio = load_branch(audio_ckpt, "audio", device)
    return FusionEOTModel(text, audio, cfg).to(device)


def load_fusion(ckpt_path: str | Path, device: torch.device) -> FusionEOTModel:
    """Rebuild a trained fusion model from a single checkpoint.

    Needs the branch configs the trainer stores alongside FusionConfig; without
    them the state_dict cannot be given a shape to load into.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "text_config" not in ckpt or "audio_config" not in ckpt:
        raise ValueError(f"{ckpt_path} is not a fusion checkpoint (no branch configs)")
    text = TextEOTModel(TextEOTConfig(**ckpt["text_config"]))
    audio = AudioEOTModel(AudioEOTConfig(**ckpt["audio_config"]))
    model = FusionEOTModel(text, audio, FusionConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
