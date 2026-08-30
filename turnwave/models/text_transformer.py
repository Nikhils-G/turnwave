"""A small causal transformer for end-of-utterance detection, written from scratch.

Modern decoder-style architecture: RMSNorm (pre-norm), rotary position embeddings
(RoPE), SwiGLU feed-forward, causal multi-head attention. Attention is computed
explicitly (matmul -> mask -> softmax -> matmul) rather than via a fused kernel —
at <10M params, clarity beats the speedup.

Classification: the hidden state of the LAST real token feeds a linear head that
outputs one logit, P(turn complete). With right-padding and a causal mask, real
tokens can never attend to pads (pads only appear at future positions), so no
separate padding mask is needed — we just gather at index lengths-1.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextEOTConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, Dh]; cos/sin: [T, Dh/2]. Rotates the two halves of each head dim."""
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TextEOTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim % 2 == 0, "RoPE needs an even head dim"
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # RoPE tables and causal mask, precomputed to max_seq_len
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.arange(cfg.max_seq_len).float()[:, None] * inv_freq[None, :]  # [T, Dh/2]
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        causal = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = att @ v  # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.proj(out))


class SwiGLU(nn.Module):
    def __init__(self, cfg: TextEOTConfig):
        super().__init__()
        # 2/3 of the usual 4x expansion keeps parameter count comparable to a GELU MLP
        hidden = int(8 * cfg.d_model / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: TextEOTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TextEOTModel(nn.Module):
    def __init__(self, cfg: TextEOTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_out = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init_weights)
        # scale residual projections down by depth (GPT-2-style) for stable training
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def backbone(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [B, T] token ids -> [B, T, D] normalized hidden states."""
        x = self.emb_dropout(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)

    def embed(self, idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, d_model]: the hidden state at the last real token.

        Right padding plus the causal mask means pads can never influence real
        tokens, so this is a clean summary of the sequence. Shared with the fusion
        model, which needs the representation rather than the logit.
        """
        hidden = self.backbone(idx)
        return hidden[torch.arange(idx.size(0), device=idx.device), lengths - 1]

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Returns one logit per sequence: P(turn complete) after sigmoid."""
        return self.head(self.embed(idx, lengths)).squeeze(-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
