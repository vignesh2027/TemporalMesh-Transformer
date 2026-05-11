"""
embedding.py — TokenEmbedding and TemporalPositionEncoder.

Novel vs standard: RoPE positional encoding is extended with per-token learned
decay scalars so that semantically distant tokens are attenuated before they
reach the attention layer — no recurrence needed.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .config import TMTConfig


class TokenEmbedding(nn.Module):
    """Standard learned token embedding with output projection scale."""

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.scale = math.sqrt(cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (B, S) → (B, S, D)
        return self.embed(token_ids) * self.scale

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"TokenEmbedding(params={p:,})"


class TemporalPositionEncoder(nn.Module):
    """
    RoPE base + learned temporal decay scalars per position.

    Decay scalar: sigmoid(w_decay · t) where t is the absolute position index
    normalised to [0, 1] over max_seq_len.  The scalar multiplies the embedding
    before it reaches MeshAttention so semantically distant tokens fade.
    """

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        self.decay_rate = cfg.decay_rate

        # Learned decay weights — one per position dimension pair
        self.w_decay = nn.Parameter(
            torch.full((cfg.d_model,), cfg.decay_rate)
        )

        # RoPE cos/sin cache (not a parameter — regenerated on device change)
        self._build_rope_cache(cfg.max_seq_len, cfg.d_model)

    def _build_rope_cache(self, max_len: int, d: int) -> None:
        half = d // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(pos, theta)  # (max_len, half)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, d)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
        cos = self.rope_cos[:seq_len].unsqueeze(0)  # (1, S, D)
        sin = self.rope_sin[:seq_len].unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, S, D) token embeddings
        Returns:
            encoded: (B, S, D) with RoPE applied
            decay_scalars: (B, S, D) sigmoid decay weights per token-dim
        """
        B, S, D = x.shape

        # RoPE
        x = self.apply_rope(x, S)

        # Temporal decay: t ∈ [0, 1] normalised position
        t = torch.arange(S, device=x.device, dtype=x.dtype) / max(S - 1, 1)
        # w_decay broadcast: (S, D) → decay per token dimension
        decay_scalars = torch.sigmoid(
            -rearrange(t, "s -> s 1") * rearrange(self.w_decay, "d -> 1 d")
        )  # (S, D)
        decay_scalars = decay_scalars.unsqueeze(0).expand(B, -1, -1)  # (B, S, D)

        return x * decay_scalars, decay_scalars

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"TemporalPositionEncoder(d={self.d_model}, params={p:,})"
