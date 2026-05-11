"""
ffn.py — DualStreamFFN: parallel syntax + semantic feed-forward network.

Novel vs standard: instead of a single FFN (d_model → 4*d_model → d_model),
DualStreamFFN runs two parallel streams of half-width (d_model → d_stream),
each specialising on syntax or semantic content, then fuses them with a learned
gate.  This gives the same parameter budget as a standard FFN while separating
representational concerns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .config import TMTConfig


class DualStreamFFN(nn.Module):
    """Two parallel feed-forward streams fused by a learned scalar gate."""

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        d = cfg.d_model
        s = cfg.ffn_stream_dim  # each stream width (default 256)

        # Syntax stream
        self.syntax_up = nn.Linear(d, s)
        self.syntax_down = nn.Linear(s, d)

        # Semantic stream
        self.semantic_up = nn.Linear(d, s)
        self.semantic_down = nn.Linear(s, d)

        # Learned fusion gate: sigmoid(linear) → scalar ∈ (0,1) per token-dim
        self.gate = nn.Linear(d, d)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, D)
        Returns:
            out: (B, S, D)
        """
        syntax = self.dropout(self.syntax_down(self.act(self.syntax_up(x))))
        semantic = self.dropout(self.semantic_down(self.act(self.semantic_up(x))))

        # Learned fusion gate
        g = torch.sigmoid(self.gate(x))  # (B, S, D)
        return g * syntax + (1.0 - g) * semantic

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"DualStreamFFN(streams=2x{self.syntax_up.out_features}, params={p:,})"
