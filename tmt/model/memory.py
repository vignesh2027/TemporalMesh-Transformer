"""
memory.py — MemoryAnchorCross: persistent cross-attention to global memory nodes.

Novel vs standard: vanilla transformers have no persistent state across forward
passes.  MemoryAnchorCross maintains 16 learnable nn.Parameter vectors as
global "anchor" nodes that every token can attend to.  After each forward pass
the anchors are updated via exponential moving average (EMA) of the current
token representations, giving the model a form of fast-weight memory without
recurrence.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .config import TMTConfig


class MemoryAnchorCross(nn.Module):
    """Cross-attention from token stream to persistent memory anchor nodes."""

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.n_anchors = cfg.memory_anchors
        self.ema_alpha = 0.9  # EMA decay for memory update

        # Persistent memory parameters — shape (M, D)
        self.memory = nn.Parameter(
            torch.randn(cfg.memory_anchors, cfg.d_model) * 0.02
        )

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, S, D) token representations
        Returns:
            out:          (B, S, D) tokens enhanced by memory cross-attention
            memory_state: (M, D) updated memory anchors (detached for logging)
        """
        B, S, D = x.shape
        M = self.n_anchors
        scale = self.d_head ** -0.5

        # Queries come from tokens, Keys/Values from memory anchors
        Q = self.q_proj(x)  # (B, S, D)
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        K = self.k_proj(mem)  # (B, M, D)
        V = self.v_proj(mem)  # (B, M, D)

        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.n_heads)
        K = rearrange(K, "b m (h d) -> b h m d", h=self.n_heads)
        V = rearrange(V, "b m (h d) -> b h m d", h=self.n_heads)

        # Attention over memory anchors: (B, H, S, M)
        attn = torch.einsum("bhsd,bhmd->bhsm", Q, K) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhsm,bhmd->bhsd", attn, V)  # (B, H, S, D//H)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)

        # EMA update of memory anchors using mean token representation
        with torch.no_grad():
            token_mean = x.mean(dim=1).mean(dim=0)  # (D,) across batch
            self.memory.data = (
                self.ema_alpha * self.memory.data
                + (1 - self.ema_alpha) * token_mean.unsqueeze(0)
            )

        return out, self.memory.detach()

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"MemoryAnchorCross(anchors={self.n_anchors}, params={p:,})"
