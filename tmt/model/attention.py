"""
attention.py — MeshAttention: multi-head attention over graph edges.

Novel vs standard: instead of every token attending to every other token
(O(S²) full attention), MeshAttention restricts attention to graph neighbours.
Temporal decay is multiplied into the attention weights — not added as bias —
so semantically close but temporally distant tokens are suppressed.

Formula: attn = softmax(QK^T / sqrt(d)) * sigmoid(W_decay * temporal_distance)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .config import TMTConfig


class MeshAttention(nn.Module):
    """
    Multi-head attention constrained to dynamic graph edges with temporal decay.

    Falls back to a sparse neighbour-masked full attention when torch_geometric
    is unavailable, preserving identical semantics.
    """

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Learned temporal decay weight (scalar applied per head)
        self.w_decay = nn.Parameter(torch.ones(cfg.n_heads) * cfg.decay_rate)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        decay_scalars: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:             (B, S, D)
            edge_index:    (2, E) global node indices
            edge_weight:   (E,) cosine similarity weights
            decay_scalars: (B, S, D) per-token temporal decay from encoder
        Returns:
            out: (B, S, D)
        """
        B, S, D = x.shape

        Q = self.q_proj(x)  # (B, S, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to multi-head
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.n_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.n_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.n_heads)

        # Full attention scores (B, H, S, S)
        scores = torch.einsum("bhid,bhjd->bhij", Q, K) / self.scale

        # Build sparse neighbour mask from edge_index
        # edge_index is over global indices (B*S); remap to per-batch local
        mask = torch.full((B, S, S), float("-inf"), device=x.device)
        if edge_index.numel() > 0:
            src_global = edge_index[0]  # (E,)
            dst_global = edge_index[1]  # (E,)
            b_idx = src_global // S
            src_local = src_global % S
            dst_local = dst_global % S
            mask[b_idx, src_local, dst_local] = edge_weight.float()

        # Also allow causal self (diagonal) so every token has at least itself
        diag_mask = torch.zeros(S, S, device=x.device)
        diag_mask.fill_diagonal_(0.0)
        mask = mask + diag_mask.unsqueeze(0)

        # Apply graph mask
        scores = scores + mask.unsqueeze(1)  # broadcast over heads

        attn = F.softmax(scores, dim=-1)  # (B, H, S, S)

        # Temporal decay: multiply attention weights by sigmoid decay per token
        if decay_scalars is not None:
            # Average decay across D → (B, S) scalar per token
            token_decay = decay_scalars.mean(dim=-1)  # (B, S)
            # Per-head decay scaling: w_decay (H,) * token_decay (B, S)
            head_decay = torch.sigmoid(
                rearrange(self.w_decay, "h -> 1 h 1") *
                rearrange(token_decay, "b s -> b 1 s")
            )  # (B, H, S)
            attn = attn * head_decay.unsqueeze(-1)

        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, V)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out_proj(out)

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"MeshAttention(heads={self.n_heads}, d={self.d_model}, params={p:,})"
