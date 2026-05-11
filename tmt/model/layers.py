"""
layers.py — TMTLayer: one full layer of the TemporalMesh Transformer.

Combines MeshAttention → DualStreamFFN → ExitGate → MemoryAnchorCross.
Tokens that have already exited (exit_mask=True) are frozen — their
representation from the exiting layer is carried forward unchanged.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import MeshAttention
from .config import TMTConfig
from .exit_gate import ExitGate
from .ffn import DualStreamFFN
from .memory import MemoryAnchorCross


class TMTLayer(nn.Module):
    def __init__(self, cfg: TMTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = MeshAttention(cfg)

        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ffn = DualStreamFFN(cfg)

        self.exit_gate = ExitGate(cfg)

        self.norm3 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.memory_cross = MemoryAnchorCross(cfg)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        exit_mask: Tensor,
        decay_scalars: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x:             (B, S, D)
            edge_index:    (2, E)
            edge_weight:   (E,)
            exit_mask:     (B, S) bool — True where token has exited
            decay_scalars: (B, S, D) optional temporal decay
        Returns:
            x:              (B, S, D) updated representations
            exit_mask:      (B, S) updated exit mask
            confidence:     (B, S) gate confidence scores
            memory_state:   (M, D) updated memory anchors
        """
        # Save exited-token representations so we can restore after layer ops
        x_frozen = x.clone()

        # MeshAttention + residual
        attn_out = self.attn(self.norm1(x), edge_index, edge_weight, decay_scalars)
        x = x + attn_out

        # DualStreamFFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        # ExitGate
        x, exit_mask, confidence = self.exit_gate(x, exit_mask)

        # Memory cross-attention + residual
        mem_out, memory_state = self.memory_cross(self.norm3(x))
        x = x + mem_out

        # Freeze exited tokens: replace with their pre-layer values
        if exit_mask.any():
            frozen = exit_mask.unsqueeze(-1).expand_as(x)
            x = torch.where(frozen, x_frozen, x)

        return x, exit_mask, confidence, memory_state

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"TMTLayer(idx={self.layer_idx}, params={p:,})"
