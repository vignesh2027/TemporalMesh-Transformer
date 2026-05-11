"""
exit_gate.py — ExitGate: per-token adaptive depth routing.

Novel vs standard: every token in a standard transformer passes through all N
layers unconditionally.  ExitGate computes a confidence scalar after each layer
norm.  If confidence > exit_threshold the token's representation is frozen and
it skips all remaining layers, halving average compute on easy tokens.

The auxiliary training loss encourages the gate to be decisive (push toward 0
or 1) without forcing early exits — the model learns when it is confident.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import TMTConfig


class ExitGate(nn.Module):
    """Single linear → sigmoid confidence gate per token."""

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        self.threshold = cfg.exit_threshold
        # Single scalar projection: d_model → 1
        self.gate_proj = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # start pessimistic

    def forward(self, x: Tensor, exit_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x:         (B, S, D) current token representations
            exit_mask: (B, S) bool — True where token has already exited

        Returns:
            x:           (B, S, D) unchanged (gating is applied in TMTLayer)
            new_mask:    (B, S) bool — updated exit mask
            confidence:  (B, S) float confidence scores for auxiliary loss
        """
        confidence = torch.sigmoid(self.gate_proj(x)).squeeze(-1)  # (B, S)

        # New exits: not already exited AND confidence above threshold
        newly_exited = (~exit_mask) & (confidence > self.threshold)
        new_mask = exit_mask | newly_exited
        return x, new_mask, confidence

    def auxiliary_loss(self, confidence: Tensor) -> Tensor:
        """
        Encourage decisive gates — push confidence toward 0 or 1.
        Loss = -E[|conf - 0.5|] so the model is penalised for being uncertain.
        """
        return -(confidence - 0.5).abs().mean()

    def __repr__(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return f"ExitGate(threshold={self.threshold}, params={p:,})"
