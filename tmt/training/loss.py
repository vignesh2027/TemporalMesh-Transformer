"""
loss.py — TMT combined training loss.

Total loss = cross_entropy(logits, targets)
           + 0.1 * exit_gate_auxiliary_loss

The auxiliary loss encourages exit gates to be decisive (confident 0 or 1)
without forcing specific tokens to exit.  The coefficient 0.1 keeps it small
enough not to override the language modelling objective.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_loss(
    logits: Tensor,
    targets: Tensor,
    confidences: List[Tensor],
    exit_gate_coeff: float = 0.1,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        logits:          (B, S, V) model output logits
        targets:         (B, S)   integer ground-truth token ids
        confidences:     list of (B, S) per-layer gate confidence scores
        exit_gate_coeff: weight for auxiliary exit gate loss
        ignore_index:    token id to exclude from cross-entropy

    Returns:
        total_loss:   scalar
        ce_loss:      scalar cross-entropy component
        gate_loss:    scalar auxiliary gate component
    """
    B, S, V = logits.shape

    # Standard next-token cross-entropy (flat over B*S)
    ce_loss = F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
        ignore_index=ignore_index,
    )

    # Exit gate auxiliary: encourage decisiveness
    # Loss = -E[|conf - 0.5|] — penalise uncertainty
    gate_loss = torch.zeros(1, device=logits.device)
    for conf in confidences:
        gate_loss = gate_loss + -(conf - 0.5).abs().mean()
    gate_loss = gate_loss / max(len(confidences), 1)

    total_loss = ce_loss + exit_gate_coeff * gate_loss
    return total_loss, ce_loss, gate_loss
