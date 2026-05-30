"""
model.py — TMTModel: full TemporalMesh Transformer.

Assembles: TokenEmbedding → TemporalPositionEncoder → MeshBuilder →
           TMTLayer × n_layers → OutputProjection.

Every forward pass returns a TMTOutput dataclass containing logits plus all
intermediate diagnostic tensors (exit_masks, graph edges, memory state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import TMTConfig
from .embedding import TemporalPositionEncoder, TokenEmbedding
from .layers import TMTLayer
from .mesh import MeshBuilder


@dataclass
class TMTOutput:
    logits: Tensor                        # (B, S, V)
    exit_masks: List[Tensor]              # per-layer (B, S) bool
    confidences: List[Tensor]             # per-layer (B, S) float
    graph_edges: Tuple[Tensor, Tensor]    # (edge_index, edge_weight)
    memory_state: Tensor                  # (M, D) final memory anchors
    decay_scalars: Tensor                 # (B, S, D) temporal decay weights


class TMTModel(nn.Module):
    """Full TemporalMesh Transformer."""

    def __init__(self, cfg: TMTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = TokenEmbedding(cfg)
        self.pos_encoder = TemporalPositionEncoder(cfg)
        self.mesh_builder = MeshBuilder(cfg.graph_k)
        self.layers = nn.ModuleList(
            [TMTLayer(cfg, i) for i in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie output projection weights to embedding for parameter efficiency
        self.output_proj.weight = self.embedding.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> TMTOutput:
        """
        Args:
            input_ids: (B, S) integer token ids
        Returns:
            TMTOutput with logits and all diagnostic fields
        """
        B, S = input_ids.shape

        # Phase 1: embed + temporal position encode
        x = self.embedding(input_ids)                      # (B, S, D)
        x, decay_scalars = self.pos_encoder(x)             # (B, S, D), (B, S, D)

        # Phase 2: build dynamic mesh graph
        x_flat = x.reshape(B * S, self.cfg.d_model)
        edge_index, edge_weight = self.mesh_builder(x_flat, B, S)

        # Phase 3: pass through TMT layers with adaptive depth routing
        exit_mask = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        exit_masks: List[Tensor] = []
        confidences: List[Tensor] = []
        memory_state: Optional[Tensor] = None

        for layer in self.layers:
            x, exit_mask, confidence, memory_state = layer(
                x, edge_index, edge_weight, exit_mask, decay_scalars
            )
            exit_masks.append(exit_mask.clone())
            confidences.append(confidence.clone())

            # Rebuild graph after each layer using updated representations
            x_flat = x.reshape(B * S, self.cfg.d_model)
            edge_index, edge_weight = self.mesh_builder(x_flat, B, S)

        # Phase 4: project to vocabulary
        x = self.norm(x)
        logits = self.output_proj(x)  # (B, S, V)

        return TMTOutput(
            logits=logits,
            exit_masks=exit_masks,
            confidences=confidences,
            graph_edges=(edge_index, edge_weight),
            memory_state=memory_state,
            decay_scalars=decay_scalars,
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"TMTModel(\n"
            f"  cfg={self.cfg},\n"
            f"  total_params={self.param_count() / 1e6:.2f}M\n"
            f")"
        )
