"""
mesh.py — MeshBuilder: constructs a dynamic token graph each forward pass.

Novel vs standard: unlike Graph Transformers that use fixed pre-defined graphs,
MeshBuilder recomputes the graph topology at every forward pass using cosine
similarity of the current token representations.  Only the top-k nearest
neighbours are connected, giving a sparse O(S·k) edge set instead of O(S²).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def build_mesh(
    x: Tensor,
    k: int,
    batch_size: int,
    seq_len: int,
) -> Tuple[Tensor, Tensor]:
    """
    Build a dynamic kNN token graph from token embeddings.

    Args:
        x:          (B*S, D) flattened token representations
        k:          number of nearest neighbours per token (graph_k)
        batch_size: B
        seq_len:    S

    Returns:
        edge_index: (2, E) COO edge list in torch_geometric format.
                    Edges are within-batch — source and target indices are
                    global node indices (0 … B*S-1).
        edge_weight:(E,) cosine similarity of each edge.
    """
    N = batch_size * seq_len  # total nodes

    # Normalise for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)  # (N, D)

    # Block-diagonal cosine similarity — only connect tokens within same sample
    # so information never leaks across batch items
    sim_rows, sim_cols, sim_vals = [], [], []

    for b in range(batch_size):
        start = b * seq_len
        end = start + seq_len
        x_b = x_norm[start:end]  # (S, D)
        sim = x_b @ x_b.T  # (S, S) cosine sim matrix

        # Zero out self-connections
        sim.fill_diagonal_(float("-inf"))

        # Top-k neighbours per token
        actual_k = min(k, seq_len - 1)
        topk_vals, topk_idx = sim.topk(actual_k, dim=-1)  # (S, k)

        src = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(-1, actual_k)
        src = src.reshape(-1) + start
        dst = topk_idx.reshape(-1) + start
        vals = topk_vals.reshape(-1)

        sim_rows.append(src)
        sim_cols.append(dst)
        sim_vals.append(vals)

    edge_index = torch.stack(
        [torch.cat(sim_rows), torch.cat(sim_cols)], dim=0
    )  # (2, E)
    edge_weight = torch.cat(sim_vals)  # (E,)

    return edge_index, edge_weight


class MeshBuilder(torch.nn.Module):
    """Thin nn.Module wrapper around build_mesh so it shows in model.repr."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: Tensor, batch_size: int, seq_len: int) -> Tuple[Tensor, Tensor]:
        return build_mesh(x, self.k, batch_size, seq_len)

    def __repr__(self) -> str:
        return f"MeshBuilder(k={self.k}, params=0)"
