"""
TMTConfig — central configuration for the TemporalMesh Transformer.

Novel vs standard: a single config surface that governs dynamic graph topology
(graph_k), per-token adaptive depth (exit_threshold), temporal decay rate, and
the dual-stream FFN — none of which exist in vanilla transformer configs.
"""
from dataclasses import dataclass, field


@dataclass
class TMTConfig:
    # Vocabulary & sequence
    vocab_size: int = 32000
    max_seq_len: int = 1024

    # Core dims
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12

    # Innovation 1 — Mesh Attention
    graph_k: int = 8  # each token connects to k nearest neighbours by cosine sim

    # Innovation 2 — Temporal decay
    decay_rate: float = 0.1  # base for learned temporal decay scalars

    # Innovation 3 — Adaptive depth routing
    exit_threshold: float = 0.85  # confidence above which a token exits early

    # Dual-stream FFN
    dual_stream: bool = True
    ffn_stream_dim: int = 256  # each stream is d_model // 2

    # Memory anchors
    memory_anchors: int = 16  # number of persistent KV memory parameter vectors

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __repr__(self) -> str:
        total_params_est = (
            self.vocab_size * self.d_model  # embedding
            + self.n_layers * (
                4 * self.d_model * self.d_model  # attention projections
                + 2 * self.d_model * self.ffn_stream_dim  # dual stream FFN
                + self.d_model  # exit gate + memory
            )
        )
        return (
            f"TMTConfig("
            f"vocab={self.vocab_size}, d={self.d_model}, "
            f"heads={self.n_heads}, layers={self.n_layers}, "
            f"k={self.graph_k}, decay={self.decay_rate}, "
            f"exit_thr={self.exit_threshold}, "
            f"~params={total_params_est / 1e6:.1f}M)"
        )
