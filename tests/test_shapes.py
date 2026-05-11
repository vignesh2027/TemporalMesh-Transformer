"""
test_shapes.py — assert every module output shape is correct.

Run: pytest tests/test_shapes.py -v
"""
import torch
import pytest

from tmt.model.config import TMTConfig
from tmt.model.embedding import TokenEmbedding, TemporalPositionEncoder
from tmt.model.mesh import MeshBuilder
from tmt.model.attention import MeshAttention
from tmt.model.ffn import DualStreamFFN
from tmt.model.exit_gate import ExitGate
from tmt.model.memory import MemoryAnchorCross
from tmt.model.layers import TMTLayer


B, S = 2, 32   # small batch and sequence for fast tests
CFG = TMTConfig(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=2,
    max_seq_len=64,
    graph_k=4,
    ffn_stream_dim=32,
    memory_anchors=4,
)


@pytest.fixture
def token_ids():
    return torch.randint(0, CFG.vocab_size, (B, S))


@pytest.fixture
def x():
    return torch.randn(B, S, CFG.d_model)


@pytest.fixture
def graph(x):
    mb = MeshBuilder(CFG.graph_k)
    x_flat = x.reshape(B * S, CFG.d_model)
    return mb(x_flat, B, S)


def test_token_embedding(token_ids):
    emb = TokenEmbedding(CFG)
    out = emb(token_ids)
    assert out.shape == (B, S, CFG.d_model), f"Expected ({B}, {S}, {CFG.d_model}), got {out.shape}"


def test_temporal_position_encoder(x):
    enc = TemporalPositionEncoder(CFG)
    out, decay = enc(x)
    assert out.shape == (B, S, CFG.d_model)
    assert decay.shape == (B, S, CFG.d_model)
    # Decay scalars should be in (0, 1)
    assert decay.min() >= 0.0 and decay.max() <= 1.0


def test_mesh_builder(x):
    mb = MeshBuilder(CFG.graph_k)
    x_flat = x.reshape(B * S, CFG.d_model)
    edge_index, edge_weight = mb(x_flat, B, S)
    # edge_index: (2, E) where E = B * S * k
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    assert edge_weight.shape[0] == edge_index.shape[1]
    # All node indices in range [0, B*S)
    assert edge_index.max() < B * S
    assert edge_index.min() >= 0


def test_mesh_attention(x, graph):
    edge_index, edge_weight = graph
    attn = MeshAttention(CFG)
    out = attn(x, edge_index, edge_weight)
    assert out.shape == (B, S, CFG.d_model)


def test_dual_stream_ffn(x):
    ffn = DualStreamFFN(CFG)
    out = ffn(x)
    assert out.shape == (B, S, CFG.d_model)


def test_exit_gate(x):
    gate = ExitGate(CFG)
    exit_mask = torch.zeros(B, S, dtype=torch.bool)
    out, new_mask, confidence = gate(x, exit_mask)
    assert out.shape == (B, S, CFG.d_model)
    assert new_mask.shape == (B, S)
    assert new_mask.dtype == torch.bool
    assert confidence.shape == (B, S)
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0


def test_memory_anchor_cross(x):
    mac = MemoryAnchorCross(CFG)
    out, mem_state = mac(x)
    assert out.shape == (B, S, CFG.d_model)
    assert mem_state.shape == (CFG.memory_anchors, CFG.d_model)


def test_tmt_layer(x, graph):
    edge_index, edge_weight = graph
    layer = TMTLayer(CFG, layer_idx=0)
    exit_mask = torch.zeros(B, S, dtype=torch.bool)
    x_out, new_mask, confidence, mem_state = layer(
        x, edge_index, edge_weight, exit_mask
    )
    assert x_out.shape == (B, S, CFG.d_model)
    assert new_mask.shape == (B, S)
    assert confidence.shape == (B, S)
    assert mem_state.shape == (CFG.memory_anchors, CFG.d_model)
