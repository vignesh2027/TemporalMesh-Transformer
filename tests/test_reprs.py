"""
test_reprs.py — __repr__ coverage for every model component and TMTConfig.
"""
from tmt.model.attention import MeshAttention
from tmt.model.config import TMTConfig
from tmt.model.embedding import TemporalPositionEncoder, TokenEmbedding
from tmt.model.exit_gate import ExitGate
from tmt.model.ffn import DualStreamFFN
from tmt.model.layers import TMTLayer
from tmt.model.memory import MemoryAnchorCross
from tmt.model.mesh import MeshBuilder
from tmt.model.model import TMTModel

CFG = TMTConfig(
    vocab_size=200,
    d_model=32,
    n_heads=2,
    n_layers=2,
    max_seq_len=32,
    graph_k=2,
    ffn_stream_dim=16,
    memory_anchors=2,
)


def test_config_repr():
    r = repr(CFG)
    assert "TMTConfig" in r
    assert "vocab=200" in r
    assert "d=32" in r
    assert "heads=2" in r
    assert "layers=2" in r


def test_token_embedding_repr():
    r = repr(TokenEmbedding(CFG))
    assert "TokenEmbedding" in r
    assert "params=" in r


def test_temporal_position_encoder_repr():
    r = repr(TemporalPositionEncoder(CFG))
    assert "TemporalPositionEncoder" in r
    assert "d=32" in r


def test_mesh_builder_repr():
    r = repr(MeshBuilder(CFG.graph_k))
    assert "MeshBuilder" in r
    assert "k=2" in r
    assert "params=0" in r


def test_mesh_attention_repr():
    r = repr(MeshAttention(CFG))
    assert "MeshAttention" in r
    assert "heads=2" in r
    assert "d=32" in r


def test_dual_stream_ffn_repr():
    r = repr(DualStreamFFN(CFG))
    assert "DualStreamFFN" in r
    assert "streams=" in r


def test_exit_gate_repr():
    r = repr(ExitGate(CFG))
    assert "ExitGate" in r
    assert "threshold=" in r


def test_memory_anchor_cross_repr():
    r = repr(MemoryAnchorCross(CFG))
    assert "MemoryAnchorCross" in r
    assert "anchors=2" in r


def test_tmt_layer_repr():
    r = repr(TMTLayer(CFG, layer_idx=0))
    assert "TMTLayer" in r
    assert "idx=0" in r


def test_tmt_model_repr():
    r = repr(TMTModel(CFG))
    assert "TMTModel" in r
    assert "params" in r


def test_param_count_positive():
    model = TMTModel(CFG)
    assert model.param_count() > 0


def test_config_param_estimate_positive():
    r = repr(CFG)
    # repr includes ~params= with a float, just check it's there
    assert "~params=" in r
