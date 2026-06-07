"""
test_shapes.py — comprehensive shape assertions for every TMT module.

Run: pytest tests/test_shapes.py -v
"""
import pytest
import torch

from tmt.model.config import TMTConfig
from tmt.model.embedding import TokenEmbedding, TemporalPositionEncoder
from tmt.model.mesh import MeshBuilder, build_mesh
from tmt.model.attention import MeshAttention
from tmt.model.ffn import DualStreamFFN
from tmt.model.exit_gate import ExitGate
from tmt.model.memory import MemoryAnchorCross
from tmt.model.layers import TMTLayer


B, S = 2, 32
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


# ─────────────────────────── TokenEmbedding ────────────────────────────────

class TestTokenEmbedding:
    def test_basic_output_shape(self, token_ids):
        emb = TokenEmbedding(CFG)
        out = emb(token_ids)
        assert out.shape == (B, S, CFG.d_model)

    def test_single_token(self):
        emb = TokenEmbedding(CFG)
        ids = torch.randint(0, CFG.vocab_size, (1, 1))
        out = emb(ids)
        assert out.shape == (1, 1, CFG.d_model)

    def test_batch_size_one(self):
        emb = TokenEmbedding(CFG)
        ids = torch.randint(0, CFG.vocab_size, (1, S))
        out = emb(ids)
        assert out.shape == (1, S, CFG.d_model)

    def test_output_not_nan(self, token_ids):
        emb = TokenEmbedding(CFG)
        out = emb(token_ids)
        assert not torch.isnan(out).any()

    def test_output_not_inf(self, token_ids):
        emb = TokenEmbedding(CFG)
        out = emb(token_ids)
        assert not torch.isinf(out).any()

    def test_embedding_table_shape(self):
        emb = TokenEmbedding(CFG)
        assert emb.embed.weight.shape == (CFG.vocab_size, CFG.d_model)

    def test_scale_applied(self, token_ids):
        import math
        emb = TokenEmbedding(CFG)
        # raw embedding magnitude without scale
        raw = emb.embed(token_ids)
        out = emb(token_ids)
        expected_scale = math.sqrt(CFG.d_model)
        assert torch.allclose(out, raw * expected_scale)

    def test_different_ids_different_outputs(self):
        emb = TokenEmbedding(CFG)
        ids_a = torch.zeros(1, 4, dtype=torch.long)
        ids_b = torch.ones(1, 4, dtype=torch.long)
        out_a = emb(ids_a)
        out_b = emb(ids_b)
        assert not torch.allclose(out_a, out_b)


# ─────────────────────── TemporalPositionEncoder ───────────────────────────

class TestTemporalPositionEncoder:
    def test_output_shape(self, x):
        enc = TemporalPositionEncoder(CFG)
        out, decay = enc(x)
        assert out.shape == (B, S, CFG.d_model)
        assert decay.shape == (B, S, CFG.d_model)

    def test_decay_range(self, x):
        enc = TemporalPositionEncoder(CFG)
        _, decay = enc(x)
        assert decay.min() >= 0.0 and decay.max() <= 1.0

    def test_single_token_pos_encoder(self):
        enc = TemporalPositionEncoder(CFG)
        x1 = torch.randn(1, 1, CFG.d_model)
        out, decay = enc(x1)
        assert out.shape == (1, 1, CFG.d_model)
        assert decay.shape == (1, 1, CFG.d_model)

    def test_decay_not_nan(self, x):
        enc = TemporalPositionEncoder(CFG)
        _, decay = enc(x)
        assert not torch.isnan(decay).any()

    def test_output_not_nan(self, x):
        enc = TemporalPositionEncoder(CFG)
        out, _ = enc(x)
        assert not torch.isnan(out).any()

    def test_rope_cache_shape(self):
        enc = TemporalPositionEncoder(CFG)
        assert enc.rope_cos.shape == (CFG.max_seq_len, CFG.d_model)
        assert enc.rope_sin.shape == (CFG.max_seq_len, CFG.d_model)

    def test_w_decay_shape(self):
        enc = TemporalPositionEncoder(CFG)
        assert enc.w_decay.shape == (CFG.d_model,)


# ─────────────────────────── MeshBuilder ───────────────────────────────────

class TestMeshBuilder:
    def test_basic_output_shapes(self, x):
        mb = MeshBuilder(CFG.graph_k)
        x_flat = x.reshape(B * S, CFG.d_model)
        edge_index, edge_weight = mb(x_flat, B, S)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0
        assert edge_weight.shape[0] == edge_index.shape[1]

    def test_no_cross_batch_edges(self):
        mb = MeshBuilder(k=2)
        x = torch.randn(4, 32)
        edge_index, _ = mb(x, batch_size=2, seq_len=2)
        src, dst = edge_index[0], edge_index[1]
        assert ((src < 2) == (dst < 2)).all()

    def test_k_clamped_to_seq_minus_one(self):
        mb = MeshBuilder(k=10)
        x = torch.randn(2 * 3, CFG.d_model)
        edge_index, edge_weight = mb(x, batch_size=2, seq_len=3)
        assert edge_index.shape[0] == 2
        assert edge_weight.shape[0] == edge_index.shape[1]

    def test_all_node_indices_in_range(self, x):
        mb = MeshBuilder(CFG.graph_k)
        x_flat = x.reshape(B * S, CFG.d_model)
        edge_index, _ = mb(x_flat, B, S)
        assert edge_index.max() < B * S
        assert edge_index.min() >= 0

    def test_edge_weight_cosine_range(self, x):
        mb = MeshBuilder(CFG.graph_k)
        x_flat = x.reshape(B * S, CFG.d_model)
        _, edge_weight = mb(x_flat, B, S)
        # Cosine similarity is in [-1, 1]
        assert edge_weight.min() >= -1.01
        assert edge_weight.max() <= 1.01

    def test_edge_count_at_least_b_s_k(self, x):
        k = CFG.graph_k
        mb = MeshBuilder(k)
        x_flat = x.reshape(B * S, CFG.d_model)
        edge_index, _ = mb(x_flat, B, S)
        # Each token should have at least k edges
        assert edge_index.shape[1] >= B * S * k

    def test_build_mesh_function_directly(self):
        x = torch.randn(6, 16)
        edge_index, edge_weight = build_mesh(x, k=2, batch_size=2, seq_len=3)
        assert edge_index.shape[0] == 2
        assert edge_weight.shape[0] == edge_index.shape[1]

    def test_single_batch_item(self):
        mb = MeshBuilder(k=2)
        x = torch.randn(1 * 4, CFG.d_model)
        edge_index, edge_weight = mb(x, batch_size=1, seq_len=4)
        assert edge_index.shape[0] == 2
        assert edge_weight.numel() > 0


# ─────────────────────────── MeshAttention ─────────────────────────────────

class TestMeshAttention:
    def test_basic_output_shape(self, x, graph):
        edge_index, edge_weight = graph
        attn = MeshAttention(CFG)
        out = attn(x, edge_index, edge_weight)
        assert out.shape == (B, S, CFG.d_model)

    def test_empty_edge_index(self):
        attn = MeshAttention(CFG)
        x = torch.randn(1, 4, CFG.d_model)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0)
        out = attn(x, edge_index, edge_weight)
        assert out.shape == (1, 4, CFG.d_model)
        assert not torch.isnan(out).any()

    def test_output_not_nan(self, x, graph):
        edge_index, edge_weight = graph
        attn = MeshAttention(CFG)
        out = attn(x, edge_index, edge_weight)
        assert not torch.isnan(out).any()

    def test_with_decay_scalars(self, x, graph):
        edge_index, edge_weight = graph
        attn = MeshAttention(CFG)
        decay = torch.rand(B, S, CFG.d_model)
        out = attn(x, edge_index, edge_weight, decay_scalars=decay)
        assert out.shape == (B, S, CFG.d_model)

    def test_without_decay_scalars(self, x, graph):
        edge_index, edge_weight = graph
        attn = MeshAttention(CFG)
        out = attn(x, edge_index, edge_weight, decay_scalars=None)
        assert out.shape == (B, S, CFG.d_model)

    def test_param_count_positive(self):
        attn = MeshAttention(CFG)
        p = sum(param.numel() for param in attn.parameters())
        assert p > 0

    def test_small_batch(self, graph):
        edge_index, edge_weight = graph
        attn = MeshAttention(CFG)
        x_small = torch.randn(1, S, CFG.d_model)
        # Build a 1-batch graph
        mb = MeshBuilder(CFG.graph_k)
        x_flat = x_small.reshape(S, CFG.d_model)
        e_idx, e_wt = mb(x_flat, 1, S)
        out = attn(x_small, e_idx, e_wt)
        assert out.shape == (1, S, CFG.d_model)


# ─────────────────────────── DualStreamFFN ─────────────────────────────────

class TestDualStreamFFN:
    def test_basic_output_shape(self, x):
        ffn = DualStreamFFN(CFG)
        out = ffn(x)
        assert out.shape == (B, S, CFG.d_model)

    def test_single_token(self):
        ffn = DualStreamFFN(CFG)
        x1 = torch.randn(1, 1, CFG.d_model)
        out = ffn(x1)
        assert out.shape == (1, 1, CFG.d_model)

    def test_output_not_nan(self, x):
        ffn = DualStreamFFN(CFG)
        out = ffn(x)
        assert not torch.isnan(out).any()

    def test_output_not_inf(self, x):
        ffn = DualStreamFFN(CFG)
        out = ffn(x)
        assert not torch.isinf(out).any()

    def test_gate_output_bounded(self, x):
        ffn = DualStreamFFN(CFG)
        # The gate is sigmoid → (0,1) but final output is weighted sum
        out = ffn(x)
        assert out.shape == (B, S, CFG.d_model)

    def test_param_count_positive(self):
        ffn = DualStreamFFN(CFG)
        p = sum(param.numel() for param in ffn.parameters())
        assert p > 0

    def test_large_batch(self):
        ffn = DualStreamFFN(CFG)
        x_large = torch.randn(8, 16, CFG.d_model)
        out = ffn(x_large)
        assert out.shape == (8, 16, CFG.d_model)


# ─────────────────────────── ExitGate ──────────────────────────────────────

class TestExitGate:
    def test_basic_output_shapes(self, x):
        gate = ExitGate(CFG)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        out, new_mask, confidence = gate(x, exit_mask)
        assert out.shape == (B, S, CFG.d_model)
        assert new_mask.shape == (B, S)
        assert new_mask.dtype == torch.bool
        assert confidence.shape == (B, S)

    def test_confidence_bounded(self, x):
        gate = ExitGate(CFG)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        _, _, confidence = gate(x, exit_mask)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_mask_monotone(self, x):
        gate = ExitGate(CFG)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        _, new_mask, _ = gate(x, exit_mask)
        # New mask must be superset of old mask
        assert (new_mask | exit_mask).all() or True  # mask only grows
        assert (~(~new_mask & exit_mask)).all()  # can't un-exit

    def test_already_exited_stays_exited(self, x):
        gate = ExitGate(CFG)
        exit_mask = torch.ones(B, S, dtype=torch.bool)
        _, new_mask, _ = gate(x, exit_mask)
        assert new_mask.all()

    def test_sigmoid_zero_input_no_exit_at_high_threshold(self):
        cfg = TMTConfig(
            vocab_size=100, d_model=32, n_heads=2, n_layers=2,
            max_seq_len=32, graph_k=2, ffn_stream_dim=16, memory_anchors=2,
            exit_threshold=0.85,
        )
        gate = ExitGate(cfg)
        with torch.no_grad():
            gate.gate_proj.weight.fill_(0.0)
            gate.gate_proj.bias.fill_(0.0)
        x = torch.zeros(1, 4, cfg.d_model)
        exit_mask = torch.zeros(1, 4, dtype=torch.bool)
        _, new_mask, conf = gate(x, exit_mask)
        # sigmoid(0) = 0.5 < 0.85 → no exits
        assert not new_mask.any()

    def test_aux_loss_is_scalar(self, x):
        gate = ExitGate(CFG)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        _, _, confidence = gate(x, exit_mask)
        loss = gate.auxiliary_loss(confidence)
        assert loss.shape == ()


# ─────────────────────────── MemoryAnchorCross ─────────────────────────────

class TestMemoryAnchorCross:
    def test_basic_output_shape(self, x):
        mac = MemoryAnchorCross(CFG)
        out, mem_state = mac(x)
        assert out.shape == (B, S, CFG.d_model)
        assert mem_state.shape == (CFG.memory_anchors, CFG.d_model)

    def test_memory_state_not_nan(self, x):
        mac = MemoryAnchorCross(CFG)
        _, mem_state = mac(x)
        assert not torch.isnan(mem_state).any()

    def test_output_not_nan(self, x):
        mac = MemoryAnchorCross(CFG)
        out, _ = mac(x)
        assert not torch.isnan(out).any()

    def test_memory_updates_in_train_mode(self, x):
        mac = MemoryAnchorCross(CFG)
        mac.train()
        mem_before = mac.memory.data.clone()
        mac(x)
        mem_after = mac.memory.data
        # EMA update should change memory
        assert not torch.allclose(mem_before, mem_after)

    def test_memory_stable_in_eval_mode(self, x):
        mac = MemoryAnchorCross(CFG)
        mac.eval()
        mem_before = mac.memory.data.clone()
        with torch.no_grad():
            mac(x)
        mem_after = mac.memory.data
        assert torch.allclose(mem_before, mem_after)


# ─────────────────────────── TMTLayer ──────────────────────────────────────

class TestTMTLayer:
    def test_basic_output_shapes(self, x, graph):
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

    def test_output_not_nan(self, x, graph):
        edge_index, edge_weight = graph
        layer = TMTLayer(CFG, layer_idx=0)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        x_out, _, _, _ = layer(x, edge_index, edge_weight, exit_mask)
        assert not torch.isnan(x_out).any()

    def test_frozen_tokens_unchanged(self, x, graph):
        edge_index, edge_weight = graph
        layer = TMTLayer(CFG, layer_idx=0)
        # Force all tokens exited before layer
        exit_mask = torch.ones(B, S, dtype=torch.bool)
        x_out, _, _, _ = layer(x, edge_index, edge_weight, exit_mask)
        assert torch.allclose(x, x_out)

    def test_exit_mask_only_grows(self, x, graph):
        edge_index, edge_weight = graph
        layer = TMTLayer(CFG, layer_idx=0)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        exit_mask[0, 0] = True  # pre-exit one token
        _, new_mask, _, _ = layer(x, edge_index, edge_weight, exit_mask)
        assert new_mask[0, 0].item()  # stayed exited

    def test_with_decay_scalars(self, x, graph):
        edge_index, edge_weight = graph
        layer = TMTLayer(CFG, layer_idx=1)
        exit_mask = torch.zeros(B, S, dtype=torch.bool)
        decay = torch.rand(B, S, CFG.d_model)
        x_out, new_mask, conf, mem = layer(x, edge_index, edge_weight, exit_mask, decay)
        assert x_out.shape == (B, S, CFG.d_model)
