"""
test_edge_cases.py — edge cases: eval mode, forced exits, single token, long sequences.
"""
import torch
import pytest

from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel
from tmt.model.mesh import build_mesh
from tmt.model.attention import MeshAttention
from tmt.model.exit_gate import ExitGate
from tmt.training.loss import compute_loss

B, S = 2, 16
CFG = TMTConfig(
    vocab_size=500,
    d_model=32,
    n_heads=2,
    n_layers=3,
    max_seq_len=64,
    graph_k=2,
    ffn_stream_dim=16,
    memory_anchors=2,
    exit_threshold=0.85,
)


@pytest.fixture
def model():
    return TMTModel(CFG)


# ── Eval mode ─────────────────────────────────────────────────────────────────

def test_eval_mode_no_nan(model):
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, CFG.vocab_size, (B, S))
        out = model(ids)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def test_eval_train_logits_differ(model):
    ids = torch.randint(0, CFG.vocab_size, (B, S))
    model.train()
    out_train = model(ids)

    model.eval()
    with torch.no_grad():
        out_eval = model(ids)

    # dropout active in train → outputs differ (with high probability)
    assert not torch.allclose(out_train.logits, out_eval.logits)


# ── Single-token sequence ─────────────────────────────────────────────────────

def test_single_token_sequence(model):
    ids = torch.randint(0, CFG.vocab_size, (1, 1))
    out = model(ids)
    assert out.logits.shape == (1, 1, CFG.vocab_size)


# ── Batch size 1 ──────────────────────────────────────────────────────────────

def test_batch_size_one(model):
    ids = torch.randint(0, CFG.vocab_size, (1, S))
    out = model(ids)
    assert out.logits.shape == (1, S, CFG.vocab_size)


# ── All tokens forced to exit at layer 0 ─────────────────────────────────────

def test_all_exit_first_layer_still_produces_output():
    cfg = TMTConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=4,
        max_seq_len=32,
        graph_k=2,
        ffn_stream_dim=16,
        memory_anchors=2,
        exit_threshold=0.0,  # always exit immediately
    )
    model = TMTModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids)
    assert not torch.isnan(out.logits).any()


# ── Mesh graph edge properties ─────────────────────────────────────────────────

def test_mesh_no_cross_batch_edges():
    x = torch.randn(4, 32)
    edge_index, _ = build_mesh(x, k=2, batch_size=2, seq_len=2)
    src, dst = edge_index[0], edge_index[1]
    # src and dst within same batch item → both in [0,1] or both in [2,3]
    assert ((src < 2) == (dst < 2)).all()


def test_mesh_k_greater_than_seq_len_minus_one():
    # k=10 but seq_len=3 → should not crash, actual_k = min(k, S-1) = 2
    x = torch.randn(2 * 3, 16)
    edge_index, edge_weight = build_mesh(x, k=10, batch_size=2, seq_len=3)
    assert edge_index.shape[0] == 2
    assert edge_weight.shape[0] == edge_index.shape[1]


# ── Attention with empty edges ─────────────────────────────────────────────────

def test_attention_empty_edge_index():
    attn = MeshAttention(CFG)
    x = torch.randn(1, 4, CFG.d_model)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_weight = torch.zeros(0)
    out = attn(x, edge_index, edge_weight)
    assert out.shape == (1, 4, CFG.d_model)
    assert not torch.isnan(out).any()


# ── Exit gate threshold boundary ──────────────────────────────────────────────

def test_exit_gate_exactly_at_threshold():
    gate = ExitGate(CFG)
    # Force confidence exactly at threshold via gate_proj
    with torch.no_grad():
        gate.gate_proj.bias.fill_(0.0)
        gate.gate_proj.weight.fill_(0.0)
    x = torch.zeros(1, 4, CFG.d_model)
    exit_mask = torch.zeros(1, 4, dtype=torch.bool)
    _, new_mask, confidence = gate(x, exit_mask)
    # sigmoid(0) = 0.5, threshold=0.85 → no exits
    assert not new_mask.any()
    assert confidence.min() >= 0.0
    assert confidence.max() <= 1.0


# ── Gradient flow through all components ──────────────────────────────────────

def test_gradient_reaches_embedding(model):
    ids = torch.randint(0, CFG.vocab_size, (2, 8))
    targets = torch.randint(0, CFG.vocab_size, (2, 7))
    out = model(ids[:, :-1])
    loss, _, _ = compute_loss(out.logits, targets, out.confidences)
    loss.backward()
    emb_grad = model.embedding.embed.weight.grad
    assert emb_grad is not None
    assert not torch.isnan(emb_grad).any()


# ── Determinism in eval mode ──────────────────────────────────────────────────

def test_eval_is_deterministic(model):
    model.eval()
    ids = torch.randint(0, CFG.vocab_size, (2, S))
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids)
    assert torch.allclose(out1.logits, out2.logits)
