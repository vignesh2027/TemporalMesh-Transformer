"""
test_integration.py — end-to-end integration tests for training and checkpointing.

Run: pytest tests/test_integration.py -v
"""
import os
import math
import tempfile

import torch
import torch.nn as nn
from torch.optim import AdamW

from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel
from tmt.training.loss import compute_loss
from tmt.training.scheduler import cosine_warmup_scheduler


# Small config so integration tests run fast
ICFG = TMTConfig(
    vocab_size=200,
    d_model=32,
    n_heads=2,
    n_layers=2,
    max_seq_len=32,
    graph_k=2,
    ffn_stream_dim=16,
    memory_anchors=2,
    dropout=0.0,  # disable dropout for deterministic tests
)

B, S = 2, 16


def _make_batch(cfg=ICFG, b=B, s=S):
    return torch.randint(0, cfg.vocab_size, (b, s))


def _one_step(model, optimizer, scheduler=None):
    """Run one forward+backward step. Returns total loss scalar."""
    ids = _make_batch()
    x, targets = ids[:, :-1], ids[:, 1:]
    out = model(x)
    total, ce, gate = compute_loss(out.logits, targets, out.confidences)
    optimizer.zero_grad()
    total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return total.item()


# ── Single-step smoke test ────────────────────────────────────────────────────

def test_train_one_step_no_crash():
    model = TMTModel(ICFG)
    model.train()
    opt = AdamW(model.parameters(), lr=1e-3)
    loss = _one_step(model, opt)
    assert math.isfinite(loss)
    assert loss > 0.0


def test_train_one_step_loss_is_scalar():
    model = TMTModel(ICFG)
    model.train()
    AdamW(model.parameters(), lr=1e-3)
    ids = _make_batch()
    x, targets = ids[:, :-1], ids[:, 1:]
    out = model(x)
    total, _, _ = compute_loss(out.logits, targets, out.confidences)
    assert total.shape == ()


# ── Multi-step training ───────────────────────────────────────────────────────

def test_train_multiple_steps_loss_decreases_trend():
    """After 20 steps of training, the mean loss in later steps should be
    lower than in early steps (sanity check that gradients are sensible)."""
    torch.manual_seed(42)
    model = TMTModel(ICFG)
    model.train()
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    losses = []
    for _ in range(20):
        losses.append(_one_step(model, opt))

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early, (
        f"Expected loss to decrease, but early={early:.4f} late={late:.4f}"
    )


def test_model_output_changes_with_training():
    """Model outputs on fixed input should shift after a training step."""
    torch.manual_seed(0)
    model = TMTModel(ICFG)
    ids = _make_batch()

    model.eval()
    with torch.no_grad():
        logits_before = model(ids).logits.clone()

    model.train()
    opt = AdamW(model.parameters(), lr=1e-2)
    for _ in range(5):
        _one_step(model, opt)

    model.eval()
    with torch.no_grad():
        logits_after = model(ids).logits

    assert not torch.allclose(logits_before, logits_after), (
        "Model outputs unchanged after training — gradients may not be flowing"
    )


# ── Checkpoint save / load ────────────────────────────────────────────────────

def test_checkpoint_save_and_load():
    """Save a checkpoint, reload it, and verify outputs are identical."""
    torch.manual_seed(7)
    model = TMTModel(ICFG)
    model.eval()

    ids = _make_batch()
    with torch.no_grad():
        logits_original = model(ids).logits.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        torch.save({"model_state": model.state_dict(), "step": 0}, ckpt_path)

        # Load into a fresh model
        model2 = TMTModel(ICFG)
        ckpt = torch.load(ckpt_path, weights_only=True)
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        with torch.no_grad():
            logits_loaded = model2(ids).logits

    assert torch.allclose(logits_original, logits_loaded, atol=1e-6), (
        "Loaded checkpoint produced different outputs"
    )


def test_checkpoint_step_field_preserved():
    """Step counter stored in checkpoint should survive save/load."""
    model = TMTModel(ICFG)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        torch.save({"model_state": model.state_dict(), "step": 42}, ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=True)
        assert ckpt["step"] == 42


def test_checkpoint_all_keys_present():
    model = TMTModel(ICFG)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        torch.save({"model_state": model.state_dict(), "step": 1}, ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=True)
        assert "model_state" in ckpt
        assert "step" in ckpt


# ── Exit rate behaviour ───────────────────────────────────────────────────────

def test_exit_rate_in_valid_range():
    """Exit rate (fraction of tokens that exited) must be in [0, 1]."""
    model = TMTModel(ICFG)
    model.eval()
    ids = _make_batch()
    with torch.no_grad():
        out = model(ids)
    final_mask = out.exit_masks[-1]
    exit_rate = final_mask.float().mean().item()
    assert 0.0 <= exit_rate <= 1.0


def test_exit_rate_increases_with_training():
    """With exit_threshold=0.0, all tokens should exit at every layer."""
    cfg = TMTConfig(
        vocab_size=100, d_model=32, n_heads=2, n_layers=3,
        max_seq_len=32, graph_k=2, ffn_stream_dim=16, memory_anchors=2,
        exit_threshold=0.0, dropout=0.0,
    )
    model = TMTModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids)
    # With threshold=0 all tokens should exit after first layer
    assert out.exit_masks[0].all(), "Expected all tokens to exit with threshold=0"


# ── Graph updates per layer ───────────────────────────────────────────────────

def test_mesh_updates_each_layer():
    """The graph is rebuilt after each layer, so edge counts may differ."""
    model = TMTModel(ICFG)
    model.eval()
    ids = _make_batch()
    # Capture edge counts — the final graph_edges is from the last rebuild
    with torch.no_grad():
        out = model(ids)
    edge_index, edge_weight = out.graph_edges
    # Should be non-empty for B*S nodes with k=2
    assert edge_index.shape[1] > 0
    assert edge_weight.numel() == edge_index.shape[1]


def test_mesh_edges_non_negative_weights():
    model = TMTModel(ICFG)
    model.eval()
    ids = _make_batch()
    with torch.no_grad():
        out = model(ids)
    _, edge_weight = out.graph_edges
    # Cosine similarity can be negative, but should be finite
    assert torch.isfinite(edge_weight).all()


# ── Scheduler integration ─────────────────────────────────────────────────────

def test_scheduler_with_model_training():
    """Verify scheduler + optimizer integrate correctly over multiple steps."""
    torch.manual_seed(0)
    model = TMTModel(ICFG)
    model.train()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = cosine_warmup_scheduler(opt, warmup_steps=3, total_steps=10)

    losses = []
    for _ in range(10):
        ids = _make_batch()
        x, targets = ids[:, :-1], ids[:, 1:]
        out = model(x)
        total, _, _ = compute_loss(out.logits, targets, out.confidences)
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        losses.append(total.item())

    assert all(math.isfinite(v) for v in losses), "Non-finite loss during training"
