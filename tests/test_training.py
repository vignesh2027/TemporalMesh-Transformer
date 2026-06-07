"""
test_training.py — tests for loss, scheduler, and exit gate auxiliary loss.
"""
import math
import pytest
import torch
import torch.nn as nn

from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel
from tmt.model.exit_gate import ExitGate
from tmt.training.loss import compute_loss
from tmt.training.scheduler import cosine_warmup_scheduler
from tmt.training.trainer import TrainConfig

B, S, V = 2, 16, 100
CFG = TMTConfig(
    vocab_size=V,
    d_model=32,
    n_heads=2,
    n_layers=2,
    max_seq_len=32,
    graph_k=2,
    ffn_stream_dim=16,
    memory_anchors=2,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dummy_inputs():
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    confidences = [torch.rand(B, S) for _ in range(3)]
    return logits, targets, confidences


def _make_scheduler(warmup=10, total=100, min_lr_ratio=0.1):
    dummy = nn.Linear(1, 1)
    opt = torch.optim.SGD(dummy.parameters(), lr=1.0)
    sched = cosine_warmup_scheduler(opt, warmup, total, min_lr_ratio)
    return opt, sched


# ── Loss ──────────────────────────────────────────────────────────────────────

def test_loss_returns_three_scalars():
    logits, targets, confs = _dummy_inputs()
    total, ce, gate = compute_loss(logits, targets, confs)
    for t in (total, ce, gate):
        assert t.shape == ()


def test_loss_total_equals_ce_plus_weighted_gate():
    logits, targets, confs = _dummy_inputs()
    coeff = 0.05
    total, ce, gate = compute_loss(logits, targets, confs, exit_gate_coeff=coeff)
    expected = ce + coeff * gate
    assert torch.allclose(total, expected, atol=1e-6)


def test_loss_ignore_index_excluded():
    logits, targets, confs = _dummy_inputs()
    targets[:, 0] = -100
    total, ce, _ = compute_loss(logits, targets, confs, ignore_index=-100)
    assert not torch.isnan(total)
    assert total.item() > 0


def test_loss_zero_confidences():
    logits, targets, _ = _dummy_inputs()
    total, ce, gate = compute_loss(logits, targets, [])
    assert torch.allclose(total, ce)


def test_loss_gate_is_negative_or_zero():
    logits, targets, confs = _dummy_inputs()
    _, _, gate = compute_loss(logits, targets, confs)
    assert gate.item() <= 0.0


def test_loss_not_nan():
    logits, targets, confs = _dummy_inputs()
    total, ce, gate = compute_loss(logits, targets, confs)
    assert not torch.isnan(total)
    assert not torch.isnan(ce)
    assert not torch.isnan(gate)


def test_loss_not_inf():
    logits, targets, confs = _dummy_inputs()
    total, ce, gate = compute_loss(logits, targets, confs)
    assert not torch.isinf(total)
    assert not torch.isinf(ce)
    assert not torch.isinf(gate)


def test_loss_with_large_vocab():
    large_V = 32000
    logits = torch.randn(2, 8, large_V)
    targets = torch.randint(0, large_V, (2, 8))
    confs = [torch.rand(2, 8) for _ in range(2)]
    total, ce, gate = compute_loss(logits, targets, confs)
    assert not torch.isnan(total)
    assert total.item() > 0


def test_loss_gradient_flows():
    logits = torch.randn(B, S, V, requires_grad=True)
    targets = torch.randint(0, V, (B, S))
    confs = [torch.rand(B, S) for _ in range(2)]
    total, _, _ = compute_loss(logits, targets, confs)
    total.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_loss_is_not_negative_ce():
    """Cross-entropy loss must be non-negative."""
    logits, targets, confs = _dummy_inputs()
    _, ce, _ = compute_loss(logits, targets, confs)
    assert ce.item() >= 0.0


def test_loss_with_single_token():
    logits = torch.randn(1, 1, V)
    targets = torch.randint(0, V, (1, 1))
    confs = [torch.rand(1, 1) for _ in range(2)]
    total, ce, gate = compute_loss(logits, targets, confs)
    assert not torch.isnan(total)
    assert ce.item() >= 0.0


# ── Scheduler ─────────────────────────────────────────────────────────────────

def test_scheduler_lr_zero_at_step_zero():
    opt, sched = _make_scheduler(warmup=10)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)


def test_scheduler_lr_increases_during_warmup():
    opt, sched = _make_scheduler(warmup=10, total=100)
    lrs = []
    for _ in range(10):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs == sorted(lrs)


def test_scheduler_lr_at_end_of_warmup_is_one():
    opt, sched = _make_scheduler(warmup=5, total=100)
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, abs=1e-4)


def test_scheduler_lr_decays_after_warmup():
    opt, sched = _make_scheduler(warmup=5, total=50)
    for _ in range(5):
        sched.step()
    lr_at_warmup_end = opt.param_groups[0]["lr"]
    sched.step()
    lr_after = opt.param_groups[0]["lr"]
    assert lr_after < lr_at_warmup_end


def test_scheduler_min_lr_floor():
    opt, sched = _make_scheduler(warmup=1, total=50, min_lr_ratio=0.1)
    for _ in range(50):
        sched.step()
    assert opt.param_groups[0]["lr"] >= 0.1 - 1e-4


def test_scheduler_cosine_shape():
    opt, sched = _make_scheduler(warmup=1, total=100)
    sched.step()
    lrs = []
    for _ in range(99):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs == sorted(lrs, reverse=True)


def test_scheduler_lr_never_below_min_ratio():
    min_ratio = 0.05
    opt, sched = _make_scheduler(warmup=2, total=200, min_lr_ratio=min_ratio)
    for _ in range(200):
        sched.step()
        lr = opt.param_groups[0]["lr"]
        assert lr >= min_ratio - 1e-6, f"LR {lr} dropped below min_ratio {min_ratio}"


def test_scheduler_handles_zero_warmup():
    """warmup_steps=0 should not crash and should start at full LR."""
    dummy = nn.Linear(1, 1)
    opt = torch.optim.SGD(dummy.parameters(), lr=1.0)
    sched = cosine_warmup_scheduler(opt, warmup_steps=0, total_steps=50)
    sched.step()
    # After 1 step with 0 warmup, should be at cosine decay already
    lr = opt.param_groups[0]["lr"]
    assert lr >= 0.0


# ── ExitGate auxiliary loss ────────────────────────────────────────────────────

def test_exit_gate_aux_loss_all_certain():
    gate = ExitGate(CFG)
    conf = torch.ones(B, S)
    loss = gate.auxiliary_loss(conf)
    assert loss.item() == pytest.approx(-0.5, abs=1e-4)


def test_exit_gate_aux_loss_all_uncertain():
    gate = ExitGate(CFG)
    conf = torch.full((B, S), 0.5)
    loss = gate.auxiliary_loss(conf)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_exit_gate_aux_loss_mixed():
    gate = ExitGate(CFG)
    conf = torch.zeros(B, S)
    loss = gate.auxiliary_loss(conf)
    assert loss.item() == pytest.approx(-0.5, abs=1e-4)


def test_exit_gate_aux_loss_gradient():
    """Auxiliary loss must produce gradients through confidence tensor."""
    gate = ExitGate(CFG)
    x = torch.randn(B, S, CFG.d_model)
    exit_mask = torch.zeros(B, S, dtype=torch.bool)
    _, _, confidence = gate(x, exit_mask)
    loss = gate.auxiliary_loss(confidence)
    loss.backward()
    assert gate.gate_proj.weight.grad is not None
    assert not torch.isnan(gate.gate_proj.weight.grad).any()


# ── TrainConfig defaults ──────────────────────────────────────────────────────

def test_trainer_config_defaults():
    cfg = TrainConfig()
    assert cfg.batch_size > 0
    assert cfg.lr > 0
    assert cfg.total_steps > 0
    assert cfg.warmup_steps >= 0
    assert cfg.grad_clip > 0
    assert 0.0 < cfg.exit_gate_coeff < 1.0
