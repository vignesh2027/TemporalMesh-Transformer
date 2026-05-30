"""
test_training.py — tests for loss, scheduler, and exit gate auxiliary loss.
"""
import pytest
import torch

from tmt.model.config import TMTConfig
from tmt.model.exit_gate import ExitGate
from tmt.training.loss import compute_loss
from tmt.training.scheduler import cosine_warmup_scheduler

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


# ── Loss ──────────────────────────────────────────────────────────────────────

def _dummy_inputs():
    logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))
    confidences = [torch.rand(B, S) for _ in range(3)]
    return logits, targets, confidences


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


# ── Scheduler ─────────────────────────────────────────────────────────────────

def _make_scheduler(warmup=10, total=100, min_lr_ratio=0.1):
    import torch.nn as nn
    dummy = nn.Linear(1, 1)
    opt = torch.optim.SGD(dummy.parameters(), lr=1.0)
    sched = cosine_warmup_scheduler(opt, warmup, total, min_lr_ratio)
    return opt, sched


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
    # After warmup, lr should decrease monotonically
    assert lrs == sorted(lrs, reverse=True)


# ── ExitGate auxiliary loss ────────────────────────────────────────────────────

def test_exit_gate_aux_loss_all_certain():
    gate = ExitGate(CFG)
    conf = torch.ones(B, S)
    loss = gate.auxiliary_loss(conf)
    # conf=1 → |1 - 0.5| = 0.5 → aux_loss = -0.5
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
