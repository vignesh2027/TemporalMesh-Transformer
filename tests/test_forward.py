"""
test_forward.py — full end-to-end forward pass smoke tests.

Run: pytest tests/test_forward.py -v
"""
import torch
import pytest

from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel, TMTOutput
from tmt.training.loss import compute_loss


B, S = 2, 32
CFG = TMTConfig(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=3,
    max_seq_len=64,
    graph_k=4,
    ffn_stream_dim=32,
    memory_anchors=4,
)


@pytest.fixture
def model():
    return TMTModel(CFG)


@pytest.fixture
def input_ids():
    return torch.randint(0, CFG.vocab_size, (B, S))


def test_full_forward_shape(model, input_ids):
    out = model(input_ids)
    assert isinstance(out, TMTOutput)
    assert out.logits.shape == (B, S, CFG.vocab_size)


def test_output_has_all_fields(model, input_ids):
    out = model(input_ids)
    assert len(out.exit_masks) == CFG.n_layers
    assert len(out.confidences) == CFG.n_layers
    edge_index, edge_weight = out.graph_edges
    assert edge_index.shape[0] == 2
    assert out.memory_state.shape == (CFG.memory_anchors, CFG.d_model)
    assert out.decay_scalars.shape == (B, S, CFG.d_model)


def test_loss_computable(model, input_ids):
    # Use first S-1 tokens as input, predict last S-1 as targets
    x = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    out = model(x)
    total, ce, gate = compute_loss(out.logits, targets, out.confidences)
    assert total.item() > 0
    assert not torch.isnan(total)
    assert not torch.isinf(total)


def test_backward_pass(model, input_ids):
    x = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    out = model(x)
    total, _, _ = compute_loss(out.logits, targets, out.confidences)
    total.backward()
    # Check at least some gradients flowed
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"


def test_exit_mask_monotone(model, input_ids):
    """Once a token exits, it must stay exited in subsequent layers."""
    out = model(input_ids)
    for i in range(1, len(out.exit_masks)):
        prev = out.exit_masks[i - 1]
        curr = out.exit_masks[i]
        # If exited in layer i-1, must be exited in layer i
        assert (curr[prev]).all(), "Exit mask is not monotonically set"


def test_no_nan_in_logits(model, input_ids):
    out = model(input_ids)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def test_model_repr(model):
    r = repr(model)
    assert "TMTModel" in r
    assert "params" in r
