"""
test_forward.py — full end-to-end forward pass tests.

Run: pytest tests/test_forward.py -v
"""
import pytest
import torch

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
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"


def test_exit_mask_monotone(model, input_ids):
    """Once a token exits, it must stay exited in subsequent layers."""
    out = model(input_ids)
    for i in range(1, len(out.exit_masks)):
        prev = out.exit_masks[i - 1]
        curr = out.exit_masks[i]
        assert (curr[prev]).all(), "Exit mask is not monotonically set"


def test_no_nan_in_logits(model, input_ids):
    out = model(input_ids)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def test_model_repr(model):
    r = repr(model)
    assert "TMTModel" in r
    assert "params" in r


def test_param_count_positive(model):
    assert model.param_count() > 0


def test_different_inputs_different_outputs(model):
    ids_a = torch.randint(0, CFG.vocab_size, (B, S))
    ids_b = torch.randint(0, CFG.vocab_size, (B, S))
    model.eval()
    with torch.no_grad():
        out_a = model(ids_a)
        out_b = model(ids_b)
    # Different tokens → different logits (with overwhelming probability)
    assert not torch.allclose(out_a.logits, out_b.logits)


def test_memory_state_is_tensor(model, input_ids):
    out = model(input_ids)
    assert isinstance(out.memory_state, torch.Tensor)
    assert out.memory_state.shape == (CFG.memory_anchors, CFG.d_model)


def test_graph_edges_valid(model, input_ids):
    out = model(input_ids)
    edge_index, edge_weight = out.graph_edges
    assert edge_index.shape[0] == 2
    assert edge_weight.shape[0] == edge_index.shape[1]
    assert edge_index.max() < B * S
    assert edge_index.min() >= 0


def test_decay_scalars_bounded(model, input_ids):
    out = model(input_ids)
    assert out.decay_scalars.min() >= 0.0
    assert out.decay_scalars.max() <= 1.0


def test_model_train_eval_modes(model, input_ids):
    model.train()
    assert model.training
    model.eval()
    assert not model.training
    with torch.no_grad():
        out = model(input_ids)
    assert out.logits.shape == (B, S, CFG.vocab_size)


def test_forward_with_max_seq_len():
    cfg = TMTConfig(
        vocab_size=200, d_model=32, n_heads=2, n_layers=2,
        max_seq_len=16, graph_k=2, ffn_stream_dim=16, memory_anchors=2,
    )
    model = TMTModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    out = model(ids)
    assert out.logits.shape == (1, cfg.max_seq_len, cfg.vocab_size)


def test_confidences_all_bounded(model, input_ids):
    out = model(input_ids)
    for i, conf in enumerate(out.confidences):
        assert conf.min() >= 0.0, f"Confidence at layer {i} has value < 0"
        assert conf.max() <= 1.0, f"Confidence at layer {i} has value > 1"
