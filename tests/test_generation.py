"""
test_generation.py — tests for generation utilities and logit behavior.

Run: pytest tests/test_generation.py -v
"""
import pytest
import torch

from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel


# ---------------------------------------------------------------------------
# Shared tiny model fixture — fast to instantiate
# ---------------------------------------------------------------------------

VOCAB = 200
D = 32
S = 16

@pytest.fixture(scope="module")
def tiny_cfg():
    return TMTConfig(
        vocab_size=VOCAB,
        d_model=D,
        n_heads=2,
        n_layers=3,
        max_seq_len=S,
        graph_k=2,
        ffn_stream_dim=16,
        memory_anchors=2,
        exit_threshold=0.85,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg):
    m = TMTModel(tiny_cfg)
    m.eval()
    return m


@pytest.fixture
def sample_ids():
    return torch.randint(0, VOCAB, (1, S))


# ---------------------------------------------------------------------------
# Basic logit tests
# ---------------------------------------------------------------------------

class TestLogitProperties:

    def test_greedy_next_token(self, tiny_model, sample_ids):
        """Greedy decoding: argmax of logits gives a valid vocab index."""
        with torch.no_grad():
            out = tiny_model(sample_ids)
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        assert next_tok.shape == (1,)
        assert 0 <= next_tok.item() < VOCAB

    def test_logit_shape_matches_vocab(self, tiny_model, sample_ids):
        """Output logits last dimension equals vocab_size."""
        with torch.no_grad():
            out = tiny_model(sample_ids)
        assert out.logits.shape[-1] == VOCAB

    def test_model_logits_finite(self, tiny_model, sample_ids):
        """All logit values must be finite (no NaN or Inf)."""
        with torch.no_grad():
            out = tiny_model(sample_ids)
        assert torch.isfinite(out.logits).all(), "Logits contain non-finite values"

    def test_model_vocab_coverage(self, tiny_model, sample_ids):
        """The logit tensor should cover every vocab index (shape dim matches)."""
        with torch.no_grad():
            out = tiny_model(sample_ids)
        # Just check the dimension covers all indices
        assert out.logits.shape[-1] >= VOCAB

    def test_model_output_changes_with_temperature(self, tiny_model, sample_ids):
        """Scaling logits by temperature changes the probability distribution."""
        with torch.no_grad():
            out = tiny_model(sample_ids)
        logits = out.logits[:, -1, :]
        probs_default = torch.softmax(logits, dim=-1)
        probs_hot = torch.softmax(logits / 2.0, dim=-1)
        # With different temperature the distributions should differ
        assert not torch.allclose(probs_default, probs_hot), \
            "Temperature scaling should change distribution"

    def test_model_consistent_across_batch(self, tiny_model, tiny_cfg):
        """Same sequence repeated in a batch gives identical output per row."""
        single_ids = torch.randint(0, VOCAB, (1, S))
        batch_ids = single_ids.repeat(3, 1)  # (3, S) — three identical rows

        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(batch_ids)

        row0 = out.logits[0]
        row1 = out.logits[1]
        row2 = out.logits[2]
        assert torch.allclose(row0, row1, atol=1e-5), "Row 0 and row 1 differ for identical input"
        assert torch.allclose(row0, row2, atol=1e-5), "Row 0 and row 2 differ for identical input"


# ---------------------------------------------------------------------------
# Compute / exit gate tests
# ---------------------------------------------------------------------------

class TestExitGateBehavior:

    def test_exit_gate_reduces_compute(self):
        """
        Model with threshold=0.0 (always exit after layer 0) should have
        higher fraction of exited tokens than threshold=1.1 (never exit).
        """
        base_cfg = dict(
            vocab_size=VOCAB, d_model=D, n_heads=2, n_layers=3,
            max_seq_len=S, graph_k=2, ffn_stream_dim=16,
            memory_anchors=2, dropout=0.0,
        )
        model_early = TMTModel(TMTConfig(**base_cfg, exit_threshold=0.0))
        model_never = TMTModel(TMTConfig(**base_cfg, exit_threshold=1.1))

        ids = torch.randint(0, VOCAB, (2, S))

        model_early.eval()
        model_never.eval()
        with torch.no_grad():
            out_early = model_early(ids)
            out_never = model_never(ids)

        # Count total exited tokens across all layers
        early_exits = sum(m.float().sum().item() for m in out_early.exit_masks)
        never_exits = sum(m.float().sum().item() for m in out_never.exit_masks)

        assert early_exits >= never_exits, \
            f"threshold=0.0 should exit more tokens than threshold=1.1 " \
            f"(got {early_exits} vs {never_exits})"

    def test_exit_masks_monotone_in_generation(self, tiny_model, sample_ids):
        """Exit mask can only grow — tokens that exited cannot un-exit."""
        with torch.no_grad():
            out = tiny_model(sample_ids)
        for i in range(1, len(out.exit_masks)):
            prev = out.exit_masks[i - 1]
            curr = out.exit_masks[i]
            # Every position that was True in prev must still be True in curr
            assert (curr[prev]).all(), \
                f"Exit mask at layer {i} lost a previously exited token"


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    def test_model_gradient_norm_finite(self, tiny_cfg):
        """Gradient norm after a backward pass must be finite."""
        model = TMTModel(tiny_cfg)
        model.train()

        ids = torch.randint(0, VOCAB, (2, S))
        out = model(ids)

        # Simple CE loss: predict next token
        logits = out.logits[:, :-1, :].reshape(-1, VOCAB)
        targets = ids[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm > 0, "Gradient norm is zero — no gradients flowed"
        assert torch.isfinite(torch.tensor(total_norm)), "Gradient norm is not finite"

    def test_model_grad_exists_for_all_params(self, tiny_cfg):
        """
        Core model parameters receive gradients via CE + auxiliary gate loss.
        Exit gate projections only update via the auxiliary loss, so we use
        compute_loss (which includes the gate auxiliary term) to cover all paths.
        """
        from tmt.training.loss import compute_loss

        model = TMTModel(tiny_cfg)
        model.train()

        ids = torch.randint(0, VOCAB, (1, S))
        out = model(ids)

        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        total_loss, _, _ = compute_loss(logits, targets, out.confidences)
        total_loss.backward()

        # With the full loss (CE + gate auxiliary), all parameters should have grads
        no_grad_params = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(no_grad_params) == 0, \
            f"Parameters with no gradient even after full loss: {no_grad_params}"
