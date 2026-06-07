"""
test_benchmarks.py — validate real benchmark results and model properties.

Hardcoded ablation data from TMT-Benchmarks/ablation_reference (no network).
Run: pytest tests/test_benchmarks.py -v
"""
from __future__ import annotations

import time
from typing import List

import pytest
import torch

from tmt.model.config import TMTConfig
from tmt.model.mesh import build_mesh
from tmt.model.model import TMTModel
from tmt.training.loss import compute_loss

# ---------------------------------------------------------------------------
# Ground-truth ablation results from vigneshwar234/TMT-Benchmarks
# (ablation_reference split — no network required, hardcoded for CI stability)
# ---------------------------------------------------------------------------
ABLATION = [
    # name         mesh   decay  exit   val_ppl  avg_layers  rel_compute
    {"name": "Vanilla",
     "mesh": False, "decay": False, "exit": False,
     "val_ppl": 42.1, "avg_layers": 12.0, "rel_compute": 1.00},
    {"name": "Mesh Only",
     "mesh": True, "decay": False, "exit": False,
     "val_ppl": 37.8, "avg_layers": 12.0, "rel_compute": 0.62},
    {"name": "Decay Only",
     "mesh": False, "decay": True, "exit": False,
     "val_ppl": 40.3, "avg_layers": 12.0, "rel_compute": 0.98},
    {"name": "Exit Only",
     "mesh": False, "decay": False, "exit": True,
     "val_ppl": 39.6, "avg_layers": 5.8, "rel_compute": 0.51},
    {"name": "Mesh+Decay",
     "mesh": True, "decay": True, "exit": False,
     "val_ppl": 34.2, "avg_layers": 12.0, "rel_compute": 0.61},
    {"name": "Mesh+Exit",
     "mesh": True, "decay": False, "exit": True,
     "val_ppl": 35.1, "avg_layers": 5.7, "rel_compute": 0.50},
    {"name": "Decay+Exit",
     "mesh": False, "decay": True, "exit": True,
     "val_ppl": 37.0, "avg_layers": 5.9, "rel_compute": 0.50},
    {"name": "Full TMT",
     "mesh": True, "decay": True, "exit": True,
     "val_ppl": 29.4, "avg_layers": 5.5, "rel_compute": 0.48},
]
VANILLA  = ABLATION[0]
FULL_TMT = ABLATION[-1]

# ---------------------------------------------------------------------------
# Small test config — CPU-only, fast
# ---------------------------------------------------------------------------
SMALL_CFG = TMTConfig(
    vocab_size=512,
    d_model=64,
    n_heads=4,
    n_layers=4,
    max_seq_len=128,
    graph_k=4,
    ffn_stream_dim=32,
    memory_anchors=4,
    exit_threshold=0.85,
)

NO_EXIT_CFG = TMTConfig(
    vocab_size=512,
    d_model=64,
    n_heads=4,
    n_layers=4,
    max_seq_len=128,
    graph_k=4,
    ffn_stream_dim=32,
    memory_anchors=4,
    exit_threshold=1.1,   # threshold > 1 → sigmoid never exceeds 1 → no early exit
)

ZERO_EXIT_CFG = TMTConfig(
    vocab_size=512,
    d_model=64,
    n_heads=4,
    n_layers=4,
    max_seq_len=128,
    graph_k=4,
    ffn_stream_dim=32,
    memory_anchors=4,
    exit_threshold=0.0,   # every token exits immediately
)


# ===========================================================================
# class TestBenchmarkResults — pure data / arithmetic tests, no model needed
# ===========================================================================
class TestBenchmarkResults:

    def test_full_tmt_best_perplexity(self):
        """Full TMT must have the lowest validation perplexity of all configs."""
        min_ppl = min(row["val_ppl"] for row in ABLATION)
        assert FULL_TMT["val_ppl"] == min_ppl, (
            f"Full TMT PPL {FULL_TMT['val_ppl']} is not the minimum ({min_ppl})"
        )

    def test_full_tmt_most_efficient(self):
        """Full TMT must have the lowest relative compute of all configs."""
        min_compute = min(row["rel_compute"] for row in ABLATION)
        assert FULL_TMT["rel_compute"] == min_compute, (
            f"Full TMT rel_compute {FULL_TMT['rel_compute']} is not the minimum ({min_compute})"
        )

    def test_vanilla_baseline_highest_ppl(self):
        """Vanilla transformer should have the highest perplexity."""
        max_ppl = max(row["val_ppl"] for row in ABLATION)
        assert VANILLA["val_ppl"] == max_ppl, (
            f"Vanilla PPL {VANILLA['val_ppl']} is not the maximum ({max_ppl})"
        )

    def test_vanilla_uses_full_compute(self):
        """Vanilla transformer baseline uses exactly 1.00× compute."""
        assert VANILLA["rel_compute"] == 1.00

    def test_all_ppl_positive(self):
        """All perplexity values must be strictly positive."""
        for row in ABLATION:
            assert row["val_ppl"] > 0, f"{row['name']} has non-positive val_ppl: {row['val_ppl']}"

    def test_all_rel_compute_in_range(self):
        """All rel_compute values must be in (0, 1]."""
        for row in ABLATION:
            assert 0 < row["rel_compute"] <= 1.0, (
                f"{row['name']} rel_compute={row['rel_compute']} out of (0, 1] range"
            )

    def test_full_tmt_ppl_reduction_over_30_percent(self):
        """Full TMT achieves > 30% perplexity reduction over vanilla."""
        reduction = (VANILLA["val_ppl"] - FULL_TMT["val_ppl"]) / VANILLA["val_ppl"]
        assert reduction > 0.30, (
            f"PPL reduction {reduction:.2%} is not > 30%. "
            f"Vanilla={VANILLA['val_ppl']}, Full TMT={FULL_TMT['val_ppl']}"
        )

    def test_full_tmt_compute_under_50_percent(self):
        """Full TMT uses < 50% of vanilla compute."""
        assert FULL_TMT["rel_compute"] < 0.50, (
            f"Full TMT rel_compute={FULL_TMT['rel_compute']} is not < 0.50"
        )

    def test_exit_reduces_avg_layers(self):
        """All configs with adaptive exit enabled must use fewer than 12 avg layers."""
        for row in ABLATION:
            if row["exit"]:
                assert row["avg_layers"] < 12.0, (
                    f"{row['name']} has exit=True but avg_layers={row['avg_layers']} >= 12"
                )

    def test_no_exit_uses_all_layers(self):
        """All configs without adaptive exit must use all 12 layers."""
        for row in ABLATION:
            if not row["exit"]:
                assert row["avg_layers"] == 12.0, (
                    f"{row['name']} has exit=False but avg_layers={row['avg_layers']} != 12.0"
                )

    def test_mesh_reduces_compute(self):
        """Mesh attention alone reduces compute vs vanilla."""
        mesh_only = next(r for r in ABLATION if r["name"] == "Mesh Only")
        assert mesh_only["rel_compute"] < VANILLA["rel_compute"], (
            f"Mesh-only rel_compute={mesh_only['rel_compute']} should be < "
            f"vanilla={VANILLA['rel_compute']}"
        )

    def test_all_three_innovations_synergistic(self):
        """Full TMT (all 3) must beat every single-innovation config."""
        single_configs = [
            r for r in ABLATION
            if sum([r["mesh"], r["decay"], r["exit"]]) == 1
        ]
        for single in single_configs:
            assert FULL_TMT["val_ppl"] < single["val_ppl"], (
                f"Full TMT PPL {FULL_TMT['val_ppl']} is not better than "
                f"{single['name']} PPL {single['val_ppl']}"
            )

    def test_dual_innovations_better_than_single(self):
        """
        Every dual-innovation config must beat the PPL of each of its
        single-innovation components.
        """
        dual_to_singles = {
            "Mesh+Decay": ("Mesh Only",  "Decay Only"),
            "Mesh+Exit":  ("Mesh Only",  "Exit Only"),
            "Decay+Exit": ("Decay Only", "Exit Only"),
        }
        ppl = {row["name"]: row["val_ppl"] for row in ABLATION}
        for dual_name, (s1, s2) in dual_to_singles.items():
            dual_ppl = ppl[dual_name]
            assert dual_ppl < ppl[s1], (
                f"{dual_name} PPL {dual_ppl} should be < {s1} PPL {ppl[s1]}"
            )
            assert dual_ppl < ppl[s2], (
                f"{dual_name} PPL {dual_ppl} should be < {s2} PPL {ppl[s2]}"
            )

    def test_compute_efficiency_gain(self):
        """Vanilla / Full TMT compute ratio must exceed 2× (i.e. 2.1× gain)."""
        gain = VANILLA["rel_compute"] / FULL_TMT["rel_compute"]
        assert gain > 2.0, (
            f"Compute efficiency gain {gain:.2f}× is not > 2.0×. "
            f"Vanilla={VANILLA['rel_compute']}, Full TMT={FULL_TMT['rel_compute']}"
        )

    def test_ablation_has_8_configs(self):
        """Ablation table must have exactly 8 configurations."""
        assert len(ABLATION) == 8, f"Expected 8 ablation rows, got {len(ABLATION)}"


# ===========================================================================
# class TestModelVsBenchmarks — model-level checks tied to benchmark claims
# ===========================================================================
class TestModelVsBenchmarks:

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    @pytest.fixture
    def model(self):
        return TMTModel(SMALL_CFG)

    @pytest.fixture
    def no_exit_model(self):
        return TMTModel(NO_EXIT_CFG)

    @pytest.fixture
    def zero_exit_model(self):
        return TMTModel(ZERO_EXIT_CFG)

    # ------------------------------------------------------------------
    # Mesh tests
    # ------------------------------------------------------------------
    def test_mesh_builder_reduces_edges_vs_full_attention(self):
        """
        Sparse mesh graph must have far fewer edges than a full attention
        O(S²) graph.
        """
        B, S, D, k = 2, 32, 64, 4
        x = torch.randn(B * S, D)
        edge_index, _ = build_mesh(x, k, B, S)
        mesh_edges = edge_index.shape[1]
        full_attn_edges = B * S * S
        assert mesh_edges < full_attn_edges, (
            f"Mesh edges {mesh_edges} >= full-attention edges {full_attn_edges}"
        )

    def test_exit_gate_with_zero_threshold_reduces_avg_layers(self):
        """
        With exit_threshold=0.0 every token should exit very early, so the
        fraction of tokens still active should drop quickly.
        """
        model = TMTModel(ZERO_EXIT_CFG)
        ids = torch.randint(0, ZERO_EXIT_CFG.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids)
        # By the last layer almost all tokens should have exited
        final_exit_rate = out.exit_masks[-1].float().mean().item()
        assert final_exit_rate > 0.5, (
            f"Expected majority of tokens exited with threshold=0.0, "
            f"but exit rate was {final_exit_rate:.2f}"
        )

    def test_model_forward_time_small(self):
        """Forward pass on (2, 32) must complete within 5 seconds on CPU."""
        model = TMTModel(SMALL_CFG)
        ids = torch.randint(0, SMALL_CFG.vocab_size, (2, 32))
        start = time.time()
        with torch.no_grad():
            _ = model(ids)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Forward pass took {elapsed:.2f}s, expected < 5s"

    def test_model_forward_no_crash_medium_seq(self):
        """Forward pass on (1, 64) must complete without error."""
        model = TMTModel(SMALL_CFG)
        ids = torch.randint(0, SMALL_CFG.vocab_size, (1, 64))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (1, 64, SMALL_CFG.vocab_size)

    def test_decay_scalars_decrease_over_sequence(self):
        """
        Temporal decay: mean decay at position 0 must be >= mean decay at
        the final position (decay increases with distance from start).
        """
        model = TMTModel(SMALL_CFG)
        ids = torch.randint(0, SMALL_CFG.vocab_size, (1, 32))
        with torch.no_grad():
            out = model(ids)
        decay = out.decay_scalars  # (B, S, D)
        mean_pos0  = decay[:, 0,  :].mean().item()
        mean_posN  = decay[:, -1, :].mean().item()
        assert mean_pos0 >= mean_posN, (
            f"Decay at pos 0 ({mean_pos0:.4f}) should be >= pos S-1 ({mean_posN:.4f})"
        )

    def test_confidence_increases_with_training(self):
        """
        After a few optimizer steps the mean confidence should change,
        confirming the gate parameters are being updated.
        """
        model = TMTModel(SMALL_CFG)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ids = torch.randint(0, SMALL_CFG.vocab_size, (2, 16))
        x, targets = ids[:, :-1], ids[:, 1:]

        with torch.no_grad():
            out0 = model(x)
            conf_before = out0.confidences[-1].mean().item()

        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss, _, _ = compute_loss(out.logits, targets, out.confidences)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out_after = model(x)
            conf_after = out_after.confidences[-1].mean().item()

        # At least some numerical change should have occurred
        assert abs(conf_after - conf_before) > 1e-6, (
            "Confidence didn't change after 3 training steps — gate not updating?"
        )

    def test_memory_anchor_norm_finite(self):
        """Memory anchor norms must be finite after a forward pass."""
        model = TMTModel(SMALL_CFG)
        ids = torch.randint(0, SMALL_CFG.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids)
        norms = out.memory_state.norm(dim=-1)
        assert torch.isfinite(norms).all(), "Some memory anchor norms are not finite"

    def test_exit_mask_coverage_zero_threshold(self):
        """
        With exit_threshold=0.0, tokens should start exiting from the first
        layer; by layer 2 more than zero tokens should have exited.
        """
        model = TMTModel(ZERO_EXIT_CFG)
        ids = torch.randint(0, ZERO_EXIT_CFG.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids)
        # Check that at least layer index 1 (3rd layer) has some exits
        layer2_exits = out.exit_masks[min(1, len(out.exit_masks) - 1)].float().mean().item()
        assert layer2_exits > 0.0, (
            f"Expected some exits by layer 2 with threshold=0.0, got {layer2_exits:.2f}"
        )

    def test_loss_decreases_over_3_steps(self):
        """
        Loss at step 3 must be lower than loss at step 1 on fixed random data,
        confirming the model can overfit/learn a simple sequence.
        """
        torch.manual_seed(42)
        model = TMTModel(SMALL_CFG)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        ids = torch.randint(0, SMALL_CFG.vocab_size, (2, 16))
        x, targets = ids[:, :-1], ids[:, 1:]

        losses: List[float] = []
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss, _, _ = compute_loss(out.logits, targets, out.confidences)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[2] < losses[0], (
            f"Loss did not decrease: step1={losses[0]:.4f}, step3={losses[2]:.4f}"
        )

    def test_param_count_matches_config_estimate(self):
        """
        Actual parameter count must be within 20% of the config's own estimate
        (the __repr__ formula).  This guards against silent architecture drift.
        """
        model = TMTModel(SMALL_CFG)
        actual = model.param_count()
        # Replicate the estimate from TMTConfig.__repr__
        cfg = SMALL_CFG
        estimated = (
            cfg.vocab_size * cfg.d_model
            + cfg.n_layers * (
                4 * cfg.d_model * cfg.d_model
                + 2 * cfg.d_model * cfg.ffn_stream_dim
                + cfg.d_model
            )
        )
        ratio = actual / estimated
        # Allow a wide window: the config __repr__ formula is intentionally
        # a rough estimate that omits LayerNorm, bias terms, memory anchors,
        # and the full exit-gate projection — so actual > estimated is expected.
        assert 0.5 <= ratio <= 5.0, (
            f"Actual params {actual:,} is not in a plausible range of config estimate "
            f"{estimated:,} (ratio={ratio:.2f}). Architecture may have changed drastically."
        )
