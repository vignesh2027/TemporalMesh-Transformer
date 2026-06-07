"""
test_config.py — tests for TMTConfig defaults, repr, validation, and serialisation.

Run: pytest tests/test_config.py -v
"""
import pytest
from dataclasses import asdict

from tmt.model.config import TMTConfig


# ── Default values ────────────────────────────────────────────────────────────

class TestTMTConfigDefaults:
    def test_vocab_size_default(self):
        cfg = TMTConfig()
        assert cfg.vocab_size == 32000

    def test_d_model_default(self):
        cfg = TMTConfig()
        assert cfg.d_model == 512

    def test_n_heads_default(self):
        cfg = TMTConfig()
        assert cfg.n_heads == 8

    def test_n_layers_default(self):
        cfg = TMTConfig()
        assert cfg.n_layers == 12

    def test_max_seq_len_default(self):
        cfg = TMTConfig()
        assert cfg.max_seq_len == 1024

    def test_graph_k_default(self):
        cfg = TMTConfig()
        assert cfg.graph_k == 8

    def test_exit_threshold_default(self):
        cfg = TMTConfig()
        assert cfg.exit_threshold == 0.85

    def test_dual_stream_default(self):
        cfg = TMTConfig()
        assert cfg.dual_stream is True

    def test_memory_anchors_default(self):
        cfg = TMTConfig()
        assert cfg.memory_anchors == 16

    def test_dropout_default(self):
        cfg = TMTConfig()
        assert 0.0 <= cfg.dropout <= 1.0

    def test_decay_rate_default(self):
        cfg = TMTConfig()
        assert cfg.decay_rate > 0.0


# ── Custom values ─────────────────────────────────────────────────────────────

class TestTMTConfigCustom:
    def test_custom_vocab_size(self):
        cfg = TMTConfig(vocab_size=1000)
        assert cfg.vocab_size == 1000

    def test_custom_d_model(self):
        cfg = TMTConfig(d_model=128)
        assert cfg.d_model == 128

    def test_custom_n_heads(self):
        cfg = TMTConfig(n_heads=4)
        assert cfg.n_heads == 4

    def test_custom_graph_k(self):
        cfg = TMTConfig(graph_k=4)
        assert cfg.graph_k == 4

    def test_custom_exit_threshold(self):
        cfg = TMTConfig(exit_threshold=0.5)
        assert cfg.exit_threshold == 0.5

    def test_custom_memory_anchors(self):
        cfg = TMTConfig(memory_anchors=8)
        assert cfg.memory_anchors == 8


# ── Field types ───────────────────────────────────────────────────────────────

class TestTMTConfigTypes:
    def test_vocab_size_is_int(self):
        assert isinstance(TMTConfig().vocab_size, int)

    def test_d_model_is_int(self):
        assert isinstance(TMTConfig().d_model, int)

    def test_n_heads_is_int(self):
        assert isinstance(TMTConfig().n_heads, int)

    def test_n_layers_is_int(self):
        assert isinstance(TMTConfig().n_layers, int)

    def test_exit_threshold_is_float(self):
        assert isinstance(TMTConfig().exit_threshold, float)

    def test_decay_rate_is_float(self):
        assert isinstance(TMTConfig().decay_rate, float)

    def test_dropout_is_float(self):
        assert isinstance(TMTConfig().dropout, float)

    def test_dual_stream_is_bool(self):
        assert isinstance(TMTConfig().dual_stream, bool)


# ── Repr ──────────────────────────────────────────────────────────────────────

class TestTMTConfigRepr:
    def test_repr_contains_vocab(self):
        cfg = TMTConfig(vocab_size=500)
        assert "500" in repr(cfg)

    def test_repr_contains_d_model(self):
        cfg = TMTConfig(d_model=64)
        r = repr(cfg)
        assert "64" in r

    def test_repr_contains_heads(self):
        cfg = TMTConfig(n_heads=4)
        assert "4" in repr(cfg)

    def test_repr_contains_layers(self):
        cfg = TMTConfig(n_layers=6)
        assert "6" in repr(cfg)

    def test_repr_contains_exit_threshold(self):
        cfg = TMTConfig(exit_threshold=0.9)
        assert "0.9" in repr(cfg)

    def test_repr_param_estimate_positive(self):
        cfg = TMTConfig(vocab_size=1000, d_model=64, n_layers=2,
                        ffn_stream_dim=32, n_heads=4)
        r = repr(cfg)
        # Extract the ~params= number
        assert "~params=" in r
        # The number after ~params= should be parseable and > 0
        import re
        m = re.search(r"~params=([\d.]+)M", r)
        assert m is not None
        assert float(m.group(1)) > 0.0

    def test_repr_is_string(self):
        assert isinstance(repr(TMTConfig()), str)


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestTMTConfigSerialisation:
    def test_config_to_dict(self):
        cfg = TMTConfig(vocab_size=500, d_model=64, n_heads=4)
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert d["vocab_size"] == 500
        assert d["d_model"] == 64
        assert d["n_heads"] == 4

    def test_dict_has_all_fields(self):
        cfg = TMTConfig()
        d = asdict(cfg)
        expected_keys = {
            "vocab_size", "max_seq_len", "d_model", "n_heads", "n_layers",
            "graph_k", "decay_rate", "exit_threshold", "dual_stream",
            "ffn_stream_dim", "memory_anchors", "dropout", "layer_norm_eps",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_roundtrip_via_dict(self):
        cfg = TMTConfig(vocab_size=2000, d_model=128, n_heads=8, n_layers=4)
        d = asdict(cfg)
        cfg2 = TMTConfig(**d)
        assert cfg2.vocab_size == cfg.vocab_size
        assert cfg2.d_model == cfg.d_model
        assert cfg2.n_layers == cfg.n_layers

    def test_d_model_divisible_by_n_heads(self):
        cfg = TMTConfig(d_model=64, n_heads=4)
        assert cfg.d_model % cfg.n_heads == 0

    def test_ffn_stream_dim_positive(self):
        cfg = TMTConfig(ffn_stream_dim=256)
        assert cfg.ffn_stream_dim > 0
