# Contributing to TemporalMesh Transformer

## Setup

```bash
git clone https://github.com/vignesh2027/TemporalMesh-Transformer.git
cd TemporalMesh-Transformer
python3 -m venv .venv && source .venv/bin/activate

# Minimal test install (CPU-only torch, fast)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops pytest pytest-cov ruff
pip install -e . --no-deps
```

## Before Opening a PR

```bash
# All 15 tests must pass
pytest tests/ -v

# No lint errors
ruff check tmt/ tests/
```

## Adding Tests

- Shape test for a new module → `tests/test_shapes.py`
- End-to-end / invariant test → `tests/test_forward.py`
- Use a small config (`d_model=64, n_heads=4`) so tests stay fast on CPU

## Project Map

```
tmt/model/      ← Architecture (config, embedding, mesh, attention, ffn, exit_gate, memory, layers, model)
tmt/training/   ← Trainer, loss, LR scheduler
tmt/data/       ← Dataset loader, tokenizer wrapper
tests/          ← pytest test suite
docs/           ← GitHub Pages site
paper/          ← Figures + PDF
```

## CI

Every PR triggers GitHub Actions on Python 3.10, 3.11, and 3.12 (Ubuntu, CPU torch).
The `lint` job runs `ruff check` separately. Both must be green to merge.
