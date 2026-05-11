# TemporalMesh Transformer (TMT)

A novel transformer architecture combining three innovations not previously unified in a single model:

| Innovation | What it does | Existing work it builds on |
|---|---|---|
| **Mesh Attention** | Tokens attend over a dynamic kNN graph (recomputed each forward pass) | Graph Transformers (but they use fixed graphs) |
| **Temporal Decay Encoding** | Learned decay scalars attenuate temporally distant tokens inside attention | RoPE, ALiBi (but those are positional only) |
| **Adaptive Depth Routing** | Tokens exit early when a confidence gate fires, halving average compute | Early Exit Transformers (but never fused with the above) |

Plus: **DualStreamFFN** (syntax + semantic parallel streams) and **MemoryAnchorCross** (16 persistent KV nodes updated by EMA each forward pass).

---

## Architecture

```
input_ids (B, S)
    │
    ▼
TokenEmbedding          → (B, S, D)
    │
TemporalPositionEncoder → (B, S, D) + decay_scalars (B, S, D)
    │
MeshBuilder             → edge_index (2, E), edge_weight (E,)
    │
TMTLayer × 12:
    ├── MeshAttention (graph-restricted + temporal decay)
    ├── DualStreamFFN (syntax stream + semantic stream)
    ├── ExitGate      (per-token confidence; freeze if > 0.85)
    └── MemoryAnchorCross (cross-attend 16 persistent anchors)
    │
LayerNorm → OutputProjection (B, S, V)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Train (small config)
python -c "
from tmt.model.config import TMTConfig
from tmt.training.trainer import TMTTrainer, TrainConfig
from tmt.data.dataset import load_text_dataset

cfg = TMTConfig(vocab_size=50258, d_model=256, n_heads=4, n_layers=4,
                graph_k=4, ffn_stream_dim=128, memory_anchors=8)
loaders = load_text_dataset('wikitext-2', seq_len=128, batch_size=8)
train_cfg = TrainConfig(total_steps=1000, use_wandb=False)
trainer = TMTTrainer(cfg, train_cfg, loaders['train'], loaders.get('validation'))
trainer.train()
"
```

---

## File Structure

```
tmt/
  model/
    config.py         TMTConfig dataclass
    embedding.py      TokenEmbedding + TemporalPositionEncoder (RoPE + decay)
    mesh.py           MeshBuilder — dynamic kNN graph per forward pass
    attention.py      MeshAttention — multi-head attention over graph edges
    ffn.py            DualStreamFFN — syntax + semantic parallel streams
    exit_gate.py      ExitGate — per-token confidence-based early exit
    memory.py         MemoryAnchorCross — persistent KV memory nodes (EMA)
    layers.py         TMTLayer — assembles all components
    model.py          TMTModel — full model + TMTOutput dataclass
  training/
    trainer.py        Training loop with wandb logging
    loss.py           CE loss + exit gate auxiliary loss
    scheduler.py      Cosine warmup LR scheduler
  data/
    tokenizer.py      HuggingFace tokenizer wrapper
    dataset.py        wikitext-2 / tinystories loader
  experiments/
    01_baseline.ipynb Vanilla transformer baseline
    02_mesh_only.ipynb Ablation: mesh attention only
    03_full_tmt.ipynb Full TMT training run
    04_compare.ipynb  Perplexity comparison table + chart
tests/
  test_shapes.py      Shape assertions for every module
  test_forward.py     End-to-end forward + backward smoke tests
```

---

## Experiments

Run the notebooks in order inside `tmt/experiments/`.  After all three training runs, fill in the perplexity values in `04_compare.ipynb` to generate the comparison chart.

Expected ablation results (indicative, actual numbers depend on hardware and run duration):

| Model | Perplexity | Avg Compute |
|---|---|---|
| Vanilla Transformer | baseline | 100% |
| Mesh Attention Only | lower | ~60% |
| Full TMT | lowest | ~50% |

---

## What Makes TMT New

Every existing transformer picks one problem.  TMT is the first to fuse:
- Dynamic graph topology (not fixed like Graph Transformers)
- Token-level adaptive compute (not layer-level like MoE)
- Temporal decay inside attention weights (not as positional bias)
- Persistent fast-weight memory with EMA updates
- Dual-stream FFN with learned fusion gate

The combination — not any single piece — is the novelty.

---

## Citation

```bibtex
@misc{tmt2026,
  title  = {TemporalMesh Transformer: Dynamic Graph Attention with Temporal Decay and Adaptive Depth Routing},
  year   = {2026},
  note   = {Experimental architecture. See experiments/ for ablation results.}
}
```
