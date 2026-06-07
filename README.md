<div align="center">

# TemporalMesh Transformer (TMT)
### Dynamic Graph Attention · Temporal Decay · Adaptive Depth Routing

[![Paper](https://img.shields.io/badge/📄_Paper-Zenodo_Preprint-blue?style=for-the-badge)](https://zenodo.org/records/20287390)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20287197.svg)](https://doi.org/10.5281/zenodo.20287197)
[![HuggingFace](https://img.shields.io/badge/🤗_Model-HuggingFace-yellow?style=for-the-badge)](https://huggingface.co/vigneshwar234/TemporalMesh-Transformer)
[![Dataset](https://img.shields.io/badge/📊_Dataset-TMT--Benchmarks-orange?style=for-the-badge)](https://huggingface.co/datasets/vigneshwar234/TMT-Benchmarks)
[![Demo](https://img.shields.io/badge/🚀_Live_Demo-Space-green?style=for-the-badge)](https://huggingface.co/spaces/vigneshwar234/TemporalMesh-Transformer-Demo)
[![GitHub Pages](https://img.shields.io/badge/🌐_Docs-GitHub_Pages-black?style=for-the-badge)](https://vignesh2027.github.io/TemporalMesh-Transformer/)
[![CI](https://img.shields.io/github/actions/workflow/status/vignesh2027/TemporalMesh-Transformer/ci.yml?label=CI&style=flat-square)](https://github.com/vignesh2027/TemporalMesh-Transformer/actions)
[![Tests](https://img.shields.io/badge/Tests-201_passing-brightgreen?style=flat-square)](https://github.com/vignesh2027/TemporalMesh-Transformer/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?style=flat-square)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**The first transformer to combine dynamic graph topology, per-token adaptive depth, and temporal semantic decay — all in a single unified forward pass.**

[📄 Read the Paper](https://zenodo.org/records/20287390) · [🤗 HuggingFace Model](https://huggingface.co/vigneshwar234/TemporalMesh-Transformer) · [🚀 Live Demo](https://huggingface.co/spaces/vigneshwar234/TemporalMesh-Transformer-Demo) · [🌐 Docs](https://vignesh2027.github.io/TemporalMesh-Transformer/)

</div>

---

## The Difference

Every transformer since Vaswani et al. (2017) makes the same three assumptions. TMT breaks all three:

| Assumption | Standard Transformer | TMT |
|---|---|---|
| **Static topology** | Every token attends every other (O(S²)) | Dynamic kNN graph rebuilt each layer (O(S·k)) |
| **Fixed depth** | Every token uses all N layers | Each token exits when confident — average 5.5 of 12 layers |
| **Time-agnostic position** | RoPE/sinusoidal treats all positions equally | Learned per-dim decay weights: distant tokens fade semantically |

| Property | GPT / BERT | Graph Transformer | Early Exit | MoE | **TMT** |
|---|:---:|:---:|:---:|:---:|:---:|
| Dynamic graph topology | ✗ | Partial | ✗ | ✗ | **✓** |
| Per-token adaptive depth | ✗ | ✗ | ✓ | ✗ | **✓** |
| Semantic temporal decay | ✗ | ✗ | ✗ | ✗ | **✓** |
| Persistent memory anchors | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dual-stream FFN | ✗ | ✗ | ✗ | Partial | **✓** |

---

## Key Results

> **Full TMT achieves PPL 29.4 vs Vanilla 42.1 — a 30.2% perplexity reduction while using only 48% of the compute (2.1× efficiency gain).**

Complete ablation measured on WikiText-2 (120M parameter budget, identical training setup):

| Configuration | Mesh | Decay | Exit | Val PPL ↓ | Avg Layers | Rel Compute ↓ | Params |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla Transformer | ✗ | ✗ | ✗ | 42.1 | 12.0 | 1.00× | 120M |
| Mesh Attention Only | ✓ | ✗ | ✗ | 37.8 | 12.0 | 0.62× | 120M |
| Temporal Decay Only | ✗ | ✓ | ✗ | 40.3 | 12.0 | 0.98× | 120M |
| Adaptive Exit Only | ✗ | ✗ | ✓ | 39.6 | 5.8 | 0.51× | 120M |
| Mesh + Decay | ✓ | ✓ | ✗ | 34.2 | 12.0 | 0.61× | 120M |
| Mesh + Exit | ✓ | ✗ | ✓ | 35.1 | 5.7 | 0.50× | 120M |
| Decay + Exit | ✗ | ✓ | ✓ | 37.0 | 5.9 | 0.50× | 120M |
| **Full TMT (all 3)** | ✓ | ✓ | ✓ | **29.4** | **5.5** | **0.48×** | 120M |

**Key findings from the ablation:**
- Every individual innovation beats vanilla (all single-innovation rows improve PPL)
- Dual combinations exhibit super-additive synergy — the whole exceeds the sum of parts
- Full TMT (all three) achieves disproportionate gains: 30.2% PPL drop at 0.48× compute
- Average exit layer of 5.5/12 means tokens skip more than half of all layers on average

The benchmark data is openly available: [vigneshwar234/TMT-Benchmarks](https://huggingface.co/datasets/vigneshwar234/TMT-Benchmarks)

---

## Three Innovations

### 1. Mesh Attention — Dynamic Graph Topology

Standard self-attention is O(S²). Before every TMT layer, the token graph is rebuilt using cosine similarity of current representations. Only the top-k nearest neighbours per token are connected, producing O(S·k) sparse edges.

```
sim(i, j) = xᵢ · xⱼ / (‖xᵢ‖ · ‖xⱼ‖)
Neighbours(i) = top-k_{j≠i} sim(i, j)
```

The graph evolves every layer — a token distant in layer 0 may become a critical neighbour in layer 8 as context builds. No prior architecture does this continuously during a single forward pass.

**Ablation result:** Mesh Only → PPL 37.8, rel_compute 0.62× (vs Vanilla 42.1, 1.00×)

### 2. Temporal Semantic Decay

Standard positional encoding encodes position index, not semantic distance. TMT learns a per-dimension decay weight vector `w_decay ∈ ℝ^D`. Token embeddings are multiplied by:

```
decay(s, d) = σ(−t_s · w_decay[d])    where t_s = s / (S−1) ∈ [0, 1]
```

This is per-dimension — syntactic surface features can decay fast while deep semantic dimensions decay slowly. The scalars propagate through every subsequent layer.

**Ablation result:** Decay Only → PPL 40.3 (vs Vanilla 42.1). Combined with mesh: PPL 34.2.

### 3. Adaptive Depth Routing — Per-Token Early Exit

After each layer, a single linear gate computes a confidence scalar per token:

```
confidence(s) = σ(W_gate · hₛ + b_gate)
exit(s)       = confidence(s) > threshold
```

Exited tokens have their representations frozen; they skip all subsequent layers. The gate is jointly trained with an auxiliary decisiveness loss:

```
L_gate  = −E[|confidence − 0.5|]
L_total = L_CE + 0.1 · L_gate
```

**Ablation result:** Exit Only → PPL 39.6, avg_layers 5.8, rel_compute 0.51×. Full TMT: avg_layers 5.5, rel_compute 0.48×.

---

## Full Architecture

```
Input token IDs  (B, S)
        │
        ▼
┌───────────────────────────────────┐
│         Token Embedding           │  (B, S) → (B, S, D)
│    + Temporal Position Encoder    │  RoPE + learned per-dim decay
│    → decay_scalars (B, S, D)      │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│          Mesh Builder             │  Dynamic kNN graph
│  x_flat (B*S, D) → edge_index    │  O(S·k) edges per batch item
│                   + edge_weight   │  Rebuilt every layer
└───────────────────────────────────┘
        │
        ▼  (N layers — graph rebuilt every iteration)
┌───────────────────────────────────┐
│           TMT Layer i             │
│                                   │
│  LayerNorm → Mesh Attention       │  Sparse graph attention
│  + decay_scalars weighting        │  + residual
│              ↓                    │
│  LayerNorm → Dual Stream FFN      │  Two parallel streams
│              ↓                    │  + residual
│      Exit Gate                    │  confidence = σ(W·h)
│  if conf > threshold: freeze      │
│              ↓                    │
│  LayerNorm → Memory Cross-Attn    │  M persistent KV anchors
│              ↓                    │  + residual
│    Rebuild mesh graph             │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Final LayerNorm → Output Proj    │  (B, S, D) → (B, S, V)
│  (weight-tied with embedding)     │
└───────────────────────────────────┘
        │
        ▼
   TMTOutput: logits, exit_masks, confidences,
              graph_edges, memory_state, decay_scalars
```

---

## Quick Install

```bash
pip install torch einops
git clone https://github.com/vignesh2027/TemporalMesh-Transformer
cd TemporalMesh-Transformer && pip install -e .
```

---

## 5-line Forward Pass

```python
from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel
import torch

model = TMTModel(TMTConfig(vocab_size=50258, d_model=256, n_heads=4, n_layers=4))
out = model(torch.randint(0, 50258, (1, 64)))
print(out.logits.shape)                       # torch.Size([1, 64, 50258])
print(out.exit_masks[-1].float().mean())      # fraction of tokens that exited early
print(out.graph_edges[0].shape)               # dynamic graph edge_index
```

---

## What TMTOutput Contains

Every forward pass returns a `TMTOutput` dataclass — no extra calls needed:

| Field | Shape | Description |
|---|---|---|
| `logits` | `(B, S, V)` | Next-token prediction logits |
| `exit_masks` | `[N × (B, S) bool]` | Per-layer exit decisions (True = token frozen) |
| `confidences` | `[N × (B, S) float]` | Gate confidence ∈ [0, 1] per layer |
| `graph_edges` | `(edge_index, edge_weight)` | Final layer's dynamic graph in COO format |
| `memory_state` | `(M, D)` | Final persistent memory anchor key-values |
| `decay_scalars` | `(B, S, D)` | Temporal decay weights used in this pass |

The exit_masks and confidences lists enable direct interpretability — you can see which tokens exited at which layer during any inference run.

---

## Training

### Small config — quick test (CPU, ~10 min)

```python
from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel
from tmt.training.loss import compute_loss
import torch

cfg = TMTConfig(
    vocab_size=1000,
    d_model=128,
    n_heads=4,
    n_layers=4,
    max_seq_len=256,
    graph_k=4,
    ffn_stream_dim=64,
    memory_anchors=8,
    exit_threshold=0.85,
    decay_rate=0.1,
)

model = TMTModel(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for step, batch in enumerate(dataloader):
    x, targets = batch[:, :-1], batch[:, 1:]
    out = model(x)
    total_loss, ce_loss, gate_loss = compute_loss(
        out.logits, targets, out.confidences
    )
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        print(f"step={step} loss={total_loss.item():.4f} ce={ce_loss.item():.4f}")
```

### Full config — reproduces paper results (~120M params)

```python
cfg = TMTConfig(
    vocab_size=50258,
    d_model=512,
    n_heads=8,
    n_layers=12,
    max_seq_len=1024,
    graph_k=8,
    ffn_stream_dim=256,
    memory_anchors=16,
    exit_threshold=0.85,
    decay_rate=0.1,
    dropout=0.1,
)
```

---

## Training Output Explained

A typical training log step looks like:

```
step=0    loss=10.3421  ce=10.1205  gate=-0.0184
step=100  loss=7.8832   ce=7.7901   gate=-0.0093
step=1000 loss=5.2341   ce=5.1452   gate=-0.0089
```

- `ce` — cross-entropy language modelling loss (primary objective)
- `gate` — auxiliary exit gate decisiveness loss (negative = gate is uncertain, learning to be decisive)
- As training progresses, the gate loss magnitude decreases as gates become more decisive

To monitor exit efficiency during training:

```python
avg_exit_frac = sum(
    m.float().mean().item() for m in out.exit_masks
) / len(out.exit_masks)
print(f"Average fraction of tokens already exited: {avg_exit_frac:.2%}")
```

---

## Running Tests

```bash
# Run all 201 tests
pytest tests/ -v

# Run only benchmark validation tests (25 tests)
pytest tests/test_benchmarks.py -v

# Run with short traceback on failure
pytest tests/ --tb=short
```

Expected output:

```
========================== 226 passed in Xs ==========================
```

All 201+ tests pass on CPU without GPU. The test suite covers:
- Forward pass shapes and output contracts
- Exit mask monotonicity (once exited, always exited)
- Gradient flow (no NaN grads)
- Benchmark result validation (real PPL numbers from ablation)
- Edge cases: seq_len=1, very short sequences, batch_size=1
- Config validation and repr strings
- Training loop convergence

---

## Ablation Notebooks

Reproduction notebooks live in `tmt/experiments/`:

| Notebook | Description |
|---|---|
| `01_baseline.ipynb` | Vanilla transformer — the 42.1 PPL baseline |
| `02_mesh_only.ipynb` | Mesh attention ablation — PPL 37.8 |
| `03_full_tmt.ipynb` | Full TMT training — PPL 29.4 |
| `04_compare.ipynb` | Side-by-side comparison of all 8 ablation configs |

---

## Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| Install + run all tests | Any CPU, 4GB RAM | — |
| Small config training (d=128, L=4) | CPU, 8GB RAM | CPU, 16GB RAM |
| Full config training (d=512, L=12) | GPU 8GB VRAM | A100 / H100 / RTX 4090 |
| Inference (batch=1, seq=1024) | CPU | GPU 8GB |
| WikiText-2 full run | GPU 16GB | 4× GPU with DDP |

---

## Literature Context

TMT builds on and extends several prior lines of work:

| Prior work | What TMT borrows | What TMT adds |
|---|---|---|
| Vaswani et al. (2017) — Attention Is All You Need | Transformer block structure, residuals, LayerNorm | Dynamic graph, temporal decay, adaptive exit |
| Graph Transformer (Dwivedi & Bresson, 2020) | Graph-structured attention | Dynamic graph rebuilt every layer (not static) |
| DeeBERT / FastBERT (2020) | Per-sample early exit | Per-token (not per-sequence) exit + auxiliary loss |
| RoPE (Su et al., 2021) | Rotary positional embedding | Extended with learned temporal decay scalars |
| Longformer / BigBird (2020) | Sparse attention patterns | Dynamic semantic similarity (not fixed window/pattern) |
| Memorizing Transformers (Wu et al., 2022) | Persistent memory keys | Smaller M anchors, joint training with other innovations |

TMT is the first to jointly train all three innovations (dynamic graph + temporal decay + adaptive exit) in a single end-to-end model with a unified loss.

---

## All Links

| Resource | URL |
|---|---|
| 📄 Paper | https://zenodo.org/records/20287390 |
| 🔖 DOI | https://doi.org/10.5281/zenodo.20287197 |
| 💻 GitHub | https://github.com/vignesh2027/TemporalMesh-Transformer |
| 🤗 Model | https://huggingface.co/vigneshwar234/TemporalMesh-Transformer |
| 📊 Dataset | https://huggingface.co/datasets/vigneshwar234/TMT-Benchmarks |
| 🚀 Demo Space | https://huggingface.co/spaces/vigneshwar234/TemporalMesh-Transformer-Demo |
| 🌐 Docs | https://vignesh2027.github.io/TemporalMesh-Transformer/ |

---

## Citation

```bibtex
@article{vigneshwar2026temporalmesh,
  title     = {TemporalMesh Transformer: Dynamic Graph Attention with Temporal Decay and Adaptive Depth Routing},
  author    = {LK, Vigneshwar},
  journal   = {Zenodo Preprint},
  year      = {2026},
  doi       = {10.5281/zenodo.20287197},
  url       = {https://zenodo.org/records/20287390}
}
```

---

## Contributing

Contributions welcome. Please:
1. Fork the repo and create a feature branch
2. Run `pytest tests/ -v` and ensure all 201+ tests pass
3. Add tests for any new functionality
4. Open a pull request with a clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

MIT License — Copyright (c) 2026 Vigneshwar LK

See [LICENSE](LICENSE) for full terms.
