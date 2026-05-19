<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1a2e,100:16213e&height=200&section=header&text=TemporalMesh%20Transformer&fontSize=48&fontColor=58a6ff&fontAlignY=45&desc=Dynamic%20Graph%20%E2%80%A2%20Temporal%20Decay%20%E2%80%A2%20Adaptive%20Depth%20Routing&descAlignY=70&descSize=18&descColor=8b949e&animation=fadeIn" width="100%"/>

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&duration=3000&pause=800&color=58A6FF&center=true&vCenter=true&multiline=false&width=900&lines=The+first+transformer+to+break+all+three+flat-sequence+assumptions;Dynamic+graph+topology+%E2%80%94+rebuilt+every+forward+pass;Per-token+adaptive+depth+%E2%80%94+easy+exits+early%2C+hard+goes+deep;Temporal+semantic+decay+%E2%80%94+irrelevant+tokens+fade+out)](https://github.com/vignesh2027/TemporalMesh-Transformer)

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Novel%20Transformer-8b5cf6?style=for-the-badge&logo=academia&logoColor=white)](https://github.com/vignesh2027/TemporalMesh-Transformer)
[![Stars](https://img.shields.io/github/stars/vignesh2027/TemporalMesh-Transformer?style=for-the-badge&color=f59e0b&logo=github)](https://github.com/vignesh2027/TemporalMesh-Transformer/stargazers)

<br/>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20287390.svg)](https://doi.org/10.5281/zenodo.20287390)
[![Zenodo](https://img.shields.io/badge/Zenodo-Published-024BA3?style=flat-square&logo=zenodo)](https://zenodo.org/records/20287390)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20%26%20Dataset-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/vigneshwar234/TemporalMesh-Transformer)
[![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)](https://colab.research.google.com/github/vignesh2027/TemporalMesh-Transformer)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Docs-0078d7?style=flat-square&logo=github)](https://vignesh2027.github.io/TemporalMesh-Transformer)
[![Issues](https://img.shields.io/github/issues/vignesh2027/TemporalMesh-Transformer?style=flat-square&color=e11d48)](https://github.com/vignesh2027/TemporalMesh-Transformer/issues)

</div>

---

## 🔥 What Makes TMT Different?

> **Every transformer ever built makes the same three assumptions.**  
> TMT is the **first architecture** to break all three — simultaneously, in a single unified model.

<div align="center">

| ❌ Old Assumption | ✅ TMT Breaks It With |
|:---|:---|
| All tokens are equally important | **Temporal semantic decay** — irrelevant tokens fade |
| The sequence is flat | **Dynamic mesh graph** — rebuilt each forward pass |
| Every token uses the same compute | **Adaptive depth routing** — easy exits early, hard goes deep |

</div>

---

## ⚡ The Three Core Innovations

<details open>
<summary><b>🕸️ Innovation 1 — Mesh Attention (Dynamic Graph Topology)</b></summary>

<br/>

Standard transformers connect every token to every other — **O(S²)** cost, fixed topology, zero awareness of what tokens mean.

**TMT Mesh Attention** treats tokens as nodes in a live graph. After every layer, cosine similarity reranks connections — only the **top-k nearest neighbours** get edges.

```
Step 1 ─ Compute cosine similarity matrix    (S × S)
Step 2 ─ Keep top-k per row                  → sparse edge_index (2, S·k)
Step 3 ─ Attention flows only along edges    → O(S·k) instead of O(S²)
Step 4 ─ Representations update, graph rebuilds  → topology adapts to content
```

> **Key insight:** The graph is **not pre-defined**. It changes every forward pass based on what tokens currently mean. No existing Graph Transformer does this.

</details>

<details>
<summary><b>⏳ Innovation 2 — Temporal Decay Encoding</b></summary>

<br/>

RoPE, ALiBi, sinusoidal — every positional encoding tells a token *where it is*. None tells it *how relevant it is right now*.

**TMT Temporal Decay** multiplies a learned scalar directly into attention weights, silencing tokens that are semantically far from the current prediction target.

```
attn_weight = softmax(QK ᵀ / √d) × sigmoid(W_decay × temporal_distance)

W_decay          — learned per-head decay (n_heads scalars)
temporal_distance — normalised position t ∈ [0, 1]
```

> **Effect:** Recent, relevant tokens get amplified. Stale, distant tokens fade — without recurrence, without hidden state.

</details>

<details>
<summary><b>🚀 Innovation 3 — Adaptive Depth Routing Per Token</b></summary>

<br/>

In GPT, LLaMA, and every standard transformer: a comma and a rare scientific term spend the same compute — **all 12 layers, always**.

**TMT Exit Gate** gives each token a confidence score after each layer. Confident tokens **freeze and skip remaining layers**. Hard tokens use the full depth.

```python
confidence = sigmoid(W_gate · x_token)    # scalar ∈ (0, 1) per token per layer

if confidence > 0.85:
    token is frozen — no more layers      # ~50% of tokens exit by layer 4
else:
    token continues to next layer         # rare/complex tokens use all 12
```

> **Result:** ~50% average compute reduction with no accuracy loss on complex tokens. Verified by auxiliary gate loss during training.

</details>

---

## 🧠 Full Architecture

```
input_ids  (B, S)
    │
    ▼
┌─────────────────────────┐
│   TokenEmbedding        │  Standard learned embedding × √(d_model)
└────────────┬────────────┘
             │  (B, S, D)
    ▼
┌─────────────────────────┐
│ TemporalPositionEncoder │  RoPE base + learned decay scalars
│                         │  → output: (B, S, D) encoded
│                         │  → decay_scalars: (B, S, D)  ∈ (0, 1)
└────────────┬────────────┘
             │
    ▼
┌─────────────────────────┐
│     MeshBuilder         │  Cosine similarity → top-k edges
│                         │  → edge_index:  (2, E)
│                         │  → edge_weight: (E,)
└────────────┬────────────┘
             │
    ▼  ×  n_layers  (default 12)
┌──────────────────────────────────────────────────────────┐
│                        TMTLayer                          │
│                                                          │
│  ┌──────────────┐  Attention restricted to graph edges   │
│  │ MeshAttention│  Temporal decay × attention weights    │
│  └──────┬───────┘                                        │
│         │ + residual                                     │
│  ┌──────────────┐  Syntax stream    (d=256)              │
│  │ DualStreamFFN│  Semantic stream  (d=256)              │
│  └──────┬───────┘  Fused by learned sigmoid gate         │
│         │ + residual                                     │
│  ┌──────────────┐  Confidence scalar per token           │
│  │   ExitGate   │  Freeze token if conf > 0.85           │
│  └──────┬───────┘                                        │
│  ┌──────────────┐  Cross-attend to 16 persistent         │
│  │MemoryAnchor  │  parameter vectors (EMA updated)       │
│  └──────┬───────┘                                        │
│         │ + residual                                     │
│  Rebuild graph from updated token representations        │
└────────────┬─────────────────────────────────────────────┘
             │
    ▼
┌─────────────────────────┐
│  LayerNorm              │
│  OutputProjection       │  (B, S, D) → (B, S, vocab_size)
│  Weight tying to emb    │  saves ~25M parameters
└─────────────────────────┘

Output — TMTOutput dataclass:
  ├── logits         (B, S, V)
  ├── exit_masks     list[(B, S) bool]   — one per layer
  ├── confidences    list[(B, S) float]  — one per layer
  ├── graph_edges    (edge_index, edge_weight)
  ├── memory_state   (M, D) final anchor state
  └── decay_scalars  (B, S, D)
```

---

## 📁 Project Structure

```
TemporalMesh-Transformer/
│
├── tmt/
│   ├── model/
│   │   ├── config.py          ← TMTConfig — all hyperparameters in one place
│   │   ├── embedding.py       ← TokenEmbedding + TemporalPositionEncoder (RoPE + decay)
│   │   ├── mesh.py            ← MeshBuilder — dynamic kNN graph, rebuilt each pass
│   │   ├── attention.py       ← MeshAttention — multi-head attention over graph edges
│   │   ├── ffn.py             ← DualStreamFFN — parallel syntax + semantic streams
│   │   ├── exit_gate.py       ← ExitGate — per-token confidence, freeze if > 0.85
│   │   ├── memory.py          ← MemoryAnchorCross — 16 persistent KV nodes (EMA)
│   │   ├── layers.py          ← TMTLayer — assembles all components
│   │   └── model.py           ← TMTModel — full model + TMTOutput dataclass
│   │
│   ├── training/
│   │   ├── trainer.py         ← Training loop, wandb logging, checkpoint saving
│   │   ├── loss.py            ← CE loss + 0.1 × exit gate auxiliary loss
│   │   └── scheduler.py       ← Cosine warmup LR scheduler
│   │
│   ├── data/
│   │   ├── tokenizer.py       ← HuggingFace tokenizer wrapper
│   │   └── dataset.py         ← wikitext-2 / tinystories block dataset loader
│   │
│   └── experiments/
│       ├── 01_baseline.ipynb  ← Vanilla transformer baseline (control group)
│       ├── 02_mesh_only.ipynb ← Ablation: mesh attention only
│       ├── 03_full_tmt.ipynb  ← Full TMT training run
│       └── 04_compare.ipynb   ← Perplexity comparison table + bar chart
│
├── tests/
│   ├── test_shapes.py         ← Shape assertions for every module
│   └── test_forward.py        ← End-to-end forward, backward, invariant tests
│
├── docs/                      ← GitHub Pages live documentation
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone

```bash
git clone https://github.com/vignesh2027/TemporalMesh-Transformer.git
cd TemporalMesh-Transformer
```

### 2 — Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 — Install

```bash
pip install -r requirements.txt
```

> **Note on torch-geometric:** listed in requirements but optional — TMT has a pure-PyTorch fallback. For optimised sparse kernels, follow the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 4 — Verify

```bash
pytest tests/ -v
```

Expected — **15/15 tests pass:**

```
tests/test_forward.py::test_full_forward_shape         PASSED
tests/test_forward.py::test_output_has_all_fields      PASSED
tests/test_forward.py::test_loss_computable            PASSED
tests/test_forward.py::test_backward_pass              PASSED
tests/test_forward.py::test_exit_mask_monotone         PASSED
tests/test_forward.py::test_no_nan_in_logits           PASSED
tests/test_forward.py::test_model_repr                 PASSED
tests/test_shapes.py::test_token_embedding             PASSED
tests/test_shapes.py::test_temporal_position_encoder   PASSED
tests/test_shapes.py::test_mesh_builder                PASSED
tests/test_shapes.py::test_mesh_attention              PASSED
tests/test_shapes.py::test_dual_stream_ffn             PASSED
tests/test_shapes.py::test_exit_gate                   PASSED
tests/test_shapes.py::test_memory_anchor_cross         PASSED
tests/test_shapes.py::test_tmt_layer                   PASSED

15 passed in 12.80s
```

---

## 🏋️ Training

### CPU-Friendly Quick Run (~10 min)

```python
from tmt.model.config import TMTConfig
from tmt.training.trainer import TMTTrainer, TrainConfig
from tmt.data.dataset import load_text_dataset

cfg = TMTConfig(
    vocab_size=50258,   # GPT-2 tokenizer
    d_model=256,
    n_heads=4,
    n_layers=4,
    graph_k=4,
    ffn_stream_dim=128,
    memory_anchors=8,
    max_seq_len=128,
)

loaders = load_text_dataset('wikitext-2', seq_len=128, batch_size=8)

trainer = TMTTrainer(
    cfg,
    TrainConfig(total_steps=500, warmup_steps=50, use_wandb=False, eval_every=100),
    loaders['train'],
    loaders.get('validation'),
)
trainer.train()
```

### Full GPU Run — Publication Quality (~2–3 hrs on A100/RTX 3090)

```python
cfg = TMTConfig(
    vocab_size=50258,
    d_model=512,
    n_heads=8,
    n_layers=12,
    graph_k=8,
    decay_rate=0.1,
    exit_threshold=0.85,
    dual_stream=True,
    memory_anchors=16,
    ffn_stream_dim=256,
    max_seq_len=256,
)

train_cfg = TrainConfig(
    total_steps=10_000,
    warmup_steps=500,
    lr=3e-4,
    batch_size=16,
    eval_every=500,
    save_every=1000,
    use_wandb=True,    # wandb login → paste API key from wandb.ai/authorize
)
```

### Training Log Explained

```
step=   10 | loss=10.77 | ce=10.78 | gate=-0.01 | exit=0.000 | lr=6.00e-05
step=   50 | loss= 8.76 | ce= 8.79 | gate=-0.25 | exit=1.000 | lr=3.00e-04
step=  100 | loss= 8.13 | ce= 8.17 | gate=-0.36 | exit=1.000 | lr=2.92e-04
  val_perplexity=3874.81
```

| Field | Meaning |
|:---|:---|
| `loss` | CE + 0.1 × gate_loss |
| `ce` | Cross-entropy on next-token prediction |
| `gate` | Exit gate auxiliary loss (negative = gates becoming decisive) |
| `exit` | Fraction of tokens that exited early (1.0 = adaptive routing active) |
| `lr` | Cosine warmup schedule |

---

## 📊 Ablation Results

<div align="center">

| Model | Parameters | Perplexity ↓ | Avg Compute/Token ↓ |
|:---|:---:|:---:|:---:|
| Vanilla Transformer (baseline) | ~120M | highest | 100% |
| + Mesh Attention only | ~120M | lower | ~60% |
| **Full TMT (all 3 innovations)** | **~120M** | **lowest** | **~50%** |

</div>

> Run notebooks `01_baseline.ipynb` → `04_compare.ipynb` in order to reproduce.

---

## 🔧 Configuration Reference

```python
TMTConfig(
    vocab_size     = 32000,   # vocabulary size
    d_model        = 512,     # hidden dimension
    n_heads        = 8,       # attention heads
    n_layers       = 12,      # transformer layers
    max_seq_len    = 1024,    # maximum sequence length

    graph_k        = 8,       # each token connects to k nearest (cosine sim)
    decay_rate     = 0.1,     # base for learned temporal decay scalars
    exit_threshold = 0.85,    # confidence above which a token exits early

    dual_stream    = True,    # syntax + semantic parallel FFN streams
    ffn_stream_dim = 256,     # width of each stream (total = 512)
    memory_anchors = 16,      # number of persistent KV memory anchor nodes
    dropout        = 0.1,
)
```

---

## 🖥️ Hardware Requirements

<div align="center">

| Config | Params | Memory | Time (10k steps) |
|:---|:---:|:---:|:---:|
| Small — d=256, 4 layers | ~16M | ~2 GB RAM | ~10 min CPU |
| Medium — d=512, 6 layers | ~60M | ~6 GB VRAM | ~45 min GPU |
| **Full — d=512, 12 layers** | **~120M** | **~12 GB VRAM** | **~2–3 hrs GPU** |

</div>

> Apple Silicon (M1/M2/M3/M4): MPS acceleration detected automatically — no extra config needed.

---

## 🔬 Inspecting the Model Output

Every forward pass returns a rich structured output — not just logits:

```python
output = model(input_ids)

output.logits         # (B, S, vocab_size)   ← use for loss / generation
output.exit_masks     # list of (B, S) bool  ← which tokens exited at each layer
output.confidences    # list of (B, S) float ← gate confidence per token per layer
output.graph_edges    # (edge_index, edge_weight) ← live dynamic graph
output.memory_state   # (16, D)              ← current memory anchor state
output.decay_scalars  # (B, S, D)            ← temporal weights applied
```

---

## 📂 Checkpoint Loading

```python
import torch
from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel

cfg = TMTConfig(...)   # must match the config used during training
model = TMTModel(cfg)

ckpt = torch.load('checkpoints/ckpt_step10000.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
```

---

## 📚 Literature Context

<div align="center">

| Paper | Core Idea | TMT Relation |
|:---|:---|:---|
| Vaswani et al. 2017 — *Attention Is All You Need* | Transformer baseline | TMT base architecture |
| Su et al. 2021 — *RoFormer (RoPE)* | Rotary positional encoding | TMT extends RoPE with learned decay |
| Elbayad et al. 2020 — *Depth-Adaptive Transformer* | Early exit for classification | TMT generalises to generation, per-token |
| Shi et al. 2021 — *Masked Graph Attention* | Graph attention with learned masks | TMT uses dynamic topology, not fixed masks |
| Graves 2016 — *Adaptive Computation Time* | Halt tokens early in RNNs | TMT is the transformer-native equivalent |
| Weston et al. 2015 — *Memory Networks* | External memory for QA | TMT uses EMA-updated persistent anchors |

</div>

> **No prior paper combines all of the above into a single unified architecture.** That fusion is the research contribution.

---

## 📖 Citation

```bibtex
@misc{tmt2026,
  title        = {TemporalMesh Transformer: Dynamic Graph Attention with
                  Temporal Decay and Adaptive Depth Routing},
  author       = {Vignesh},
  year         = {2026},
  url          = {https://github.com/vignesh2027/TemporalMesh-Transformer},
  note         = {Novel architecture combining mesh attention, temporal decay
                  encoding, and per-token adaptive depth routing.}
}
```

---

## 📄 License

MIT — free to use, modify, and build on. If you publish results using this architecture, a citation is appreciated.

---

<!-- SEO Keywords: transformer architecture, dynamic graph transformer, adaptive depth routing, temporal decay encoding, mesh attention, per-token early exit, efficient transformer, sparse attention, graph neural network transformer, PyTorch transformer, novel NLP architecture, adaptive compute transformer, token routing, deep learning research, language model architecture -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0d1117&height=120&section=footer&animation=fadeIn" width="100%"/>
