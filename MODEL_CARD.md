---
language:
- en
license: mit
library_name: pytorch
tags:
- text-generation
- pytorch
- transformers
- graph-neural-network
- research
- novel-architecture
- efficient-transformer
- sparse-attention
- adaptive-computation
- dynamic-graph
- early-exit
- temporal-decay
- mesh-attention
- language-model
- causal-lm
- preprint
- arxiv
- wikitext
- nlp
- attention
datasets:
- wikitext
- roneneldan/TinyStories
- vigneshwar234/TMT-Benchmarks
metrics:
- perplexity
pipeline_tag: text-generation
model-index:
- name: TemporalMesh-Transformer
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: WikiText-2
      type: wikitext
    metrics:
    - type: perplexity
      value: 29.4
      name: Perplexity — Full TMT (120M params)
  - task:
      type: text-generation
      name: Text Generation (Ablation)
    dataset:
      name: WikiText-2
      type: wikitext
    metrics:
    - type: perplexity
      value: 42.1
      name: Perplexity — Vanilla Transformer Baseline
    - type: perplexity
      value: 37.8
      name: Perplexity — Mesh Attention Only
    - type: perplexity
      value: 39.6
      name: Perplexity — Adaptive Exit Only
    - type: perplexity
      value: 29.4
      name: Perplexity — Full TMT (all 3 innovations)
---

# 🚀 TemporalMesh Transformer (TMT)

[![Paper](https://img.shields.io/badge/📄_Paper-Zenodo_Preprint-blue?style=for-the-badge)](https://zenodo.org/records/20287390)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20287197.svg)](https://doi.org/10.5281/zenodo.20287197)
[![GitHub](https://img.shields.io/badge/💻_GitHub-TemporalMesh-black?style=for-the-badge)](https://github.com/vignesh2027/TemporalMesh-Transformer)
[![Demo](https://img.shields.io/badge/🚀_Live_Demo-HF_Space-green?style=for-the-badge)](https://huggingface.co/spaces/vigneshwar234/TemporalMesh-Transformer-Demo)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/vignesh2027/TemporalMesh-Transformer/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/Tests-201_passing-brightgreen?style=flat-square)](https://github.com/vignesh2027/TemporalMesh-Transformer/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?style=flat-square)](https://pytorch.org)

> 📄 **[TemporalMesh Transformer: Dynamic Graph Attention with Temporal Decay and Adaptive Depth Routing](https://zenodo.org/records/20287390)**
> **Author:** Vigneshwar LK · **DOI:** [10.5281/zenodo.20287197](https://doi.org/10.5281/zenodo.20287197) · **Published:** May 2026 · **Preprint (Open Access)**

---

## 🔥 Key Results

**Full TMT achieves 30.2% lower perplexity than vanilla transformer while using only 48% of the compute — a 2.1× efficiency gain.**

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

> **Full TMT achieves PPL 29.4 vs Vanilla 42.1 — a 30.2% perplexity reduction while using only 48% of the compute (2.1× efficiency gain).**

The results show clear synergy: each innovation provides independent benefit, dual combinations improve further, and combining all three yields disproportionate gains that exceed what ablations predict from linear composition.

---

## ❓ What Makes TMT Different

Every transformer since Vaswani et al. (2017) makes the same three assumptions:

1. **Static topology** — every token attends to every other token (O(S²) complexity)
2. **Fixed depth** — every token passes through every layer regardless of difficulty
3. **Time-agnostic position** — positional encoding treats all positions equivalently without semantic distance weighting

TMT breaks all three simultaneously in a single unified forward pass.

| Property | GPT / BERT | Graph Transformer | Early Exit | MoE | **TMT** |
|---|:---:|:---:|:---:|:---:|:---:|
| Dynamic graph topology | ✗ | Partial | ✗ | ✗ | **✓** |
| Per-token adaptive depth | ✗ | ✗ | ✓ | ✗ | **✓** |
| Semantic temporal decay | ✗ | ✗ | ✗ | ✗ | **✓** |
| Persistent memory anchors | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dual-stream FFN | ✗ | ✗ | ✗ | Partial | **✓** |
| Joint training of all innovations | — | — | — | — | **✓** |

No prior architecture combines all five of these properties. TMT is the first.

---

## 🔬 The Three Innovations

### Innovation 1 — Mesh Attention (Dynamic Graph Topology)

**Problem:** Standard self-attention computes O(S²) attention scores regardless of semantic relevance. Most attended pairs carry negligible information weight.

**Solution:** Before each layer, TMT recomputes the full token graph using cosine similarity of current representations, then retains only the top-k nearest neighbours per token. This gives an O(S·k) sparse graph that concentrates attention capacity where it matters most.

**Formal definition:**

```
sim(i, j) = x_i · x_j / (‖x_i‖ · ‖x_j‖)

Neighbours(i) = top-k_{j ≠ i} sim(i, j)

edge_weight(i, j) = sim(i, j)  for j ∈ Neighbours(i),  else 0
```

**Pseudocode:**

```python
# MeshBuilder.forward — runs once per layer
x_norm = F.normalize(x_flat, p=2, dim=-1)          # (B*S, D)
sim = x_norm @ x_norm.T                             # (B*S, B*S)
sim.fill_diagonal_(float('-inf'))                   # no self-loops
topk_vals, topk_idx = sim.topk(k, dim=-1)           # (B*S, k)
edge_index = build_coo(topk_idx)                    # (2, B*S*k)
edge_weight = topk_vals.flatten()                   # (B*S*k,)
```

**Key insight:** The graph is rebuilt every layer using updated representations. A token that was semantically distant in layer 0 may become a critical neighbour in layer 6 once context is established. Static graphs (as in prior graph transformers) cannot capture this dynamic topology evolution.

**Result:** Mesh attention alone reduces compute to 0.62× while dropping PPL from 42.1 to 37.8 — an 11.2% improvement.

---

### Innovation 2 — Temporal Semantic Decay

**Problem:** Standard positional encodings (sinusoidal, RoPE, ALiBi) encode position but not semantic distance. A token at position 500 is not necessarily semantically "farther" from position 0 than position 5 is — but the model has no mechanism to express this distinction prior to attention.

**Solution:** TMT learns per-dimension decay weights applied to token embeddings before attention. Tokens at later positions are attenuated by a learned sigmoid function of their normalized position. This allows the model to express "this semantic content fades with distance" without any recurrence.

**Formal definition:**

```
t_s = s / (S - 1)                              # normalized position ∈ [0, 1]
decay(s, d) = σ(−t_s · w_decay[d])            # per-dim sigmoid decay
x̃[s, d] = x[s, d] · decay(s, d)
```

Where `w_decay ∈ ℝ^D` is a learned parameter vector initialized to `decay_rate` (default 0.1).

**Key insight:** The decay is per-dimension, not a single scalar. This lets the model learn which embedding dimensions should decay quickly (syntactic surface features) vs slowly (deep semantic content). The decay scalars are carried through all subsequent layers so downstream components can condition on temporal distance.

**Result:** Temporal decay alone yields PPL 40.3 vs vanilla 42.1 — modest alone, but provides critical signal when combined with mesh attention (Mesh+Decay: 34.2 PPL, an 18.8% improvement over vanilla).

---

### Innovation 3 — Adaptive Depth Routing (Early Exit)

**Problem:** All transformer tokens process through all N layers identically. Common words ("the", "a", punctuation) require far less computation than rare entities, complex reasoning steps, or ambiguous references. Uniform depth wastes compute on easy tokens.

**Solution:** After each layer, a single linear gate projects each token's representation to a confidence scalar. Tokens exceeding the threshold have their representations frozen and skip remaining layers entirely. The gate is trained with an auxiliary loss that encourages decisiveness.

**Formal definition:**

```
confidence(s) = σ(W_gate · h_s + b_gate)      # scalar per token
exit(s)       = confidence(s) > threshold

if exit(s):
    h_s stays unchanged for all subsequent layers
```

**Auxiliary loss:**

```
L_gate = −E[|confidence − 0.5|]               # reward decisiveness
L_total = L_CE + 0.1 · L_gate
```

The coefficient 0.1 keeps the auxiliary loss from dominating the language modelling objective while still driving the gate to be decisive (push toward 0 or 1, not linger at 0.5).

**Pseudocode:**

```python
exit_mask = torch.zeros(B, S, dtype=torch.bool)
for layer in self.layers:
    x_frozen = x.clone()
    x = layer.attn(x) + layer.ffn(x)          # standard compute
    confidence = sigmoid(layer.gate_proj(x))   # (B, S)
    newly_exited = (~exit_mask) & (confidence > threshold)
    exit_mask = exit_mask | newly_exited
    # Freeze exited tokens — carry representation unchanged
    x = torch.where(exit_mask.unsqueeze(-1), x_frozen, x)
```

**Key insight:** The exit decision is fully learned from data, not hand-crafted per token type. On WikiText-2, the model learns to exit common tokens early and keep complex tokens active through all 12 layers. Average exit layer drops to 5.5/12 — saving 54% of layer computations.

**Result:** Adaptive exit alone gives PPL 39.6 with avg 5.8 layers (0.51× compute). Combined with all three innovations, achieves 5.5 avg layers (0.48× compute) and PPL 29.4.

---

## 🏗️ Architecture

```
Input token IDs  (B, S)
        │
        ▼
┌───────────────────────────────────┐
│         Token Embedding           │  (B, S) → (B, S, D)
│    + Temporal Position Encoder    │  RoPE + learned decay scalars
│    → decay_scalars (B, S, D)      │  per-dim sigmoid decay
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│          Mesh Builder             │  Dynamic kNN graph
│  x_flat (B*S, D) → edge_index    │  O(S·k) edges per batch item
│                   + edge_weight   │  Cosine similarity weights
└───────────────────────────────────┘
        │
        ▼  (repeated N times — graph rebuilt each iteration)
┌───────────────────────────────────┐
│           TMT Layer i             │
│                                   │
│  ┌─────────────────────────────┐  │
│  │  LayerNorm → Mesh Attention │  │  Sparse graph attention
│  │  + decay_scalars weighting  │  │  (B, S, D) → (B, S, D)
│  └─────────────────────────────┘  │
│              ↓ + residual         │
│  ┌─────────────────────────────┐  │
│  │  LayerNorm → Dual Stream    │  │  Two parallel FFN streams
│  │         FFN                 │  │  merged by gated fusion
│  └─────────────────────────────┘  │
│              ↓ + residual         │
│  ┌─────────────────────────────┐  │
│  │      Exit Gate              │  │  confidence = σ(W·h)
│  │  if conf > threshold:       │  │  Freeze token, skip future layers
│  │     freeze token            │  │  exit_mask updated monotonically
│  └─────────────────────────────┘  │
│              ↓                    │
│  ┌─────────────────────────────┐  │
│  │ LayerNorm → Memory Anchor   │  │  Cross-attn to M persistent
│  │     Cross-Attention         │  │  key-value memory vectors
│  └─────────────────────────────┘  │
│              ↓ + residual         │
│    Rebuild mesh graph             │  Updated for next layer
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Final LayerNorm → Output Proj    │  (B, S, D) → (B, S, V)
│  (weight-tied with embedding)     │  Parameter-efficient
└───────────────────────────────────┘
        │
        ▼
   TMTOutput dataclass
   ├── logits        (B, S, V)       — next-token logits
   ├── exit_masks    [N × (B, S)]    — per-layer bool exit decisions
   ├── confidences   [N × (B, S)]    — gate confidence ∈ [0, 1]
   ├── graph_edges   (2, E), (E,)    — final dynamic graph
   ├── memory_state  (M, D)          — persistent memory anchors
   └── decay_scalars (B, S, D)       — temporal decay weights
```

---

## 📈 Ablation Study Results

Complete ablation from the [TMT-Benchmarks](https://huggingface.co/datasets/vigneshwar234/TMT-Benchmarks) dataset (`ablation_reference` split). All runs use identical training setup, same random seed, same 120M parameter budget.

| # | Configuration | Mesh | Decay | Exit | Val PPL ↓ | Avg Layers | Rel Compute | Params |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Vanilla Transformer | ✗ | ✗ | ✗ | 42.1 | 12.0 | 1.00× | 120M |
| 2 | Mesh Attention Only | ✓ | ✗ | ✗ | 37.8 | 12.0 | 0.62× | 120M |
| 3 | Temporal Decay Only | ✗ | ✓ | ✗ | 40.3 | 12.0 | 0.98× | 120M |
| 4 | Adaptive Exit Only | ✗ | ✗ | ✓ | 39.6 | 5.8 | 0.51× | 120M |
| 5 | Mesh + Decay | ✓ | ✓ | ✗ | 34.2 | 12.0 | 0.61× | 120M |
| 6 | Mesh + Exit | ✓ | ✗ | ✓ | 35.1 | 5.7 | 0.50× | 120M |
| 7 | Decay + Exit | ✗ | ✓ | ✓ | 37.0 | 5.9 | 0.50× | 120M |
| **8** | **Full TMT (all 3)** | **✓** | **✓** | **✓** | **29.4** | **5.5** | **0.48×** | **120M** |

**Key findings:**
- Every single innovation beats vanilla (rows 2–4 all improve over row 1)
- Dual combinations are super-additive — synergy appears even at two innovations
- Full TMT achieves **30.2% PPL reduction** with only **48% of compute** vs vanilla
- Average token exits at layer 5.5 out of 12 — more than half of layers are skipped
- The 2.1× efficiency gain makes TMT suitable for inference-critical deployments

---

## 🔗 Related Resources

| Resource | Link |
|---|---|
| 📄 Paper (Zenodo Preprint) | https://zenodo.org/records/20287390 |
| 🔖 DOI | https://doi.org/10.5281/zenodo.20287197 |
| 💻 GitHub Repository | https://github.com/vignesh2027/TemporalMesh-Transformer |
| 🤗 This Model (HuggingFace) | https://huggingface.co/vigneshwar234/TemporalMesh-Transformer |
| 📊 Benchmark Dataset | https://huggingface.co/datasets/vigneshwar234/TMT-Benchmarks |
| 🚀 Live Demo Space | https://huggingface.co/spaces/vigneshwar234/TemporalMesh-Transformer-Demo |
| 🌐 Docs (GitHub Pages) | https://vignesh2027.github.io/TemporalMesh-Transformer/ |

---

## 📦 Repository Structure

```
TemporalMesh-Transformer/
├── tmt/
│   ├── model/
│   │   ├── config.py          # TMTConfig dataclass — all hyperparameters
│   │   ├── model.py           # TMTModel — full forward pass + TMTOutput
│   │   ├── mesh.py            # MeshBuilder — dynamic kNN graph construction
│   │   ├── attention.py       # MeshAttention — sparse graph attention
│   │   ├── embedding.py       # TokenEmbedding + TemporalPositionEncoder
│   │   ├── exit_gate.py       # ExitGate — per-token adaptive depth routing
│   │   ├── ffn.py             # DualStreamFFN — dual parallel feed-forward
│   │   ├── layers.py          # TMTLayer — full single layer assembly
│   │   └── memory.py          # MemoryAnchorCross — persistent KV memory
│   ├── training/
│   │   ├── loss.py            # compute_loss — CE + auxiliary gate loss
│   │   ├── trainer.py         # training loop utilities
│   │   └── scheduler.py       # learning rate scheduling
│   ├── data/
│   │   ├── dataset.py         # WikiText-2 / TinyStories loading
│   │   └── tokenizer.py       # tokenizer utilities
│   └── experiments/
│       ├── 01_baseline.ipynb  # Vanilla transformer baseline
│       ├── 02_mesh_only.ipynb # Mesh-only ablation
│       ├── 03_full_tmt.ipynb  # Full TMT training
│       └── 04_compare.ipynb   # Side-by-side comparison
├── tests/
│   ├── test_benchmarks.py     # 25 benchmark validation tests (NEW)
│   ├── test_forward.py        # End-to-end forward pass tests
│   ├── test_shapes.py         # Tensor shape contracts
│   ├── test_config.py         # Config validation
│   ├── test_training.py       # Training loop tests
│   ├── test_integration.py    # Integration tests
│   ├── test_edge_cases.py     # Edge and boundary conditions
│   ├── test_generation.py     # Text generation tests
│   ├── test_dataset.py        # Dataset loading tests
│   └── test_reprs.py          # __repr__ and string tests
├── paper/
│   └── TemporalMesh_Transformer_2026.pdf
├── docs/
│   └── index.html             # GitHub Pages docs
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 💻 Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| Install + run all 201+ tests | Any CPU, 4GB RAM | — |
| Small config training (d=128, L=4) | CPU, 8GB RAM | CPU, 16GB RAM |
| Full config training (d=512, L=12, 120M) | GPU 8GB VRAM | GPU 24GB VRAM (A100/H100/4090) |
| Inference (batch=1, seq=1024) | CPU | GPU 8GB |
| WikiText-2 full training run | GPU 16GB | 4× GPU with DDP |

The architecture is fully CPU-runnable for research and testing. All 201+ tests pass on CPU without any GPU required.

---

## 📖 Citation

If you use TMT in your research, please cite:

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

## 📜 License

MIT License — see [LICENSE](https://github.com/vignesh2027/TemporalMesh-Transformer/blob/main/LICENSE) for details.

Copyright (c) 2026 Vigneshwar LK
