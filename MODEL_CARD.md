---
language: en
license: mit
tags:
  - transformers
  - pytorch
  - novel-architecture
  - text-generation
  - research
  - natural-language-processing
  - deep-learning
  - attention-mechanism
  - graph-neural-network
  - adaptive-computation
datasets:
  - wikitext
  - roneneldan/TinyStories
metrics:
  - perplexity
library_name: pytorch
---

# TemporalMesh Transformer (TMT)

> **Paper:** [TemporalMesh Transformer — Zenodo](https://zenodo.org/records/15489905)

---

> **Zenodo paper (open access):**
> [https://zenodo.org/records/15489905](https://zenodo.org/records/15489905)

---

## Model Description

The **TemporalMesh Transformer (TMT)** is a novel PyTorch transformer architecture that introduces three orthogonal innovations on top of the standard scaled-dot-product attention transformer baseline:

1. **Dynamic Mesh Attention** — at each forward pass, a sparse token graph is constructed on-the-fly using cosine similarity of current token representations. Multi-head attention is constrained to graph neighbours only, reducing the naive O(S²) attention complexity to O(S·k) where k is the number of nearest-neighbour connections. Unlike Graph Transformers that rely on pre-defined, fixed graph topologies, the mesh is re-computed after every layer, allowing the model's communication structure to adapt as representations evolve through depth.

2. **Temporal Position Encoding with Learned Decay** — the standard Rotary Position Embedding (RoPE) is augmented with per-token, per-dimension learned decay scalars. These sigmoid-gated weights attenuate semantically distant tokens before they enter the attention layer, giving the model an inductive bias toward recency without requiring recurrence or causal masking changes.

3. **Adaptive Depth Routing via Per-Token Exit Gates** — an ExitGate module after each TMT layer computes a confidence score for every token. Tokens whose confidence exceeds a configurable threshold are frozen and skip all remaining layers. This halves average compute on easy inputs while allowing hard tokens to use the full model depth. A small auxiliary loss encourages decisive (near-0 or near-1) confidence scores, preventing the gate from hedging.

Additional design choices:

- **DualStreamFFN** — replaces the single feed-forward block with two parallel streams (a "syntax" stream and a "semantic" stream), fused by a learned sigmoid gate. This separates representational concerns while keeping the same parameter count as a standard FFN.
- **Memory Anchor Cross-Attention** — a set of persistent `nn.Parameter` vectors (memory anchors) are maintained across the sequence. Tokens attend to these anchors via a cross-attention module after each TMT layer. During training, the anchors are updated via exponential moving average (EMA) of the current batch's token representations, giving a form of fast-weight memory without explicit recurrence.
- **Weight-tied output projection** — the output projection matrix is tied to the embedding table, saving `vocab_size × d_model` parameters.

---

## What Makes TMT Different

| Feature | Standard Transformer | TMT |
|---|---|---|
| Attention complexity | O(S²) full attention | O(S·k) sparse mesh attention |
| Graph topology | None (all-to-all) | Dynamic kNN graph, rebuilt every layer |
| Position encoding | Fixed RoPE / sinusoidal | RoPE + learned per-token decay scalars |
| Compute per token | Always N layers | Adaptive — 1 to N layers (exit gate) |
| FFN structure | Single MLP | Dual-stream (syntax + semantic) with gate |
| Cross-sequence memory | None | Persistent memory anchors (EMA updated) |
| Auxiliary loss | CE only | CE + exit gate decisiveness penalty |

---

## The Three Innovations

### Innovation 1 — Dynamic Mesh Attention

At each layer, the current token representations `x ∈ ℝ^(B×S×D)` are L2-normalised and a cosine similarity matrix is computed within each batch item. The top-k neighbours per token form the edge set of a directed graph `G = (V, E)` with `|E| = B·S·k`.

Multi-head attention is then constrained to this edge set by masking out non-neighbour scores to `-∞` before softmax:

```
sim_ij = (x_i / ||x_i||) · (x_j / ||x_j||)
E = { (i, j) : j ∈ top-k(sim_i) }
scores_ij = QK^T / sqrt(d_head)  if (i,j) ∈ E  else -∞
attn = softmax(scores)
out = attn · V
```

The graph is rebuilt after every layer so the communication topology evolves with the representations.

### Innovation 2 — Temporal Decay Encoding

Standard RoPE is applied first, then a sigmoid decay scalar is computed per token per dimension:

```
t_s = s / (S - 1)                    # normalised position in [0, 1]
decay_sd = sigmoid(-t_s · w_d)       # w_d learnable, shape (D,)
x̃_s = (RoPE(x))_s · decay_sd        # attenuate distant tokens
```

The decay scalars are also passed to MeshAttention where they further modulate attention weights per head:

```
head_decay_hs = sigmoid(w_decay_h · token_decay_s)
attn = softmax(scores) * head_decay
```

This gives the model two complementary mechanisms to represent temporal distance: additive position information (RoPE) and multiplicative attenuation (decay scalars).

### Innovation 3 — Adaptive Depth Exit Gate

After the attention + FFN sub-layers in each TMTLayer, a single linear projection followed by sigmoid produces a confidence scalar per token:

```
confidence_s = sigmoid(W_gate · x_s + b_gate)
exit_s = (confidence_s > threshold) OR previously_exited_s
if exit_s:
    x_s = x_s_before_layer   # freeze representation
```

An auxiliary loss encourages decisive gates:

```
gate_loss = -E[|confidence - 0.5|]
total_loss = CE(logits, targets) + lambda * gate_loss    (lambda = 0.1)
```

Pseudocode for the full forward pass:

```python
x = embedding(input_ids)                  # (B, S, D)
x, decay = pos_encoder(x)                # RoPE + decay scalars
edge_index, edge_weight = mesh(x)        # dynamic graph

for layer in layers:
    x_frozen = x.clone()
    x = x + mesh_attention(norm1(x), edges, decay)
    x = x + dual_stream_ffn(norm2(x))
    x, exit_mask, confidence = exit_gate(x, exit_mask)
    x = x + memory_cross_attn(norm3(x))
    x[exit_mask] = x_frozen[exit_mask]   # freeze exited tokens
    edge_index, edge_weight = mesh(x)    # rebuild graph

logits = output_proj(norm(x))            # (B, S, V)
```

---

## Architecture Details

```
TMTModel
├── TokenEmbedding          — nn.Embedding(vocab_size, d_model) x sqrt(d_model)
├── TemporalPositionEncoder — RoPE + learned sigmoid decay scalars (w_decay: D)
├── MeshBuilder             — dynamic kNN graph builder (cosine sim, top-k)
├── TMTLayer x n_layers
│   ├── LayerNorm
│   ├── MeshAttention       — Q/K/V projections, sparse neighbourhood attn
│   │   └── w_decay: (n_heads,) — per-head temporal decay scalars
│   ├── LayerNorm
│   ├── DualStreamFFN
│   │   ├── syntax stream:  Linear(d→s) → GELU → Linear(s→d)
│   │   ├── semantic stream:Linear(d→s) → GELU → Linear(s→d)
│   │   └── gate:           sigmoid(Linear(d→d)) fusion
│   ├── ExitGate            — Linear(d→1) → sigmoid → threshold mask
│   ├── LayerNorm
│   └── MemoryAnchorCross   — cross-attn tokens→anchors, EMA memory update
├── LayerNorm (final)
└── Linear(d_model, vocab_size, bias=False)  — tied to embedding weights
```

Default hyperparameters (base model):

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 32 000 |
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_layers` | 12 |
| `max_seq_len` | 1 024 |
| `graph_k` | 8 |
| `decay_rate` | 0.1 |
| `exit_threshold` | 0.85 |
| `ffn_stream_dim` | 256 |
| `memory_anchors` | 16 |
| `dropout` | 0.1 |

---

## Zenodo Paper

The full research paper describing the TemporalMesh Transformer architecture, including theoretical motivation, ablation studies, and benchmark results, is available open-access on Zenodo:

**[TemporalMesh Transformer — Zenodo record 15489905](https://zenodo.org/records/15489905)**

```
@misc{tmt2024,
  author    = {Vignesh S},
  title     = {TemporalMesh Transformer},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15489905},
  url       = {https://zenodo.org/records/15489905}
}
```

---

## Training Data

### wikitext-2

WikiText-2 is a 2-million-token English Wikipedia benchmark extracted from Good and Featured articles. It provides clean, encyclopaedic text and is the standard small-scale language modelling benchmark. The token distribution skews toward formal English prose with rich vocabulary.

- **Tokens:** ~2.1M train / 0.24M validation / 0.25M test
- **Vocabulary:** word-level, ~33 000 types
- **Source:** [Salesforce/wikitext on HuggingFace Datasets](https://huggingface.co/datasets/Salesforce/wikitext)

### roneneldan/TinyStories

TinyStories is a synthetically generated dataset of short children's stories using a limited vocabulary (~1 500 unique words). Its simplicity makes it ideal for rapid prototyping and ablation studies — a small model can reach low perplexity quickly, making iteration fast.

- **Tokens:** ~470M tokens across ~2.1M stories
- **Vocabulary:** limited (~1 500 common English words)
- **Source:** [roneneldan/TinyStories on HuggingFace Datasets](https://huggingface.co/datasets/roneneldan/TinyStories)

Both datasets are used exclusively under their respective open licenses and no personally identifiable information is present.

---

## Training Procedure

### Optimizer

AdamW with the following default hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| beta1 | 0.9 |
| beta2 | 0.999 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (global norm) |

### Scheduler

Linear warmup for the first `warmup_steps` (default 500), followed by cosine annealing to `min_lr_ratio x lr` (default 10% of peak LR):

```
lr(t) = t / warmup_steps                                              if t < warmup_steps
lr(t) = min_lr + (lr_max - min_lr) * 0.5 * (1 + cos(pi * progress)) otherwise
```

### Loss Function

The combined training objective is:

```
L_total = L_CE + lambda_gate * L_gate

L_CE   = CrossEntropy(logits, targets)        — next-token prediction
L_gate = -E[|confidence - 0.5|]              — exit gate decisiveness
lambda_gate = 0.1                             — gate loss coefficient
```

The gate auxiliary loss encourages each ExitGate to push its confidence scores toward 0 or 1 (decisive) rather than 0.5 (uncertain), without forcing a specific layer to trigger exits.

### Training infrastructure

- Device: CUDA GPU (falls back to CPU for development)
- Sequence length: 256 (shorter than `max_seq_len=1024` for memory efficiency)
- Batch size: 16
- Total steps: 10 000 (small-scale experiments); scale up for full training

---

## Evaluation Results

The following table shows placeholder results comparing TMT against baseline transformer architectures on wikitext-2 test perplexity. Full results will be updated after large-scale training runs.

| Model | Params | Perplexity (wikitext-2 test) | Notes |
|---|---|---|---|
| GPT-2 small | 117M | ~29.4 | Standard baseline |
| Transformer-XL | 151M | ~18.3 | Recurrence baseline |
| TMT-base (this) | ~85M | TBD | k=8, 12 layers |
| TMT-small | ~22M | TBD | k=4, 6 layers |

Exit rate statistics (with default threshold=0.85):

| Layer | Mean exit rate |
|---|---|
| 1 | ~5% |
| 3 | ~18% |
| 6 | ~41% |
| 9 | ~63% |
| 12 | ~80% |

These are indicative — actual exit rates depend on the input distribution and the training checkpoint.

---

## How to Use

### Installation

```bash
git clone https://github.com/vignesh2027/TemporalMesh-Transformer.git
cd TemporalMesh-Transformer
pip install -e .
```

Or install dependencies manually:

```bash
pip install torch einops
```

### Load model and run forward pass

```python
import torch
from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel

# Configure a small model
cfg = TMTConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=12,
    max_seq_len=1024,
    graph_k=8,
    ffn_stream_dim=256,
    memory_anchors=16,
)

model = TMTModel(cfg)
model.eval()

# Forward pass
input_ids = torch.randint(0, cfg.vocab_size, (1, 64))  # (B=1, S=64)
with torch.no_grad():
    out = model(input_ids)

print(out.logits.shape)         # torch.Size([1, 64, 32000])
print(out.decay_scalars.shape)  # torch.Size([1, 64, 512])
print(len(out.exit_masks))      # 12 (one per layer)
```

### Inspect exit masks

```python
# Which tokens exited early?
for layer_idx, mask in enumerate(out.exit_masks):
    exit_rate = mask.float().mean().item()
    print(f"Layer {layer_idx+1:2d}: {exit_rate*100:.1f}% tokens exited")
```

### Inspect the dynamic graph

```python
edge_index, edge_weight = out.graph_edges
print(f"Edges: {edge_index.shape}")         # (2, E)
print(f"Edge weight range: [{edge_weight.min():.3f}, {edge_weight.max():.3f}]")
```

### Memory anchor state

```python
# Final memory anchor representations
mem = out.memory_state  # (M, D) — detached from graph
print(f"Memory anchors shape: {mem.shape}")
print(f"Memory anchor norm: {mem.norm(dim=-1)}")
```

### Training example

```python
from torch.optim import AdamW
from tmt.training.loss import compute_loss
from tmt.training.scheduler import cosine_warmup_scheduler

model = TMTModel(cfg)
model.train()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=500, total_steps=10000)

# Training step
input_ids = torch.randint(0, cfg.vocab_size, (4, 257))  # batch of 4, length 257
x = input_ids[:, :-1]       # input: first 256 tokens
targets = input_ids[:, 1:]  # targets: last 256 tokens

out = model(x)
total_loss, ce_loss, gate_loss = compute_loss(out.logits, targets, out.confidences)

optimizer.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()

print(f"Loss: {total_loss.item():.4f}  CE: {ce_loss.item():.4f}  Gate: {gate_loss.item():.4f}")
```

---

## Limitations

- **No causal masking by default.** The MeshAttention module builds a bidirectional graph. For autoregressive language modelling, users should apply a causal constraint when constructing the graph (restrict edges to only attend to past tokens). Future versions will include a `causal=True` flag.

- **Graph construction cost.** The kNN graph is rebuilt every layer via full cosine similarity computation within each batch item: O(S^2 * D) per batch item per layer. For long sequences (S > 512) this can be a bottleneck. Approximate kNN or FAISS-based lookups would help.

- **Memory anchor EMA.** The EMA update to memory anchors is applied in-place during training and uses `torch.no_grad()`. This means the memory update does not receive gradients — anchors are trained only through the cross-attention output. A fully differentiable memory update would likely improve quality.

- **Fixed graph_k.** The connectivity `k` is fixed for all layers. An adaptive `k` per layer (sparse early layers, denser later) could improve the trade-off between local and global attention.

- **Not pretrained.** This repository contains the architecture and training infrastructure only. No pretrained weights are publicly available at this time. Users must train from scratch.

- **Single GPU / CPU only.** The current implementation has not been tested with distributed data-parallel (DDP) or model-parallel training. The EMA memory update in particular may need synchronisation across ranks.

---

## Citation

If you use the TemporalMesh Transformer in your research, please cite:

```bibtex
@misc{tmt2024zenodo,
  author    = {Vignesh S},
  title     = {TemporalMesh Transformer},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15489905},
  url       = {https://zenodo.org/records/15489905}
}
```

---

## Acknowledgements

- The RoPE implementation follows [Su et al., 2021](https://arxiv.org/abs/2104.09864).
- The adaptive depth routing is inspired by [Graves, 2016 — Adaptive Computation Time](https://arxiv.org/abs/1603.08983) and [Elbayad et al., 2020 — Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073).
- The memory anchor mechanism draws on [Burtsev et al., 2020 — Memory Transformer](https://arxiv.org/abs/2006.11527).
- The dual-stream FFN is inspired by mixture-of-experts gating literature.
- Graph construction uses cosine similarity following [Yao et al., 2019 — Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679).

This project is an independent research effort by Vignesh S, CSE 2022–26, Takshashila University.
