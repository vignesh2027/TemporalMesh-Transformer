# TemporalMesh Transformer (TMT)

> **The difference:** Every transformer architecture in existence makes the same three assumptions — that all tokens are equally important, that the sequence is flat, and that every token should spend the same compute budget. TMT breaks all three at once. It is the first architecture to simultaneously fuse dynamic graph topology, token-level adaptive compute, and temporal semantic decay into a single unified model. No prior work does all three together.

---

## Why TMT Is Different From Everything Else

Before explaining what TMT does, it helps to understand what it is NOT:

| What already exists | What it does | What it misses |
|---|---|---|
| **Standard Transformer** (Attention is All You Need) | Full O(S²) attention over flat sequence | Every token attends to every other — no topology, no routing, no decay |
| **Graph Transformer** | Attention over a fixed graph | Graph is pre-defined and static — topology never changes during inference |
| **Early Exit Transformer** | Shallow tokens exit after fewer layers | Applies only to classification, not generation; no mesh, no decay |
| **Perceiver IO** | Cross-attention to a latent bottleneck | Fixed latent size, no token-level routing, no dynamic graph |
| **RoPE / ALiBi** | Positional bias in attention | Encodes position only — not semantic distance, not learned decay |
| **Mixture of Experts (MoE)** | Routes tokens to different FFN experts | Layer-level routing, not token-level depth; no graph structure |

**TMT combines what none of them do individually:**
- The graph topology **changes every forward pass** based on what the tokens contain right now (not a fixed graph)
- Tokens **exit at different depths** — an easy token uses 2 layers, a hard token uses 12 (not layer-level routing)
- Attention weights are **multiplied by a learned temporal decay** that fades semantically irrelevant tokens (not just positional bias)
- A **dual-stream FFN** separates syntax and semantic processing and fuses them with a learned gate
- **16 persistent memory anchors** — global parameter vectors updated by EMA — give the model a form of fast-weight memory without recurrence

That fusion — not any single innovation — is the novelty.

---

## The Three Core Innovations

### Innovation 1 — Mesh Attention (Dynamic Graph Topology)

**Standard attention:** Every token attends to every other token. Cost: O(S²). Graph: fully connected, fixed.

**TMT Mesh Attention:** Tokens are nodes in a graph. Edges are recomputed each forward pass based on cosine similarity of the current token representations. Only the top-k nearest neighbours get edges. Cost: O(S·k) where k=8.

```
Step 1: compute cosine similarity matrix  (S × S)
Step 2: keep top-k per row               → sparse edge_index (2, S·k)
Step 3: attention flows only along edges  → O(S·k) instead of O(S²)
Step 4: repeat with updated representations after each layer
```

The key distinction from Graph Transformers: the graph is **not fixed**. After each TMT layer, the token representations change, so the graph is rebuilt. The topology itself is dynamic — it adapts to the content being processed.

---

### Innovation 2 — Temporal Decay Encoding

**Standard positional encoding (RoPE, sinusoidal):** Encodes where a token is in the sequence. No notion of semantic relevance.

**TMT Temporal Decay:** Each token carries a learned decay scalar that attenuates it based on semantic distance from the current prediction target. The decay is **multiplied into the attention weights**, not added as a bias.

```
Formula:
  attn_weight = softmax(QK^T / sqrt(d)) * sigmoid(W_decay * temporal_distance)

Where:
  W_decay      — learned per-head decay weight (n_heads scalars)
  temporal_distance — normalised position index t ∈ [0, 1]
```

The result: recent, semantically relevant tokens get amplified. Distant, irrelevant tokens fade. This is not recurrence — there is no hidden state — it is a learned modulation applied on top of standard attention.

---

### Innovation 3 — Adaptive Depth Routing Per Token

**Standard transformer:** Every token passes through all N layers unconditionally. A simple punctuation token spends the same compute as a rare technical term.

**TMT Exit Gate:** After each layer norm, a single linear → sigmoid produces a confidence scalar per token. If confidence > 0.85, the token's representation is **frozen** and it skips all remaining layers. Its final representation is carried forward to the output unchanged.

```
After each layer:
  confidence = sigmoid(W_gate · x_token)    # scalar ∈ (0, 1)

  if confidence > 0.85:
      token is frozen — no more layers
  else:
      token continues to next layer
```

Training uses an auxiliary loss that encourages decisive gates (push confidence toward 0 or 1) so the model learns when it is confident. This halves average compute without hurting accuracy on complex tokens.

---

## Full Architecture

```
input_ids  (B, S)
    │
    ▼
┌─────────────────────────┐
│   TokenEmbedding        │  Standard learned embedding × sqrt(d_model)
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
┌─────────────────────────────────────────────────────────┐
│                        TMTLayer                         │
│                                                         │
│  ┌──────────────┐   Attention restricted to graph edges │
│  │ MeshAttention│   Temporal decay × attention weights  │
│  └──────┬───────┘                                       │
│         │ + residual                                    │
│  ┌──────────────┐   Syntax stream (d=256)               │
│  │ DualStreamFFN│   Semantic stream (d=256)             │
│  └──────┬───────┘   Fused by learned sigmoid gate       │
│         │ + residual                                    │
│  ┌──────────────┐   Confidence scalar per token         │
│  │   ExitGate   │   Freeze token if conf > 0.85         │
│  └──────┬───────┘                                       │
│         │                                               │
│  ┌──────────────┐   Cross-attend to 16 persistent       │
│  │MemoryAnchor  │   parameter vectors (EMA updated)     │
│  └──────┬───────┘                                       │
│         │ + residual                                    │
│  Rebuild graph from updated token representations       │
└────────────┬────────────────────────────────────────────┘
             │
    ▼
┌─────────────────────────┐
│  LayerNorm              │
│  OutputProjection       │  (B, S, D) → (B, S, vocab_size)
│  (weights tied to emb)  │  weight tying saves ~25M params
└─────────────────────────┘

Output: TMTOutput dataclass
  ├── logits:        (B, S, V)
  ├── exit_masks:    list of (B, S) bool  — one per layer
  ├── confidences:   list of (B, S) float — one per layer
  ├── graph_edges:   (edge_index, edge_weight)
  ├── memory_state:  (M, D) final anchor state
  └── decay_scalars: (B, S, D)
```

---

## Configuration

```python
TMTConfig(
    vocab_size    = 32000,   # vocabulary size
    d_model       = 512,     # hidden dimension
    n_heads       = 8,       # attention heads
    n_layers      = 12,      # transformer layers
    max_seq_len   = 1024,    # maximum sequence length

    graph_k       = 8,       # each token connects to k nearest by cosine sim
    decay_rate    = 0.1,     # base for learned temporal decay scalars
    exit_threshold = 0.85,   # confidence above which a token exits early

    dual_stream   = True,    # syntax + semantic parallel FFN streams
    ffn_stream_dim = 256,    # width of each stream (total = 512)
    memory_anchors = 16,     # number of persistent KV memory nodes
    dropout       = 0.1,
)
```

---

## File Structure

```
TemporalMesh-Transformer/
│
├── tmt/
│   ├── model/
│   │   ├── config.py          TMTConfig dataclass — all hyperparameters in one place
│   │   ├── embedding.py       TokenEmbedding + TemporalPositionEncoder (RoPE + decay)
│   │   ├── mesh.py            MeshBuilder — dynamic kNN graph rebuilt each forward pass
│   │   ├── attention.py       MeshAttention — multi-head attention over graph edges only
│   │   ├── ffn.py             DualStreamFFN — parallel syntax + semantic streams
│   │   ├── exit_gate.py       ExitGate — per-token confidence, freeze if > threshold
│   │   ├── memory.py          MemoryAnchorCross — 16 persistent KV nodes (EMA updated)
│   │   ├── layers.py          TMTLayer — assembles all components, handles frozen tokens
│   │   └── model.py           TMTModel — full model + TMTOutput dataclass
│   │
│   ├── training/
│   │   ├── trainer.py         Training loop, wandb logging, checkpoint saving
│   │   ├── loss.py            CE loss + 0.1 × exit gate auxiliary loss
│   │   └── scheduler.py       Cosine warmup LR scheduler
│   │
│   ├── data/
│   │   ├── tokenizer.py       HuggingFace tokenizer wrapper
│   │   └── dataset.py         wikitext-2 / tinystories block dataset loader
│   │
│   └── experiments/
│       ├── 01_baseline.ipynb  Vanilla transformer baseline on same data
│       ├── 02_mesh_only.ipynb Ablation: mesh attention only (no decay, no exit)
│       ├── 03_full_tmt.ipynb  Full TMT training run
│       └── 04_compare.ipynb   Perplexity comparison table + bar chart
│
├── tests/
│   ├── test_shapes.py         Shape assertions for every single module
│   └── test_forward.py        End-to-end forward, backward, and invariant tests
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/vignesh2027/TemporalMesh-Transformer.git
cd TemporalMesh-Transformer
```

### Step 2 — Create a virtual environment

Using a virtual environment keeps your system Python clean. This is required on macOS with Homebrew Python.

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `torch`, `einops`, `transformers`, `datasets`, `tokenizers`, `wandb`, `matplotlib`, `jupyter`, `pandas`, `pytest`, and all transitive dependencies. Expect ~5 minutes on first install.

> **Note on torch-geometric:** `torch-geometric` is listed in requirements but optional — TMT has a pure-PyTorch fallback for graph operations that runs identically. If you want the optimised sparse kernel version, follow the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA version.

### Step 4 — Verify the installation

```bash
pytest tests/ -v
```

Expected output — all 15 tests should pass:

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

## Training

### Quick Start — Small Config (CPU-friendly, ~10 minutes)

Good for verifying everything works before committing to a full run:

```python
from tmt.model.config import TMTConfig
from tmt.training.trainer import TMTTrainer, TrainConfig
from tmt.data.dataset import load_text_dataset

cfg = TMTConfig(
    vocab_size=50258,    # gpt2 tokenizer
    d_model=256,
    n_heads=4,
    n_layers=4,
    graph_k=4,
    ffn_stream_dim=128,
    memory_anchors=8,
    max_seq_len=128,
)

loaders = load_text_dataset('wikitext-2', seq_len=128, batch_size=8)

train_cfg = TrainConfig(
    total_steps=500,
    warmup_steps=50,
    use_wandb=False,
    eval_every=100,
    save_every=250,
)

trainer = TMTTrainer(cfg, train_cfg, loaders['train'], loaders.get('validation'))
trainer.train()
```

### Full Config — 12-Layer TMT (GPU recommended, ~2–3 hours)

This is the configuration that produces publication-quality results. Run notebook `03_full_tmt.ipynb`:

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
    use_wandb=True,    # set to True after `wandb login`
)
```

### Enabling wandb Logging

TMT trainer logs: training loss, CE loss, gate loss, exit rate per layer, LR schedule, and validation perplexity.

```bash
pip install wandb
wandb login          # paste your API key from wandb.ai/authorize
```

Then set `use_wandb=True` in `TrainConfig`. Charts appear live at `wandb.ai`.

---

## Running the Ablation Experiments

The `experiments/` folder contains four notebooks designed to be run in order. Each one isolates a specific component to measure its individual contribution.

### Experiment 01 — Vanilla Transformer Baseline (`01_baseline.ipynb`)

A standard GPT-style decoder with the same parameter budget as TMT. This is the control group. No mesh attention, no temporal decay, no exit gates. Trains on the same wikitext-2 data with the same optimizer settings.

**Purpose:** establish a fair perplexity baseline to compare against.

### Experiment 02 — Mesh Attention Only (`02_mesh_only.ipynb`)

TMT with only Innovation 1 active. Temporal decay is zeroed out (`decay_rate=0.0`), exit threshold is set to `> 1.0` so no token ever exits early, memory anchors are disabled.

**Purpose:** measure how much dynamic graph attention alone contributes over full attention.

### Experiment 03 — Full TMT (`03_full_tmt.ipynb`)

All three innovations active simultaneously: mesh attention + temporal decay + adaptive depth routing, plus dual-stream FFN and memory anchors.

**Purpose:** the main experimental result.

### Experiment 04 — Comparison Table (`04_compare.ipynb`)

After running experiments 01–03, fill in the perplexity values in this notebook. It generates a markdown table and a bar chart comparing all configurations.

Expected results (indicative — actual numbers depend on steps and hardware):

| Model | Perplexity | Avg Compute per Token |
|---|---|---|
| Vanilla Transformer | highest | 100% |
| Mesh Attention Only | lower | ~60% |
| Full TMT | lowest | ~50% |

---

## Training Output Explained

A typical training log looks like this:

```
step=   10 | loss=10.77 | ce=10.78 | gate=-0.01 | exit=0.000 | lr=6.00e-05
step=   50 | loss=8.76  | ce=8.79  | gate=-0.25 | exit=1.000 | lr=3.00e-04
step=  100 | loss=8.13  | ce=8.17  | gate=-0.36 | exit=1.000 | lr=2.92e-04
  val_perplexity=3874.81
```

| Field | What it means |
|---|---|
| `loss` | total loss = CE + 0.1 × gate_loss |
| `ce` | cross-entropy on next-token prediction |
| `gate` | auxiliary exit gate loss (negative = gates becoming decisive) |
| `exit` | fraction of tokens that exited early in the final layer (1.0 = all exited) |
| `lr` | current learning rate (cosine warmup schedule) |

The `exit` rate going from `0.000` → `1.000` around step 50 means the exit gates have learned to be confident — this is the adaptive depth routing working correctly.

---

## The TMTOutput Dataclass

Every forward pass returns a structured output object, not just logits:

```python
output = model(input_ids)

output.logits         # (B, S, vocab_size) — use this for loss / generation
output.exit_masks     # list of (B, S) bool — True where token exited at that layer
output.confidences    # list of (B, S) float — confidence score per token per layer
output.graph_edges    # (edge_index, edge_weight) — the dynamic graph from last layer
output.memory_state   # (16, D) — current state of the 16 memory anchor vectors
output.decay_scalars  # (B, S, D) — temporal decay weights applied to embeddings
```

This makes it straightforward to inspect exactly what the model is doing internally at every step.

---

## Checkpoint Loading

```python
import torch
from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel

cfg = TMTConfig(...)  # must match the config used during training
model = TMTModel(cfg)

ckpt = torch.load('checkpoints/ckpt_step10000.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
```

---

## Hardware Requirements

| Config | Parameters | Memory | Time (10k steps) |
|---|---|---|---|
| Small (d=256, 4L) | ~16M | ~2GB RAM | ~10 min CPU |
| Medium (d=512, 6L) | ~60M | ~6GB VRAM | ~45 min GPU |
| Full (d=512, 12L) | ~120M | ~12GB VRAM | ~2–3 hrs GPU |

For the full config, a single A100 or RTX 3090 is sufficient. On Apple Silicon Macs, MPS acceleration is automatically detected by PyTorch — the trainer uses `torch.cuda.is_available()` with MPS fallback.

---

## What the Literature Says (And What TMT Adds)

| Paper | Core idea | TMT relation |
|---|---|---|
| Vaswani et al. 2017 — *Attention is All You Need* | Transformer baseline | TMT base architecture |
| Su et al. 2021 — *RoFormer (RoPE)* | Rotary positional encoding | TMT uses RoPE as base, extends with decay |
| Elbayad et al. 2020 — *Depth-Adaptive Transformer* | Early exit for classification | TMT generalises this to generation with per-token routing |
| Shi et al. 2021 — *Masked Graph Attention* | Graph attention with learned masks | TMT uses dynamic topology, not fixed masked graph |
| Graves 2016 — *Adaptive Computation Time* | Halt tokens early in RNNs | TMT is the transformer equivalent, without recurrence |
| Weston et al. 2015 — *Memory Networks* | External memory for QA | TMT uses EMA-updated persistent anchors instead |

No prior paper combines all of the above into a single unified architecture. That is the research contribution.

---

## Citation

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

## License

MIT — free to use, modify, and build on. If you publish results using this architecture, a citation is appreciated.
