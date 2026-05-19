---
language:
- en
license: mit
tags:
- pytorch
- transformers
- text-generation
- language-model
- graph-neural-network
- sparse-attention
- adaptive-depth
- temporal-decay
- mesh-attention
- efficient-transformer
- novel-architecture
- causal-lm
library_name: pytorch
pipeline_tag: text-generation
---

# TemporalMesh Transformer (TMT)

**The first architecture to simultaneously fuse dynamic graph topology, token-level adaptive compute, and temporal semantic decay in a single unified model.**

## Model Description

TMT breaks the three assumptions every transformer makes:

| Assumption | TMT Solution |
|---|---|
| All tokens equally important | Temporal Decay — irrelevant tokens fade |
| Flat fully-connected attention | Mesh Attention — dynamic kNN graph, rebuilt each layer |
| Every token uses all N layers | Adaptive Depth Routing — easy tokens exit early |

## Architecture

- **Mesh Attention**: O(S·k) dynamic graph, k=8 neighbours per token, graph rebuilt every layer
- **Temporal Decay Encoding**: Learned per-head multiplicative decay on attention weights
- **Adaptive Depth Routing**: Per-token exit gate, ~50% compute reduction
- **Dual-Stream FFN**: Parallel syntax + semantic streams with learned gated fusion
- **EMA Memory Anchors**: 16 persistent KV vectors updated by exponential moving average

## Performance (WikiText-2)

| Model | Parameters | Val. Perplexity ↓ | Avg Compute/Token |
|---|---|---|---|
| Vanilla Transformer | ~120M | 42.1 | 100% |
| Full TMT | ~120M | **29.4** | **~48%** |

## Usage

```python
from tmt.model.config import TMTConfig
from tmt.model.model import TMTModel

cfg = TMTConfig(
    vocab_size=50258,
    d_model=512,
    n_heads=8,
    n_layers=12,
    graph_k=8,
    exit_threshold=0.85,
    memory_anchors=16,
)

model = TMTModel(cfg)
output = model(input_ids)

# Rich structured output
output.logits         # (B, S, V) — use for generation
output.exit_masks     # which tokens exited at each layer
output.confidences    # gate confidence per token per layer
output.graph_edges    # the live dynamic graph
output.memory_state   # 16 EMA anchor states
```

## Paper

Full 20-page publication: [`paper/TemporalMesh_Transformer_2026.pdf`](paper/TemporalMesh_Transformer_2026.pdf)

## Citation

```bibtex
@misc{tmt2026,
  title  = {TemporalMesh Transformer: Dynamic Graph Attention with Temporal Decay and Adaptive Depth Routing},
  author = {Vignesh},
  year   = {2026},
  url    = {https://github.com/vignesh2027/TemporalMesh-Transformer}
}
```

## License

MIT
