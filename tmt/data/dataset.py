"""
dataset.py — loads wikitext-2 or tinystories and chunks into fixed-length blocks.
"""
from __future__ import annotations

from typing import Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class BlockDataset(Dataset):
    """Chunks a flat token sequence into non-overlapping blocks of seq_len."""

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.seq_len = seq_len
        n_blocks = len(tokens) // (seq_len + 1)
        # +1 so we can shift for next-token targets
        self.data = tokens[: n_blocks * (seq_len + 1)].reshape(n_blocks, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.data[idx]
        return {"input_ids": chunk}


def load_text_dataset(
    name: str = "wikitext-2",
    seq_len: int = 256,
    batch_size: int = 16,
    tokenizer_name: str = "gpt2",
) -> Dict[str, DataLoader]:
    """
    Returns {"train": DataLoader, "validation": DataLoader}.
    Supported names: "wikitext-2", "tinystories".
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    if name == "wikitext-2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    elif name == "tinystories":
        raw = load_dataset("roneneldan/TinyStories")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    def tokenize(examples):
        return tok(examples["text"], truncation=False, return_attention_mask=False)

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)

    loaders = {}
    for split in ("train", "validation"):
        if split not in tokenized:
            continue
        all_ids = []
        for sample in tokenized[split]["input_ids"]:
            all_ids.extend(sample)
        flat = torch.tensor(all_ids, dtype=torch.long)
        ds = BlockDataset(flat, seq_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
        )

    return loaders
