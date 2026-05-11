"""
tokenizer.py — thin wrapper around HuggingFace tokenizer for TMT.
"""
from __future__ import annotations

from typing import List, Union

from transformers import AutoTokenizer


class TMTTokenizer:
    """Wraps a HuggingFace tokenizer with a consistent TMT interface."""

    def __init__(self, name: str = "gpt2") -> None:
        self.hf = AutoTokenizer.from_pretrained(name)
        if self.hf.pad_token is None:
            self.hf.add_special_tokens({"pad_token": "[PAD]"})
        self.vocab_size = len(self.hf)

    def encode(self, text: Union[str, List[str]], max_length: int = 1024) -> dict:
        return self.hf(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    def decode(self, token_ids) -> str:
        return self.hf.decode(token_ids, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f"TMTTokenizer(vocab={self.vocab_size})"
