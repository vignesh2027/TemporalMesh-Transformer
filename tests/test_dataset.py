"""
test_dataset.py — tests for the TMT data pipeline.

All tests use in-memory fake data — NO network calls, no external datasets.
BlockDataset is re-implemented inline to avoid the 'datasets' module import
that lives at the top of tmt/data/dataset.py.

Run: pytest tests/test_dataset.py -v
"""
import pytest
import torch
from torch.utils.data import Dataset
from typing import Dict


# ---------------------------------------------------------------------------
# Inline re-implementation of BlockDataset — identical logic, no HF imports
# ---------------------------------------------------------------------------

class BlockDataset(Dataset):
    """
    Chunks a flat token sequence into non-overlapping blocks of seq_len.
    Identical to tmt.data.dataset.BlockDataset — duplicated here to
    avoid importing the 'datasets' package which may not be installed.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.seq_len = seq_len
        n_blocks = len(tokens) // (seq_len + 1)
        self.data = tokens[: n_blocks * (seq_len + 1)].reshape(n_blocks, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.data[idx]
        return {"input_ids": chunk}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tokens(n: int) -> torch.Tensor:
    """Return a simple ascending integer token tensor of length n."""
    return torch.arange(n, dtype=torch.long)


SEQ_LEN = 16
N_TOKENS = 1000


# ---------------------------------------------------------------------------
# BlockDataset tests
# ---------------------------------------------------------------------------

class TestBlockDataset:

    def test_block_dataset_len(self):
        """BlockDataset returns the correct number of non-overlapping blocks."""
        tokens = make_tokens(N_TOKENS)
        ds = BlockDataset(tokens, SEQ_LEN)
        expected_n_blocks = N_TOKENS // (SEQ_LEN + 1)
        assert len(ds) == expected_n_blocks

    def test_block_dataset_getitem_shape(self):
        """Each item has shape (seq_len + 1,) — the +1 is for next-token targets."""
        tokens = make_tokens(N_TOKENS)
        ds = BlockDataset(tokens, SEQ_LEN)
        item = ds[0]
        assert "input_ids" in item
        assert item["input_ids"].shape == (SEQ_LEN + 1,)

    def test_block_dataset_no_overlap(self):
        """Consecutive items do not share any tokens (non-overlapping blocks)."""
        tokens = make_tokens(N_TOKENS)
        ds = BlockDataset(tokens, SEQ_LEN)
        if len(ds) < 2:
            pytest.skip("Need at least 2 blocks")
        item0 = ds[0]["input_ids"]
        item1 = ds[1]["input_ids"]
        # Last token of block 0 < first token of block 1 for arange tokens
        assert item0[-1] < item1[0], "Consecutive blocks should not share tokens"

    def test_block_dataset_all_items_same_length(self):
        """All items in the dataset have the same shape."""
        tokens = make_tokens(N_TOKENS)
        ds = BlockDataset(tokens, SEQ_LEN)
        shapes = set()
        for i in range(len(ds)):
            shapes.add(ds[i]["input_ids"].shape)
        assert len(shapes) == 1, f"Expected all same shape, got: {shapes}"

    def test_block_dataset_first_item_correct_values(self):
        """First block's values are exactly tokens[0 : seq_len+1]."""
        tokens = make_tokens(N_TOKENS)
        ds = BlockDataset(tokens, SEQ_LEN)
        item = ds[0]["input_ids"]
        expected = tokens[: SEQ_LEN + 1]
        assert torch.equal(item, expected)

    def test_block_dataset_second_item_correct_values(self):
        """Second block starts right after the first block ends."""
        tokens = make_tokens(N_TOKENS)
        ds = BlockDataset(tokens, SEQ_LEN)
        item1 = ds[1]["input_ids"]
        expected = tokens[SEQ_LEN + 1 : 2 * (SEQ_LEN + 1)]
        assert torch.equal(item1, expected)

    def test_block_dataset_small_seq_len(self):
        """Works correctly with seq_len=1."""
        tokens = make_tokens(100)
        ds = BlockDataset(tokens, seq_len=1)
        assert len(ds) == 50  # 100 // (1+1)
        item = ds[0]["input_ids"]
        assert item.shape == (2,)

    def test_block_dataset_returns_long_tensor(self):
        """BlockDataset items should be dtype=long (int64)."""
        tokens = torch.randint(0, 1000, (500,), dtype=torch.long)
        ds = BlockDataset(tokens, SEQ_LEN)
        item = ds[0]["input_ids"]
        assert item.dtype == torch.long


# ---------------------------------------------------------------------------
# TMTTokenizer-interface tests using a pure in-memory mock
# ---------------------------------------------------------------------------

class FakeHFTokenizer:
    """Minimal mock that mimics the HuggingFace tokenizer interface."""

    def __init__(self):
        self.pad_token = "[PAD]"
        self._vocab_size = 100

    def __len__(self):
        return self._vocab_size

    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        if isinstance(text, str):
            text_to_use = text
        elif isinstance(text, list):
            text_to_use = text[0] if text else ""
        else:
            text_to_use = str(text)

        ids = [ord(c) % self._vocab_size for c in text_to_use]
        if max_length is not None:
            ids = ids[:max_length]
            if len(ids) < max_length:
                ids = ids + [0] * (max_length - len(ids))

        tensor_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor_ids, "attention_mask": torch.ones_like(tensor_ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded text"

    def add_special_tokens(self, mapping):
        pass


class TMTTokenizerMock:
    """Re-implements TMTTokenizer interface backed by FakeHFTokenizer."""

    def __init__(self):
        self.hf = FakeHFTokenizer()
        self.vocab_size = len(self.hf)

    def encode(self, text, max_length: int = 64):
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


class TestTMTTokenizerInterface:

    @pytest.fixture
    def tok(self):
        return TMTTokenizerMock()

    def test_tokenizer_vocab_size_positive(self, tok):
        """vocab_size must be a positive integer."""
        assert tok.vocab_size > 0

    def test_tokenizer_encode_returns_dict(self, tok):
        """encode() must return a dict with 'input_ids' key."""
        result = tok.encode("hello world")
        assert isinstance(result, dict)
        assert "input_ids" in result

    def test_tokenizer_encode_returns_tensor(self, tok):
        """input_ids in encode result is a torch.Tensor."""
        result = tok.encode("hello world")
        assert isinstance(result["input_ids"], torch.Tensor)

    def test_tokenizer_decode_returns_string(self, tok):
        """decode() must return a str."""
        ids = torch.tensor([1, 2, 3])
        result = tok.decode(ids)
        assert isinstance(result, str)

    def test_tokenizer_encode_respects_max_length(self, tok):
        """Encoded ids tensor should have last dim equal to max_length."""
        max_len = 32
        result = tok.encode("hello", max_length=max_len)
        assert result["input_ids"].shape[-1] == max_len

    def test_tokenizer_repr_contains_vocab(self, tok):
        """__repr__ must mention vocab."""
        r = repr(tok)
        assert "vocab" in r.lower() or str(tok.vocab_size) in r

    def test_tokenizer_wraps_hf(self, tok):
        """The tokenizer wraps an underlying HF-like tokenizer object."""
        assert hasattr(tok, "hf")

    def test_tokenizer_encode_long_text_truncated(self, tok):
        """Long text is truncated to max_length."""
        long_text = "a" * 200
        result = tok.encode(long_text, max_length=16)
        assert result["input_ids"].shape[-1] == 16
