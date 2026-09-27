"""Token dataset in the tokenizer team's ``.bin`` format, and batching for training.

Format (from the tokenizer report): a 36-byte little-endian header followed by ``uint16``
token ids.

    struct "<IIIIQIQ":  magic (0x544F4B31, "TOK1"), version, vocab_size, eos_id,
                        total_tokens (uint64), split, reserved (uint64)

Compatibility rules enforced here (from the same report):
  * the model vocabulary size must equal the tokenizer's (``check_compatible``);
  * ids are stored as uint16, so every id must be < 65536;
  * the header is 36 bytes, not 32.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

MAGIC = 0x544F4B31
HEADER_FMT = "<IIIIQIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)      # 36
assert HEADER_SIZE == 36
MAX_ID = 65535
SPLIT_TRAIN, SPLIT_VAL = 0, 1                  # the report leaves the meaning of `split` open; documented convention


@dataclass(frozen=True)
class TokHeader:
    magic: int
    version: int
    vocab_size: int
    eos_id: int
    total_tokens: int
    split: int
    reserved: int = 0


def read_header(path: str) -> TokHeader:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"{path}: file shorter than the {HEADER_SIZE}-byte header")
    h = TokHeader(*struct.unpack(HEADER_FMT, raw))
    if h.magic != MAGIC:
        raise ValueError(f"{path}: bad magic 0x{h.magic:08X} (expected 0x{MAGIC:08X} 'TOK1')")
    expected = HEADER_SIZE + 2 * h.total_tokens
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(f"{path}: size {actual} != header-declared {expected} "
                         f"(total_tokens={h.total_tokens}); truncated or corrupt")
    return h


def read_tokens(path: str, mmap: bool = True) -> Tuple[TokHeader, np.ndarray]:
    """(header, uint16 array). Memory-mapped by default so multi-GB corpora do not load into RAM."""
    h = read_header(path)
    if h.total_tokens == 0:
        return h, np.zeros(0, dtype="<u2")
    if mmap:
        return h, np.memmap(path, dtype="<u2", mode="r", offset=HEADER_SIZE, shape=(h.total_tokens,))
    return h, np.fromfile(path, dtype="<u2", count=h.total_tokens, offset=HEADER_SIZE)


def write_bin(path: str, ids: Union[np.ndarray, Iterable[int]], vocab_size: int, eos_id: int,
              split: int = SPLIT_TRAIN, version: int = 1) -> TokHeader:
    """Write ids in the ``.bin`` format (also what the C++ tool produces; used for tests and Python-side tooling)."""
    arr = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > MAX_ID):
        raise ValueError(f"token ids must be in [0, {MAX_ID}] (stored as uint16); got [{arr.min()}, {arr.max()}]")
    if arr.size and arr.max() >= vocab_size:
        raise ValueError(f"token id {arr.max()} >= vocab_size {vocab_size}")
    if not 0 <= eos_id < vocab_size:
        raise ValueError(f"eos_id {eos_id} outside [0, {vocab_size})")
    header = TokHeader(MAGIC, version, vocab_size, eos_id, int(arr.size), split, 0)
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, *(getattr(header, k) for k in
                                          ("magic", "version", "vocab_size", "eos_id", "total_tokens", "split", "reserved"))))
        f.write(arr.astype("<u2").tobytes())
    return header


def check_compatible(header: TokHeader, model_vocab_size: int) -> None:
    """Raise if the dataset was tokenised with a vocabulary that does not fit the model."""
    if header.vocab_size != model_vocab_size:
        raise ValueError(f"dataset vocab_size {header.vocab_size} != model vocab_size {model_vocab_size}: "
                         "use the same vocabulary the model was configured for (do not retrain a different tokenizer)")


class TokenDataset:
    """Random fixed-length windows over a token array; targets are the inputs shifted by one."""

    def __init__(self, path: str, seq_len: int, model_vocab_size: Optional[int] = None, mmap: bool = True):
        self.header, self.tokens = read_tokens(path, mmap=mmap)
        if model_vocab_size is not None:
            check_compatible(self.header, model_vocab_size)
        if len(self.tokens) < seq_len + 1:
            raise ValueError(f"{path}: {len(self.tokens)} tokens is too few for seq_len={seq_len} (+1 target)")
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Number of distinct window start positions."""
        return len(self.tokens) - self.seq_len

    def window(self, start: int) -> Tuple[np.ndarray, np.ndarray]:
        chunk = np.asarray(self.tokens[start:start + self.seq_len + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]

    def get_batch(self, batch_size: int, generator: Optional[torch.Generator] = None,
                  device: Union[str, torch.device] = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(self), (batch_size,), generator=generator).tolist()
        xs, ys = zip(*(self.window(s) for s in starts))
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        return x, y

    def sequential_batches(self, batch_size: int, max_batches: Optional[int] = None,
                           device: Union[str, torch.device] = "cpu") -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping windows in order (deterministic; used for validation)."""
        starts = list(range(0, len(self), self.seq_len))
        n = 0
        for i in range(0, len(starts), batch_size):
            pairs = [self.window(s) for s in starts[i:i + batch_size]]
            xs, ys = zip(*pairs)
            yield (torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device))
            n += 1
            if max_batches is not None and n >= max_batches:
                return


# ── building datasets from text ─────────────────────────────────────────────

def document_split(n_docs: int, val_frac: float = 0.05, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Split by DOCUMENT (never by chunk) so no validation text also appears in training.
    Returns (train_indices, val_indices), both sorted; at least one validation document when n_docs >= 2."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    order = np.random.RandomState(seed).permutation(n_docs)
    n_val = max(1, int(round(n_docs * val_frac))) if n_docs >= 2 else 0
    return sorted(order[n_val:].tolist()), sorted(order[:n_val].tolist())


def encode_documents(tokenizer, docs: Iterable[str], eos_id: Optional[int] = None) -> np.ndarray:
    """Tokenise documents and append EOS after each one (document boundaries stay learnable).
    User text is encoded WITHOUT special tokens so a document containing '<|endoftext|>' cannot forge a boundary."""
    eos = tokenizer.eos_id if eos_id is None else eos_id
    out: List[int] = []
    for d in docs:
        out.extend(tokenizer.encode(d, allowed_special=set()))
        out.append(eos)
    return np.asarray(out, dtype=np.int64)


class NearDuplicateFilter:
    """Drops documents that mostly repeat text already seen (exact copies, re-published chapters,
    the same article with a different header/footer).

    A document is described by a content-defined sample (~1/``stride``) of 64-bit hashes of its ``k``-word shingles.
    It is a near-duplicate when at least ``threshold`` of its hashes were already seen. Hashes of
    KEPT documents are remembered, so the filter is order dependent (first occurrence wins) and
    deterministic. Documents with fewer than ``k`` words are only compared for exact equality.
    """

    def __init__(self, k: int = 12, stride: int = 3, threshold: float = 0.3):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.k, self.stride, self.threshold = k, stride, threshold
        self._seen: set = set()
        self._exact: set = set()

    @staticmethod
    def _h(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")

    def shingles(self, text: str) -> set:
        """Content-defined sample of the document's ``k``-word shingles: a shingle is kept when its hash is
        divisible by ``stride`` (~1/stride of them). Unlike sampling every stride-th position, this survives
        inserted/removed header words, which shift every position."""
        w = text.split()
        hs = (self._h(" ".join(w[i:i + self.k])) for i in range(max(len(w) - self.k + 1, 0)))
        return {h for h in hs if h % self.stride == 0}

    def is_duplicate(self, text: str) -> bool:
        """True if ``text`` should be dropped. Otherwise it is remembered and False is returned."""
        exact = self._h(text)
        sh = self.shingles(text)
        dup = exact in self._exact or (bool(sh) and len(sh & self._seen) / len(sh) >= self.threshold)
        if not dup:
            self._exact.add(exact)
            self._seen |= sh
        return dup
