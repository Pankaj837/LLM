"""
Byte-Pair Encoding (BPE) Tokenizer for the LLM pipeline.

Fully compatible with the C++ BPE Tokenizer library and .tok vocabulary files:
- Parses V1 tiktoken-style .tok files (Base64-encoded token bytes + specials)
- Implements exact pretokenization matching C++ utils.cpp (contractions, whitespace prefixing, unicode)
- Implements lowest-rank pair merging matching C++ encoder.cpp
- Encodes and decodes arbitrary text with special token support (<|endoftext|>)
- Backward compatible with SimpleTokenizer for shape verification
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Dict, List, Optional, Set


# =======================================================================
# Pre-tokenization (pure Python port of tokenizer/src/utils.cpp)
# =======================================================================

CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _char_class(b: int) -> int:
    """0: space, 1: letter, 2: digit, 3: other."""
    if b in (32, 9, 10, 13, 12, 11):  # ' ', '\t', '\n', '\r', '\f', '\v'
        return 0
    if (65 <= b <= 90) or (97 <= b <= 122) or (b >= 128):  # ASCII letters or UTF-8 continuation/lead
        return 1
    if 48 <= b <= 57:  # '0'-'9'
        return 2
    return 3


def pretokenize(text: str) -> List[bytes]:
    """Splits raw text into byte chunks matching C++ tok::pretokenize."""
    raw_bytes = text.encode("utf-8")
    n = len(raw_bytes)
    i = 0
    chunks: List[bytes] = []

    while i < n:
        # 1. Contractions
        if raw_bytes[i] == 39:  # ord("'") == 39
            matched_len = 0
            for c in CONTRACTIONS:
                cb = c.encode("utf-8")
                if raw_bytes[i : i + len(cb)] == cb:
                    matched_len = len(cb)
                    break
            if matched_len > 0:
                chunks.append(raw_bytes[i : i + matched_len])
                i += matched_len
                continue

        cls = _char_class(raw_bytes[i])

        if cls == 0:  # Space
            j = i
            while j < n and _char_class(raw_bytes[j]) == 0:
                j += 1
            run_len = j - i

            if j < n:
                if run_len > 1:
                    chunks.append(raw_bytes[i : i + run_len - 1])
                next_cls = _char_class(raw_bytes[j])
                k = j + 1
                while k < n and _char_class(raw_bytes[k]) == next_cls:
                    k += 1
                chunks.append(raw_bytes[j - 1 : k])
                i = k
            else:
                chunks.append(raw_bytes[i : j])
                i = j
        else:
            j = i + 1
            while j < n and _char_class(raw_bytes[j]) == cls:
                j += 1
            chunks.append(raw_bytes[i : j])
            i = j

    return chunks


# =======================================================================
# BPE Tokenizer
# =======================================================================

class BPETokenizer:
    """Byte-level BPE tokenizer with .tok persistence and C++ parity."""

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab_size: int = 8000,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}
        self._piece_cache: Dict[bytes, List[int]] = {}

        if vocab_path and os.path.isfile(vocab_path):
            self.load(vocab_path)
        else:
            # Initialize with standard 256 byte vocabulary
            self._init_byte_vocab()
            specials = special_tokens or {"<|endoftext|>": 256}
            for token_str, token_id in specials.items():
                self.add_special_token(token_str, token_id)

    @classmethod
    def from_file(cls, path: str, vocab_size: Optional[int] = None) -> "BPETokenizer":
        """Load a ``.tok`` vocabulary. Raises FileNotFoundError instead of silently falling
        back to a byte-level vocabulary. ``vocab_size`` defaults to ``n_vocab``."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"vocab file not found: {path}")
        tok = cls(vocab_path=path)
        tok.vocab_size = vocab_size if vocab_size is not None else tok.n_vocab
        return tok

    @staticmethod
    def file_sha256(path: str) -> str:
        """Identity of a vocabulary file (stored in checkpoints to detect mismatches)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def _init_byte_vocab(self):
        """Initializes the base 256 byte vocabulary."""
        for b in range(256):
            byte_token = bytes([b])
            self.token_to_id[byte_token] = b
            self.id_to_token[b] = byte_token

    def add_special_token(self, token_str: str, token_id: Optional[int] = None):
        """Registers a special token."""
        if token_id is None:
            token_id = len(self.token_to_id) + len(self.special_to_id)
        self.special_to_id[token_str] = token_id
        self.id_to_special[token_id] = token_str

    def load(self, path: str):
        """Loads a .tok vocabulary file (matching tok::Vocabulary::load)."""
        self.token_to_id.clear()
        self.id_to_token.clear()
        self.special_to_id.clear()
        self.id_to_special.clear()
        self._piece_cache.clear()

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"Empty vocab file: {path}")

        idx = 0
        if lines[idx] == "# tiktoken-style-vocab":
            idx += 1
        if idx < len(lines) and lines[idx] == "V1":
            idx += 1

        merge_count = int(lines[idx])
        idx += 1

        for _ in range(merge_count):
            parts = lines[idx].split()
            b64_str, tid = parts[0], int(parts[1])
            token_bytes = base64.b64decode(b64_str)
            self.token_to_id[token_bytes] = tid
            self.id_to_token[tid] = token_bytes
            idx += 1

        if idx < len(lines) and lines[idx] == "SPECIALS":
            idx += 1
            special_count = int(lines[idx])
            idx += 1
            for _ in range(special_count):
                parts = lines[idx].split()
                b64_str, tid = parts[0], int(parts[1])
                special_text = base64.b64decode(b64_str).decode("utf-8", errors="replace")
                self.special_to_id[special_text] = tid
                self.id_to_special[tid] = special_text
                idx += 1

    def save(self, path: str):
        """Saves current vocabulary to a .tok file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("# tiktoken-style-vocab\nV1\n")
            f.write(f"{len(self.token_to_id)}\n")
            for tid in range(len(self.token_to_id)):
                token_bytes = self.id_to_token[tid]
                b64 = base64.b64encode(token_bytes).decode("ascii")
                f.write(f"{b64} {tid}\n")
            f.write("SPECIALS\n")
            f.write(f"{len(self.special_to_id)}\n")
            for text, tid in self.special_to_id.items():
                b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
                f.write(f"{b64} {tid}\n")

    def _bpe_encode_piece(self, piece: bytes) -> List[int]:
        """Encodes a single byte piece by merging lowest-ranked pairs."""
        if not piece:
            return []
        cached = self._piece_cache.get(piece)
        if cached is not None:
            return list(cached)

        parts: List[bytes] = [bytes([b]) for b in piece]

        while len(parts) > 1:
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(parts) - 1):
                merged = parts[i] + parts[i + 1]
                rank = self.token_to_id.get(merged)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
            parts.pop(best_idx + 1)

        ids = [self.token_to_id[p] for p in parts]
        self._piece_cache[piece] = ids
        return list(ids)

    def encode(
        self, text: str, allowed_special: Optional[Set[str]] = None
    ) -> List[int]:
        """Encodes input string to token IDs."""
        if allowed_special is None:
            allowed_special = set(self.special_to_id.keys())

        if not allowed_special:
            ids: List[int] = []
            for chunk in pretokenize(text):
                ids.extend(self._bpe_encode_piece(chunk))
            return ids

        ids: List[int] = []
        pos = 0
        n = len(text)

        while pos < n:
            best_at = -1
            best_special = ""

            for special in allowed_special:
                if not special:
                    continue
                found = text.find(special, pos)
                if found != -1:
                    if best_at == -1 or found < best_at or (found == best_at and len(special) > len(best_special)):
                        best_at = found
                        best_special = special

            if best_at == -1:
                # Encode remaining text plainly
                for chunk in pretokenize(text[pos:]):
                    ids.extend(self._bpe_encode_piece(chunk))
                break

            if best_at > pos:
                for chunk in pretokenize(text[pos:best_at]):
                    ids.extend(self._bpe_encode_piece(chunk))

            ids.append(self.special_to_id[best_special])
            pos = best_at + len(best_special)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back to a string."""
        byte_chunks: List[bytes] = []
        for tid in ids:
            if tid in self.id_to_special:
                byte_chunks.append(self.id_to_special[tid].encode("utf-8"))
            elif tid in self.id_to_token:
                byte_chunks.append(self.id_to_token[tid])
            elif 0 <= tid <= 255:
                byte_chunks.append(bytes([tid]))
            else:
                # Unknown ID fallback
                byte_chunks.append(f"<unk:{tid}>".encode("utf-8"))

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    @property
    def eot_token_id(self) -> int:
        return self.special_to_id.get("<|endoftext|>", 256)

    @property
    def eos_id(self) -> int:
        """Alias so the reward-model code and the LM pipeline share one API."""
        return self.eot_token_id

    @property
    def n_vocab(self) -> int:
        """Embedding rows needed to represent every id this tokenizer can emit."""
        top = max(list(self.id_to_token) + list(self.id_to_special), default=-1)
        return top + 1


# =======================================================================
# Backward Compatibility Wrappers
# =======================================================================

class SimpleTokenizer(BPETokenizer):
    """Pure-Python byte-level tokenizer for tests (retains backwards compatibility)."""
    pass


class SubprocessTokenizer:
    """Placeholder for future standalone binary CLI invocation."""
    def __init__(self, binary_path: str = "", vocab_path: Optional[str] = None):
        self.binary_path = binary_path
        self.vocab_path = vocab_path

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("Use BPETokenizer for in-process BPE encoding")

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError("Use BPETokenizer for in-process BPE decoding")
