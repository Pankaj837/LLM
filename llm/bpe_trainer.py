"""Efficient pure-Python byte-level BPE trainer that writes the team's ``.tok`` format.

Purpose: Python-side tooling and tests. The team's production vocabulary comes from the C++ trainer
(``prepare_dataset``); this produces a *format-compatible* vocabulary so the rest of the stack can be
trained and tested without the C++ binary. Do not report results obtained with it as "the team's tokenizer".

Algorithm: classic incremental BPE. Pre-token frequencies are counted once (same pretokenizer as the
encoder); each merge only re-counts the words that contain the merged pair, using a lazily-updated
max-heap. Deterministic: ties on count break on the (byte-string) pair, smallest first.

Consistency with the encoder (``BPETokenizer``: merge two neighbours when their concatenation is a
token, lowest id first): a merge whose concatenation already exists as a token is applied to the words
but does not add a duplicate vocabulary entry.

Output layout: ids 0..255 raw bytes, then merge tokens in merge order (id == rank), then specials.
"""
from __future__ import annotations

import base64
import heapq
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .tokenizer import pretokenize


def train_bpe(texts: Iterable[str], vocab_size: int, min_count: int = 2, reserved_specials: int = 1) -> List[bytes]:
    """Return the token list (index == id) of length <= ``vocab_size - reserved_specials``.

    Stops early when no pair occurs at least ``min_count`` times.
    """
    target = vocab_size - reserved_specials
    if target < 256:
        raise ValueError(f"vocab_size {vocab_size} leaves no room for the 256 byte tokens")

    freq: Counter = Counter()
    for t in texts:
        for piece in pretokenize(t):
            freq[piece] += 1

    vocab: List[bytes] = [bytes([b]) for b in range(256)]
    index: Dict[bytes, int] = {tok: i for i, tok in enumerate(vocab)}

    words: List[List[int]] = [list(w) for w in freq]           # symbols are token ids
    counts: List[int] = list(freq.values())

    pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
    where: Dict[Tuple[int, int], set] = defaultdict(set)
    for wi, w in enumerate(words):
        for a, b in zip(w, w[1:]):
            pair_count[(a, b)] += counts[wi]
            where[(a, b)].add(wi)

    def key(pair: Tuple[int, int], c: int):
        return (-c, vocab[pair[0]], vocab[pair[1]], pair)

    heap = [key(p, c) for p, c in pair_count.items() if c >= min_count]
    heapq.heapify(heap)

    while len(vocab) < target and heap:
        negc, _, _, pair = heapq.heappop(heap)
        c = pair_count.get(pair, 0)
        if c != -negc:                                          # stale entry
            if c >= min_count:
                heapq.heappush(heap, key(pair, c))
            continue
        if c < min_count:
            break

        merged = vocab[pair[0]] + vocab[pair[1]]
        new_id = index.get(merged)
        if new_id is None:
            new_id = len(vocab)
            vocab.append(merged)
            index[merged] = new_id

        touched = set()
        for wi in list(where[pair]):
            w, f = words[wi], counts[wi]
            if pair[0] not in w:                                # index entries may be stale
                continue
            for a, b in zip(w, w[1:]):                          # remove this word's pair contributions
                pair_count[(a, b)] -= f
                touched.add((a, b))
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for a, b in zip(out, out[1:]):                      # add the new contributions
                pair_count[(a, b)] += f
                where[(a, b)].add(wi)
                touched.add((a, b))
        del where[pair]
        pair_count.pop(pair, None)
        for p in touched:
            cnt = pair_count.get(p, 0)
            if cnt >= min_count:
                heapq.heappush(heap, key(p, cnt))
    return vocab


def write_tok(vocab: Sequence[bytes], path: str, specials: Sequence[str] = ("<|endoftext|>",)) -> int:
    """Write the ``.tok`` file (specials get the ids right after the merge vocabulary). Returns n_vocab."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# tiktoken-style-vocab\nV1\n{len(vocab)}\n")
        for i, tok in enumerate(vocab):
            f.write(f"{base64.b64encode(tok).decode()} {i}\n")
        f.write(f"SPECIALS\n{len(specials)}\n")
        for j, s in enumerate(specials):
            f.write(f"{base64.b64encode(s.encode('utf-8')).decode()} {len(vocab) + j}\n")
    return len(vocab) + len(specials)
