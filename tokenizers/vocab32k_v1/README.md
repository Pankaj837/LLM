# vocab32k_v1 — 32,000-token byte-level BPE (candidate, 21 Oct 2026)

> There are **two candidates**: this one (digit chunks learned from the data, e.g. `134945` → ` 134`,`9`,`45`) and
> `../vocab32k_digits_v1/` (every number digit by digit, ~5–6% more tokens). See `../README.md` for how to choose.

| | |
|---|---|
| File | `vocab32k.tok` (V1 format; ids 0–255 raw bytes, 31,743 merges, `<|endoftext|>` = 31999) |
| n_vocab | 32,000 (fits `uint16` token files) |
| SHA-256 | `50a0910f2f56f9ffd4f50e2d16bd4edfee7a021e07800d0e3e4cdb9e6df32da9` |
| Status | **Candidate, verified (below). Frozen once the team lead approves; after that nobody retrains or edits it.** |

Why it exists: Ravikant's 32K tokenizer could not be delivered (no vocabulary can be trained until there is data). A
vocabulary only needs a *balanced sample* of the delivered data, not the final merged corpus, so it was trained from
one so tokenization is not blocked.

## Training sample (227 MB of text, capped at 8,000 characters per document, evenly spaced row groups)
| Source | MB | Source | MB |
|---|---|---|---|
| Gokul Biology | 25 | Ravi Code | 32 |
| Gokul Earth & Environment | 25 | Ravi Scientific Programming | 30 |
| Gokul Scientific Discovery | 25 | Ravi Scientific Q&A | 24 |
| Math (4 cleaned OpenWebMath shards, Shrot) | 43 | Chemistry text | 15 |
| CAD/STEP (CadQuery code, STEP snippets, Q&A) | 10 | | |

Reproduce: `python scripts/train_vocab.py <samples>/*.jsonl --vocab-size 32000 --out vocab32k.tok` (329 s, measured).
Trained with the repo's pure-Python trainer (`llm.bpe_trainer`), not the C++ trainer (which did not finish in 1.5 h on 50 MB).
(An earlier candidate, SHA-256 `8b00fc00…c023b45`, used only the CAD *descriptions*, so it had never seen CadQuery/STEP text;
it was replaced before anything was tokenized with it.)

## Verification (`python scripts/verify_vocab.py ... --cli native/tokenizer/build/encode_cli.exe`) — PASS
- Lossless round-trip on held-out documents of every domain (different row groups than the training sample).
- **C++ parity: 3003 / 3003 lines give identical token ids** from the C++ `encode_cli` (built with MSVC) and `llm.tokenizer`.
- `<|endoftext|>` typed in user text does not produce the EOS id.
- Bytes per token on held-out text (higher = fewer tokens), against OpenAI's cl100k for reference:

| Domain | this vocab | cl100k |
|---|---|---|
| Math | 3.76 | 3.88 |
| Code | 3.73 | 4.18 |
| Scientific Programming | 3.38 | 3.93 |
| Scientific Q&A | 3.25 | 3.25 |
| CAD/STEP | 3.27 | 3.32 |
| Chemistry | 3.94 | 3.93 |
| Earth & Environment | 4.55 | 4.38 |
| Scientific Discovery | 4.29 | 4.08 |
| Biology | 3.84 | 3.22 |
| Astronomy | 4.54 | 4.59 |
| Multi-step (synthetic arithmetic) | 2.86 | 2.55 |
| Finance (numbers only) | 2.58 | 2.65 |

## Limits (read before relying on it)
- The sample has **no** physics, engineering, astronomy, quant-finance or general-language text (none was accessible or
  delivered) and its code is mostly TypeScript/Java/C++. Byte-level BPE can encode anything, so nothing breaks, but those
  domains tokenize less efficiently (code is ~11% worse than cl100k above).
- Held-out text comes from the same files as the training sample, so these numbers are in-distribution.
- If the data mix changes materially, retraining takes minutes — do it **before** tokenizing the corpus, never after.
