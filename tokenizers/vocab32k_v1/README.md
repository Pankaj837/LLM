# vocab32k_v1 — 32,000-token byte-level BPE (candidate, 21 Oct 2026)

| | |
|---|---|
| File | `vocab32k.tok` (V1 format; ids 0–255 raw bytes, 31,744 merges, `<|endoftext|>` = 31999) |
| n_vocab | 32,000 (fits `uint16` token files) |
| SHA-256 | `8b00fc007ce370fe7166de8add1978e500d13c34c70c5faf20abc0bc8c023b45` |
| Status | **Candidate, verified (below). Frozen once the team lead approves; after that nobody retrains or edits it.** |

Why it exists: Ravikant's 32K tokenizer could not be delivered (no vocabulary can be trained until there is data). A
vocabulary only needs a *balanced sample* of the delivered data, not the final merged corpus, so it was trained from
one so tokenization is not blocked.

## Training sample (218 MB of text, capped at 8,000 characters per document, evenly spaced row groups)
| Source | MB | Source | MB |
|---|---|---|---|
| Gokul Biology | 25 | Ravi Code | 32 |
| Gokul Earth & Environment | 25 | Ravi Scientific Programming | 30 |
| Gokul Scientific Discovery | 25 | Ravi Scientific Q&A | 24 |
| Math (4 cleaned OpenWebMath shards, Shrot) | 43 | Chemistry text | 15 |
| CAD/STEP | 1 | | |

Reproduce: `python scripts/train_vocab.py <samples>/*.jsonl --vocab-size 32000 --out vocab32k.tok` (225 s, measured).
Trained with the repo's pure-Python trainer (`llm.bpe_trainer`), not the C++ trainer (which did not finish in 1.5 h on 50 MB).

## Verification (`python scripts/verify_vocab.py ... --cli native/tokenizer/build/encode_cli.exe`) — PASS
- Lossless round-trip on held-out documents of every domain (different row groups than the training sample).
- **C++ parity: 3003 / 3003 lines give identical token ids** from the C++ `encode_cli` (built with MSVC) and `llm.tokenizer`.
- `<|endoftext|>` typed in user text does not produce the EOS id.
- Bytes per token on held-out text (higher = fewer tokens), against OpenAI's cl100k for reference:

| Domain | this vocab | cl100k | Domain | this vocab | cl100k |
|---|---|---|---|---|---|
| CAD/STEP | 5.69 | 4.85 | Biology | 3.89 | 3.28 |
| Earth & Env. | 4.58 | 4.39 | Sci. Discovery | 4.33 | 4.08 |
| Chemistry | 3.77 | 3.87 | Math | 3.62 | 3.69 |
| Code | 3.87 | 4.36 | Sci. Programming | 3.43 | 3.98 |
| Sci. Q&A | 3.29 | 3.25 | | | |

## Limits (read before relying on it)
- The sample has **no** physics, engineering, astronomy, quant-finance or general-language text (none was accessible or
  delivered) and its code is mostly TypeScript/Java/C++. Byte-level BPE can encode anything, so nothing breaks, but those
  domains tokenize less efficiently (code is ~10% worse than cl100k above).
- Held-out text comes from the same files as the training sample, so these numbers are in-distribution.
- If the data mix changes materially, retraining takes minutes — do it **before** tokenizing the corpus, never after.
