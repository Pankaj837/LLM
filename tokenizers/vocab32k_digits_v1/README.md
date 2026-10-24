# vocab32k_digits_v1 — 32,000-token byte-level BPE, numbers encoded digit by digit (candidate, 24 Oct 2026)

Same training sample (227 MB, 9 domains), same trainer and same format as `../vocab32k_v1/`; the only difference is that **no
token may contain two adjacent digits** (`scripts/train_vocab.py --split-digits`). The encoder is unchanged: it only merges
what the vocabulary contains, so every number becomes single digits, always in the same way.

| | |
|---|---|
| File | `vocab32k.tok` (V1 format, `<|endoftext|>` = 31999, n_vocab 32,000, fits `uint16`) |
| SHA-256 | `e6bd30d24b67d85b880ad3d6a282303887e5bd94d39bcc8f27b9587ea2cb383d` |
| Status | **Candidate, verified; one of two options — the team lead chooses, then it is frozen.** Recommendation: this one |

## Why
In `vocab32k_v1` the number `134945` is encoded as ` 134`, `9`, `45`; `1234567890123` as ` 123`, `45`, `678`, `90`, `123`.
The split depends on which chunks happened to be frequent in the sample, so the same digit has a different role in
different numbers. For a model whose domains are mathematics, multi-step calculation, physics, quant finance and CAD
geometry, consistent place value matters for learning arithmetic and numeric reasoning. Digit-level number tokens are the
choice made by LLaMA, PaLM and Qwen-style models. Here, ` 134945` is ` 1`,`3`,`4`,`9`,`4`,`5`.

## What it costs (bytes per token on the same held-out documents, deterministic sampling; higher = fewer tokens = cheaper)
| Domain | `vocab32k_v1` | **digits** | change | cl100k |
|---|---|---|---|---|
| Math | 3.76 | 3.63 | −3% | 3.88 |
| Code | 3.73 | 3.65 | −2% | 4.18 |
| Scientific Programming | 3.38 | 3.34 | −1% | 3.93 |
| Scientific Q&A | 3.25 | 3.15 | −3% | 3.25 |
| CAD/STEP | 3.27 | 3.20 | −2% | 3.32 |
| Chemistry | 3.94 | 3.87 | −2% | 3.93 |
| Earth & Environment | 4.55 | 4.36 | −4% | 4.38 |
| Scientific Discovery | 4.29 | 4.06 | −5% | 4.08 |
| Biology | 3.84 | 3.27 | −15% | 3.22 |
| Astronomy | 4.54 | 4.43 | −2% | 4.59 |
| Multi-step (synthetic arithmetic) | 2.86 | 2.22 | −22% | 2.55 |
| Finance (numbers only) | 2.58 | 1.90 | −26% | 2.65 |

Weighted over the proposed mix (`docs/TRAINING_PLAN_1p5B.md` §3) it is about 5–6% more tokens for the same text, i.e.
about 5–6% less text per GPU-hour; numbers-only data (finance, generated arithmetic) and Biology (identifiers) cost
the most. That is the price of regular numbers.

## Verification (`scripts/verify_vocab.py ... --cli native/tokenizer/build/encode_cli.exe`) — PASS
Lossless round-trip on held-out documents of every domain; **C++ parity: 3003 / 3003 lines identical**;
`<|endoftext|>` typed in user text does not produce the EOS id; no token contains two adjacent digits (0 of 31,999).

## Limits
Same as `vocab32k_v1`: no physics, engineering, astronomy-heavy or general-language text in the sample; held-out text comes
from the same files as the training sample. If this candidate is chosen, anything tokenized with `vocab32k_v1` (for example
the smoke-test data) must be tokenized again — the two vocabularies are not interchangeable.
