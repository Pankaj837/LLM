# Tokenizers

| Directory | What | SHA-256 | Status |
|---|---|---|---|
| `vocab32k_v1/` | 32K byte-level BPE; number chunks learned from the data | `50a0910f2f56f9ffd4f50e2d16bd4edfee7a021e07800d0e3e4cdb9e6df32da9` | candidate |
| `vocab32k_digits_v1/` | same sample and trainer, **no token with two adjacent digits** (numbers digit by digit) | `e6bd30d24b67d85b880ad3d6a282303887e5bd94d39bcc8f27b9587ea2cb383d` | candidate — **recommended** |

Both were trained on the same 227 MB balanced sample of the delivered data, both pass the same verification (lossless
round-trip, C++ encoder gives identical ids on 3003/3003 lines, EOS cannot be forged by text; `scripts/verify_vocab.py`).

## How to choose
The only difference is how numbers are written. The data is mathematics, multi-step calculation, physics, quant finance, CAD
geometry and code, so consistent digits are worth ~5–6% more tokens (per-domain numbers in `vocab32k_digits_v1/README.md`).
If numbers-only text (finance records, generated arithmetic) is judged unimportant and token cost matters more than arithmetic
regularity, take `vocab32k_v1`. The two are **not interchangeable**: choose before tokenizing anything for a real run.

## How to freeze
1. The team lead states the choice in the group; the choice is written here as `FROZEN.md` (directory, SHA-256, date, who approved).
2. Every `manifest.json` written by `tokenize_parquet.py` / `tokenize_text.py` carries `tokenizer_sha256`; it must equal the frozen
   hash (`scripts/check_bins.py --vocab` checks the vocabulary size and EOS id, the manifest check is part of the pre-launch list).
3. A frozen vocabulary is never edited or retrained. A different vocabulary means a new directory and re-tokenizing everything.
