# Tokenizers

**Decision (Raju, 24–25 Oct): `vocab32k_v2` is frozen** — see `FROZEN.md`. The text below explains how the two candidates differ and is kept for the later digits evaluation.

| Directory | What | SHA-256 | Status |
|---|---|---|---|
| `vocab32k_v2/` | 32K byte-level BPE; number chunks learned from the data | `10b3c5aecf6b81746bf86f8c18f8c576bf716a3620e7632e8fc2684006b51cdc` | **FROZEN 25 Oct** (`FROZEN.md`) |
| `vocab32k_digits_v2/` | same sample and trainer, **no token with two adjacent digits** (numbers digit by digit) | `b2e45a916f6b01868d36ca181887021fcee36e8d91f1d4835745553d3397aa87` | not frozen: to be evaluated separately, only if digit-level numerical reasoning shows a clear need (Raju) |

Both were trained on the same 227 MB balanced sample of the delivered data, both pass the same verification (lossless
round-trip, C++ encoder gives identical ids on 3003/3003 lines, EOS cannot be forged by text; `scripts/verify_vocab.py`).
Both have the same 64 special tokens (ids 31,936–31,999; EOS `<|endoftext|>` = 31,936): `<|pad|>`, `<|bos|>`, `<|system|>`, `<|user|>`,
`<|assistant|>`, `<|end_of_turn|>`, `<|tool_call|>`, `<|tool_result|>` and 55 reserved slots, so chat / tool-use / structured-output
fine-tuning later never needs to resize a frozen vocabulary. Pretraining uses only EOS. Names can still change before the freeze.

## How to choose
The only difference is how numbers are written. The data is mathematics, multi-step calculation, physics, quant finance, CAD
geometry and code, so consistent digits are worth ~5–6% more tokens (per-domain numbers in `vocab32k_digits_v2/README.md`).
If numbers-only text (finance records, generated arithmetic) is judged unimportant and token cost matters more than arithmetic
regularity, take `vocab32k_v2`. The two are **not interchangeable**: choose before tokenizing anything for a real run.

## How to freeze
1. The team lead states the choice in the group; the choice is written here as `FROZEN.md` (directory, SHA-256, date, who approved).
2. Every `manifest.json` written by `tokenize_parquet.py` / `tokenize_text.py` carries `tokenizer_sha256`; it must equal the frozen
   hash (`scripts/check_bins.py --vocab` checks the vocabulary size and EOS id, the manifest check is part of the pre-launch list).
3. A frozen vocabulary is never edited or retrained. A different vocabulary means a new directory and re-tokenizing everything.
