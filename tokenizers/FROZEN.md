# FROZEN tokenizer

| | |
|---|---|
| Directory | `tokenizers/vocab32k_v2/` |
| File | `vocab32k.tok` |
| SHA-256 | `10b3c5aecf6b81746bf86f8c18f8c576bf716a3620e7632e8fc2684006b51cdc` |
| n_vocab / EOS | 32,000 / `<|endoftext|>` = 31,936 (64 special tokens at ids 31,936–31,999) |
| Frozen | 25 Oct 2026 |
| Decided by | Raju sir (24–25 Oct: "go with vocab32k"; "evaluate vocab32k_digits_v2 separately later if digit-level numerical reasoning shows a clear need") |
| Verified | lossless round-trip on held-out text of 12 domains; C++ and Python encoders give identical ids on 3003/3003 lines (`scripts/verify_vocab.py`) |

**This vocabulary is never edited or retrained.** Every `manifest.json` written by `tokenize_parquet.py` / `tokenize_text.py` must carry this
SHA-256 as `tokenizer_sha256`; every checkpoint stores it. A different vocabulary means a new directory and re-tokenizing everything.

`tokenizers/vocab32k_digits_v2/` (numbers digit by digit, ~5–6% more tokens) stays in the repository as a **not-frozen experiment**: it is
evaluated separately, and only if digit-level numerical reasoning shows a clear need (Raju's condition) would it replace this one, for a new run.
