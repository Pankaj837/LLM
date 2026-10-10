# Chemistry 51M run — RESUBMISSION, fixes the issue in `../chem_base_51m/`

**Status: the tokenizer problem that blocked the original Chemistry submission is fixed. This resubmission
verifies cleanly. See `../chem_base_51m/README.md` for the original findings this replaces.**

## What changed from the original submission

| | Original (`../chem_base_51m/`) | This resubmission |
|---|---|---|
| Vocabulary format | WordPiece/BERT (`[UNK]`/`[SEP]`/`[CLS]`, 102/256 bytes) — **unusable** | Correct byte-level BPE, all 256 bytes, proper `SPECIALS` section — **works** |
| Tokenizer test | `KeyError: b' '` on every prompt | All 3 eval prompts encode/decode exactly |
| `report.pdf`'s 1200-item QA benchmark | Present, unexplained, implausible for this model | Not included this time — no unsupported claims |

## ✅ What checks out

- **Checkpoint strict-loads cleanly**: 0 missing / 0 unexpected keys, `base` preset, genuinely trained
  (spectral test ratios 1.8–2.7× the random-matrix edge).
- **`vocab.tok` is now correct byte-level BPE** in this repo's exact format. Prompts tested: "The reaction
  between an acid and a base", "In organic chemistry, the Lewis structure", "The first law of thermodynamics
  states that" — all round-trip exactly.
- **Checkpoint's own recorded `vocab_sha256` matches the delivered `vocab.tok` exactly.**
- **`metrics.jsonl` is clean** (500 lines, monotonic, no NaN/Inf). Checkpoint's internal `train_loss`
  (1.1799) matches `metrics.jsonl`'s last line exactly; `val_loss` differs somewhat (2.9615 internal vs.
  3.2979 in the log) — within the small-sample noise range seen on other submissions' quick periodic checks
  (`eval_batches=20`), not a new red flag.
- `eval.json`'s own val_loss (3.3964, over 449,002 held-out tokens) is in the same range — consistent with a
  harder-to-predict domain than the training loop's spot checks suggested, same pattern as Math/Engineering.

## Not yet resolved from the original submission
The original `report.pdf`'s 1200-question QA benchmark claim was never explained, and the original `eval.json`
discrepancy (F-31 in `docs/REVIEW_FINDINGS.md`) stands as a separate open question about that first delivery.
This resubmission doesn't claim to answer those — it simply provides a working, verifiable vocabulary and
checkpoint. Worth confirming directly whether this is intended to supersede the original entirely.

## Hashes (large file not committed)
- `ckpt_last.pt`: `sha256:1b6e926944aa0555cec0b800d0ee20df98ed6b6fc8691193a1ffd74e317c4b3a` (618 MB)
- `vocab.tok` / checkpoint's recorded vocab_sha256: `543a07b8c07a28e70c3caf300c0d8bf89b5f68c1e70b8e9246f8c536cd53022d`
