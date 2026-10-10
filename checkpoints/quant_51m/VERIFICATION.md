# Independent verification of Om's Quant Finance submission

**Status: fully verified. This is the most thoroughly self-documented submission received — Om's own
`README.md` and `RESULTS.md` in this folder already cover the methodology in detail; this file records what
I checked independently rather than taking the self-report at face value.**

## What was independently re-checked

- **All 4 checkpoint SHA-256 hashes** (`submission_quant_base/{ckpt_last,ckpt_best}.pt`,
  `submission_quant_finetune/{ckpt_last,ckpt_best}.pt`) were recomputed from the delivered files and matched
  **exactly** against Om's own `SHA256SUMS.txt` in each folder — no corruption, no substitution.
- **`submission_quant_base/ckpt_best.pt`** (step 500, the recommended from-scratch checkpoint): strict-loads
  with 0 missing/unexpected keys, 51,665,920 params; `vocab_sha256` matches the delivered `vocab.tok` exactly;
  tokenizer round-trips "The price of a European call option" and other finance prompts exactly; spectral test
  ratios 1.3–1.7× the random-matrix edge (lower than the 5000-step submissions elsewhere in this repo, as
  expected — this checkpoint only saw 500 steps); checkpoint's internal `val_loss` (3.739564) matches
  `eval_best.json`'s `val.loss` (3.739576) to 4 decimal places — the tightest eval/checkpoint agreement of any
  submission reviewed so far.
- **`submission_quant_finetune/ckpt_best.pt`** (step 5700, the overall-best Quant checkpoint, fine-tuned from
  the team's shared base model): same checks, all pass; spectral ratios 2.2–3.7× (higher, consistent with
  far more total training than Run A); internal `val_loss` (3.430881) matches `eval_best.json` (3.430873)
  almost exactly.
- **The fine-tune's claimed starting point** — the team's shared 51.5M base checkpoint at step 5000,
  SHA-256 `60b3cb85…e06a` — **independently confirmed**: this is the exact hash of the checkpoint Prabanjan
  separately submitted to complete `../51m_model_run/` (Biology + Earth & Science), and it's also the exact
  Git LFS object hash referenced in Aarav's Scientific Programming submission's `models/base/ckpt_best.pt`
  pointer. Three independent submissions referencing the identical shared checkpoint hash is strong
  corroboration this is a real, shared artifact, not three separate claims.
- Om's own reported "false alarm" in his `lr_schedule_ok` checker was independently sensible: the two flagged
  points (step 20 during warmup, steps 4970–4990 at the cosine floor) are exactly where a correct schedule
  would sit, so his correction is right.

## Assessment

No inconsistencies found. The self-reported overfitting on Run A's `ckpt_last.pt` (step 5000, perplexity 435,
9/18 generated samples copying 20+ words verbatim) is itself a strong authenticity signal — nobody fabricating
a result would report their own model memorizing and failing this clearly, with exact verbatim-copy counts.
**Recommendation (matches Om's own): use `submission_quant_finetune/ckpt_best.pt`, not either `ckpt_last.pt`.**

Checkpoint files (4 × ~591 MB) and the reproduce/Kaggle notebook kit are not committed here (exceed GitHub's
100 MB push limit) — hashes above and in each submission's own `SHA256SUMS.txt` are the record.
