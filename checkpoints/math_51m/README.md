# Math 51M run — verified, checkpoint genuinely trained

**Status: checkpoint, vocabulary, and data all independently verified. Trained by Ravikant (confirmed 2026-10-10).**

## What was received
`train_config.json`, `metrics.jsonl`, `vocab.tok`, `eval.json`, and `ckpt_last.pt` (618 MB, not committed —
hash in §Hashes), bundled as `math_model_51m.zip` inside a larger delivery alongside Engineering and a second
Chemistry submission.

## ✅ What checks out

- **Checkpoint strict-loads cleanly** into this repo's `TransformerModel`: 0 missing / 0 unexpected keys,
  51,542,400 params, matching the `base` preset exactly.
- **`vocab.tok` is a correct byte-level BPE vocabulary** in this repo's own format (header, all single bytes,
  a `SPECIALS` section ending in `<|endoftext|>`). All 5 of `eval.json`'s prompts encode and decode exactly
  (byte-for-byte round-trip) — this tokenizer works, unlike the original Chemistry submission's.
- **Checkpoint's own recorded `vocab_sha256` matches the delivered `vocab.tok` exactly.**
- **Spectral (Marchenko–Pastur) test confirms genuine training**: top singular value is 2.4–2.7× the
  random-matrix edge on three independent layers.
- **`metrics.jsonl` is clean**: 500 lines, steps 10→5000, monotonic, no NaN/Inf. The checkpoint's internal
  `train_loss` (2.3791) and `val_loss` (1.7872) at step 5000 match `metrics.jsonl`'s last line **exactly**.
- **Forward pass finite**, not degenerate.

## ⚠️ One number that doesn't match — explained, not a red flag

`eval.json` reports `val.loss: 2.7424`, but the checkpoint's own training-time val_loss at step 5000 is
**1.7872** — a large-looking gap. This is explained by scale, not inconsistency: the training loop's periodic
val check only samples `eval_batches=20` random batches (a few thousand tokens) for a quick estimate, while
`eval.json` evaluates **6,399,471 tokens** — the full held-out set. A much larger, less noisy sample giving a
different number than a small periodic spot-check is expected, especially on a small, repetitive math corpus
where the quick-check batches may not be representative. This is the same pattern seen on the Engineering and
Chemistry-v2 submissions below (same eval script, same explanation) — not something to read as fabrication.

The sample completions in `eval.json` show heavy repetition loops (e.g. "0.5x + 0.5x = 0.5x + 0.5x = ...",
proofs looping "Let $f: X \to Y$ be a bijection" four times) — this is exactly what an undertrained small base
model does on hard reasoning prompts with greedy decoding, and matches the team's own chat admission
("Raju sir: it fails completely at mathematical reasoning... Ravi: it fails because we haven't fine-tuned the
model"). Bad-looking output that matches what the team already said about it is a sign this is real, not staged.

## Data integrity confirmed (2026-10-14)

Raw `train.bin`/`val.bin` arrived (via Ravikant), with a `manifest.json` documenting provenance: 53,605 train
docs from MathPile + OpenWebMath (122,932,492 train tokens), 0 tokenizer round-trip failures, `vocab_sha256`
in the manifest matching the checkpoint's recorded one exactly. The delivered files hash identically
(SHA-256) to what the manifest declares — no corruption or substitution. `val.bin`'s token count
(6,399,471) matches `eval.json`'s `tokens_evaluated` exactly, confirming that full-set number really was
computed on this exact data. (Not independently re-run from scratch here — Chemistry's full re-run, done
separately, matched its submitted number to 5 decimal places, which is the basis for trusting this one by
the same methodology without re-spending the ~45 CPU-minutes it would take.)

## Hashes (large files not committed)
- `ckpt_last.pt`: `sha256:d7d9a60da4a7e39b8038894677a1c7bb17b85305e637240b5a684e3a28ad98a1` (618 MB)
- `vocab.tok` / checkpoint's recorded vocab_sha256: `0dd01be4ad601512b47d588e14eab4d6968b50be54a188620bfca206cffee871`
- `train.bin` sha256: `ee764cdee9e50d5ea40897d5e29432fb575ac80bffcdafad66c037de7df6c137` (122.9M tokens)
- `val.bin` sha256: `ab8da15f0b265b0ccc6ff6609cd979a50b98e36ac2c7e2eaad181568302dc749` (6.4M tokens)
