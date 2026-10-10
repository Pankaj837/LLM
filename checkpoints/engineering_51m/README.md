# Engineering 51M run — verified, checkpoint genuinely trained

**Status: checkpoint, vocabulary, and data all independently verified. Same explained eval-vs-training-log gap
as Math (see below); attribution needs confirming the same way.**

## What was received
`train_config.json`, `metrics.jsonl`, `vocab.tok`, `eval.json`, and `ckpt_last.pt` (618 MB, not committed —
hash in §Hashes), bundled as `engineering_model_51m.zip` alongside Math and a second Chemistry submission.

## ✅ What checks out

- **Checkpoint strict-loads cleanly**: 0 missing / 0 unexpected keys, `base` preset config.
- **`vocab.tok` is a correct byte-level BPE vocabulary** in this repo's own format. All 3 of `eval.json`'s
  prompts ("In software engineering, the design pattern", "The computational efficiency of the algorithm can
  be analyzed as", "Theorem: The neural network architecture satisfies") round-trip exactly.
- **Checkpoint's own recorded `vocab_sha256` matches the delivered `vocab.tok` exactly.**
- **Spectral test confirms genuine training**: ratios 2.4–3.0× the random-matrix edge.
- **`metrics.jsonl` is clean** (500 lines, monotonic, no NaN/Inf), and its last line
  (`train_loss: 2.7037, val_loss: 3.3616` at step 5000) matches the checkpoint's own internal meta **exactly**.
- **Forward pass finite.**

## ⚠️ Eval number gap — same explanation as Math

`eval.json` reports `val.loss: 2.9451` (over 17,150,433 tokens — the full held-out set) vs. the checkpoint's
own training-time spot-check val_loss of 3.3616 (20 random batches). Same cause as the Math submission: a
full-val-set evaluation is expected to differ from a small periodic sample. Not a red flag on its own.

## ⚠️ Attribution — needs confirming

Same open question as Math: no "Engineering" domain by that exact name is on the roster (closest: Afnan's
"Engineering and Mathematics"). If Afnan trained both this and the Math model as his two sub-domains, that
would resolve both attribution questions at once — please confirm.

## Hashes (large files not committed)
- `ckpt_last.pt`: `sha256:5f39cddcc6bd5dd0b46a844468c47bc0dbcf81874daad3f4c11f630afda3082e` (618 MB)
- `vocab.tok` / checkpoint's recorded vocab_sha256: `fdfd7505d2e3deb97311311e7c01863bf026e8f879b35b924cd05eaf0c8cbdb8`
