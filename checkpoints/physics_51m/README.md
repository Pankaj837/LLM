# Physics 51M run (shared by Prabanjan) — verified, best submission so far

**Status: this is the most trustworthy training submission received to date. Checkpoint, vocabulary, and dataset
all independently verified as internally consistent and genuinely trained. One open question before this can be
merged in: no one on the current team roster is assigned "Physics" — see §Open question below.**

## What was received
A full ZIP (`drive-download-20260930T225747Z-1-001.zip`, 1.4 GB) containing: a working copy of this repository
(`LLM/`), `Physics_vocab.tok`, `train.bin` / `val.bin`, and `LLM/runs/51m_model_run/{train_config.json,
metrics.jsonl, ckpt_last.pt, ckpt_best.pt}`. Only the small files (`train_config.json`, `metrics.jsonl`, this
README) are committed here — `Physics_vocab.tok` (116 KB), `train.bin` (453 MB), `val.bin` (24 MB), `ckpt_last.pt`
and `ckpt_best.pt` (592 MB each) are not, both because of GitHub's 100 MB push limit and because they don't need
to be in git to be verified. Hashes are recorded in §Hashes for traceability.

## ✅ Everything checked out

- **The bundled code is our repo's actual code.** `model.py`, `data.py`, `tokenizer.py`, `checkpoint.py`, `gqa.py`,
  `rope.py`, `model_config.py`, and `pipeline.py` are all **byte-identical** to this repository's copies. The only
  difference is `train.py`, which adds `torch.compile` support (a `--compile` flag) that our canonical `train.py`
  doesn't have — a reasonable Colab speed optimization, not a correctness change, and it doesn't touch the model
  architecture. This means the checkpoint really was produced by (a superset of) our own training code.
- **`Physics_vocab.tok` is a genuine byte-level BPE vocabulary in this repo's own format** — correct header
  (`# tiktoken-style-vocab / V1 / 8192`), all single bytes present, a `SPECIALS` section whose one entry decodes
  to `<|endoftext|>` (this repo's own EOS convention). Loaded directly with `llm.tokenizer.BPETokenizer.from_file`
  and used to encode all 6 of the physics prompts in the eval script — every one round-trips exactly
  (encode → decode reproduces the original string byte-for-byte). This is the opposite of the Chemistry submission,
  where the vocabulary couldn't even encode a space.
- **`train.bin` / `val.bin` are valid in this repo's exact binary format**: correct magic (`TOK1`), correct
  36-byte header, `vocab_size=8193` and `eos_id=8192` matching the vocabulary and the model config exactly,
  `split` field correctly 0 (train) / 1 (val), and file size matches the header-declared token count exactly in
  both files (237,573,175 train tokens / 12,426,825 val tokens — no truncation, no corruption).
- **The checkpoint loads strictly into this repo's own `TransformerModel`**: 0 missing keys, 0 unexpected keys,
  against the exact `base` preset config declared in `train_config.json`. Forward pass on random ids produces
  finite logits (std ≈ 2.42, not degenerate).
- **Spectral (Marchenko–Pastur) test confirms genuinely trained weights**, same method used on every other
  submission: top singular value is **2.5–3.6×** the random-matrix edge on three independent layers
  (`layers.0.attn.q_proj.weight`, `layers.5.ffn.W_gate.weight`, `layers.10.ffn.W_down.weight`) — a random/untrained
  matrix would sit at ≈1.0×.
- **The checkpoint's own recorded `vocab_sha256` matches the delivered `Physics_vocab.tok` exactly**
  (`42372eeb…44b6d6`), and its internal `meta.train_loss` / `meta.val_loss` at step 5000
  (`3.0245392322540283` / `3.6246981859207152`) match `metrics.jsonl`'s last line **exactly, to every decimal** —
  unlike the Chemistry submission, where the submitted eval numbers didn't match the checkpoint's own log at all.
- **`ckpt_best.pt` and `ckpt_last.pt` are the same checkpoint** (identical step/losses) — the final step also had
  the best validation loss, consistent with a still-improving (not yet overfit) run at the point it stopped.
- **`metrics.jsonl` is otherwise clean**: 543 lines, no NaN/Inf, loss falls smoothly from 8.67 → 3.02, val_ppl
  falls from 899.6 (step 100) to a plausible **37.5** (step 5000) for a from-scratch 51.5M model after 5,000 steps
  on a harder, more technical domain — a far more credible number than Chemistry's implausible 3.24.
- **One resume event, fully explained by the submitted config**: step numbers jump from 3430 back to 3010 once
  (43 steps re-logged), matching `train_config.json`'s `"resume": "runs/51m_model_run/ckpt_last.pt"` and
  `README_COLAB.md`'s own warning to save checkpoints before a Colab session ends. This is exactly the log
  signature of a Colab disconnect around step ~3430, resumed from the last save point (step 3000, `save_interval:
  500`) — not a data-integrity problem.

## ⚠️ Open question — not a verification failure, but needs an answer before this is merged in

**No one on the current team roster (`docs/HANDOFF_GPU.md`) is assigned "Physics."** The roster has Ravikant on
Astronomy & Space / Chemistry, Prabanjan on Biology / Earth & Science, Om on Quant, Afnan on Engineering &
Mathematics, Aarav on Scientific Programming. This run is real and well-executed, so it isn't being questioned on
technical grounds — but please confirm with Prabanjan (or whoever assigned this) whether Physics is a new/added
domain, a relabeling of an existing one, or something that should be attributed to someone else, so the roster and
this checkpoint's ownership stay accurate.

## Hashes (large files not committed)
- `Physics_vocab.tok`: `sha256:42372eeb4f55ba91c262f39df1f9974bbc386ee7c12ae5c898f77c783844b6d6` (116 KB)
- `train.bin`: `sha256:a9772bb61f07d90a6c092328ba5195c69feb412b07f84d042ca19c881ba4bd4d` (453 MB)
- `val.bin`: `sha256:2f124d74fda063b850398e37d311bfcee341e315cc03f3eb4598482a1cc43375` (24 MB)
- `ckpt_last.pt`: `sha256:691e4abea40375b5dead905208909e1de993e4a110392070d03f90482af9af38` (592 MB)
- `ckpt_best.pt`: `sha256:75659fc8d51ac326c0aa2ddaad14b80e1eb56886c93d32a06c09f816f9e171fb` (592 MB, identical
  step/losses to `ckpt_last.pt`)

Both `.pt` files also exceed GitHub's 100 MB per-file push limit — Git LFS or external hosting would be needed if
this is committed later.

## What's still missing
No `eval.json` or benchmark report was included with this submission (unlike Chemistry's, which included both —
though that one didn't hold up). That's not a problem: nothing here was fabricated to look more finished than it
is. Once the roster question above is settled, running `scripts/eval_physics.py` (delivered in the same bundle,
reviewed and confirmed to be a legitimate adaptation of our own `scripts/eval_bio.py` — same pipeline/data/
checkpoint calls, physics-specific prompts, no red flags) against `ckpt_last.pt` would give a real evaluation
number to compare against the training-time val_ppl of 37.5.
