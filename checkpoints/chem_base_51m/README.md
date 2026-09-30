# Chemistry 51M run (submitted by Ravikant) — checkpoint is real; DO NOT rely on this yet

**Status: the model weights are genuinely trained. Several other parts of this submission do not hold up under
verification and need answers before this can be trusted or used. Details below, all independently checked —
none of this is taken on the submission's word.**

## What was submitted
`vocab.tok`, `metrics.jsonl`, `train_config.json`, `eval.json`, `report.pdf` (via chat), plus `cleaned.txt` and
`ckpt_last.pt` separately. Files here: the four small ones. `vocab.tok`, `cleaned.txt` (1.9 GB) and `ckpt_last.pt`
(618 MB) are **not** committed — see §6 for why, and their hashes are recorded there for traceability.

## ✅ What checks out

- **The checkpoint is genuinely trained**, not random init: loads with 0 missing/0 unexpected keys against this
  repo's own `base` config (51,542,400 params, exact match); forward pass on random ids produces finite output;
  singular-value spectral test on three weight matrices gives ratios of **2.1–2.8×** the random-matrix edge
  (a random/untrained matrix would be ≈1.0×, as the original delivered `pretrain_model.pt` was — F-01). This
  really did train on a GPU with real gradient updates.
- **`metrics.jsonl` is clean**: 500 lines, steps 10→5000, strictly increasing, no duplicates, no NaN/Inf anywhere.
  Loss falls from ~9 to 1.20; throughput a very high **~14,600–14,900 tok/s** (config: `dtype: auto`, no
  `compile`) — plausible for a real, well-utilized GPU.
- **The checkpoint's own recorded vocabulary hash matches the delivered `vocab.tok` exactly**
  (`028de116…7d6101` both sides) — so at minimum, the vocabulary file sent is the one actually used at training
  time, not a mismatched or stale one.
- **`cleaned.txt` looks like genuine chemistry text** (1.9 GB, tagged `chempile-education|config=LibreText_Chemistry`,
  real prose on inspection).

## ❌ What does not check out

### 1. `vocab.tok` is not a byte-level BPE vocabulary — it's WordPiece, and this repo's tokenizer cannot use it
Decoding the file (it parses as valid `.tok` *syntax*, which is why nothing rejected it outright):
- The first 5 tokens are **`[UNK]`, `[SEP]`, `[PAD]`, `[MASK]`, `[CLS]`** — standard BERT/WordPiece special tokens,
  not this project's `<|endoftext|>` convention.
- Only **102 of the 256 raw bytes** are present as single-byte tokens (byte `0x00` and the space byte `0x20` are
  both missing). A byte-level BPE vocabulary must contain all 256 bytes by construction — this one doesn't, because
  it isn't one.
- No `SPECIALS` section (this repo's `.tok` spec expects one after the merge list).
- Readable vocabulary entries are bare subword pieces with **no leading-space marker** (`"the"`, `"and"`, `"ing"`,
  `"tion"` — no `Ġthe`/`" the"` the way byte-level BPE encodes word boundaries) — the WordPiece convention, where
  whitespace splitting happens outside the vocab entirely.
- **Proof it's unusable as-is:** `llm.tokenizer.BPETokenizer.from_file("vocab.tok").encode("The chemical formula for sulfuric acid is")`
  raises `KeyError: b' '` immediately — it can't even encode a single space, because that byte has no entry.
  None of `eval.json`'s three prompts can be encoded by this repo's tokenizer with this vocabulary.

**This means:** whatever pipeline actually trained this model and produced `eval.json`'s samples was *not*
`src/llm/tokenizer.py`, and isn't `native/tokenizer/` either (that one is genuine byte-level BPE, verified in
`docs/CPP_TOKENIZER_PARITY.md`). It was very likely a Hugging Face `BertTokenizer`/WordPiece pipeline. The `.tok`
file we received is a repackaging of that vocabulary into this project's file *syntax*, not a working tokenizer
for this project's *code*. **We cannot currently encode new prompts or decode this model's output ourselves.**
This is the same category of problem as F-29 (Earth & Environment's incompatible header), but worse: that one was
a fixable format mismatch; this one is a different tokenization algorithm entirely.

### 2. `eval.json`'s val_loss doesn't match the checkpoint's own training log
- `eval.json`: `val_loss: 1.1762728667259217`
- The checkpoint's own metadata (written by `llm.train` at the moment it was saved, step 5000): `val_loss: 1.1572749257087707`

Not wildly different, but they should be the same number (or extremely close) if `eval.json` was produced by
loading this exact checkpoint and re-running the same evaluation. A ~1.7% relative gap suggests `eval.json` came
from a separate evaluation pass — on different data, different code, or possibly not this checkpoint at all.

### 3. `eval.json`'s perplexity is suspiciously close to a different, unrelated model's number already in this repo
`eval.json` reports `val_ppl: 3.242267292561226`. This repo's own **Earth & Environment small-model** run
(`docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`, a 6.3M-parameter model, 900 steps, a completely different dataset) reports
`val_ppl: 3.2420564159714647` — matching to 4 significant figures. Two unrelated models (51.5M vs 6.3M params,
5000 vs 900 steps, chemistry vs earth-science text) landing on the same perplexity to that precision by chance is
not plausible. This looks like a template or a previously-reported number that wasn't actually recomputed for
this run — worth asking about directly rather than assuming either way.

### 4. `report.pdf` describes an evaluation this project has no way to produce, and this model has no way to pass
The PDF describes a **graded question-answering benchmark**: 1,200 items, 94.33% micro accuracy, per-capability
breakdown (`numerical_reasoning`, `factual_recall`, `stereochemical_reasoning`), per-subfield accuracy
(stoichiometry, electrochemistry, etc.), and a "worst 20 failures" table with specific gold-vs-predicted numeric
answers (e.g. "Calculate the moles of CO2 produced... gold=9, predicted=10").

This does not match anything this checkpoint or this codebase can do:
- This model was pretrained for 5,000 steps on **raw text, next-token prediction only** — there is no
  instruction-tuning / SFT stage anywhere in this project, and this checkpoint's config confirms it's the base
  pretraining preset, not a fine-tuned one.
- A from-scratch 51.5M-parameter base language model at this training budget cannot answer graded chemistry
  problems with 94% accuracy and produce clean numeric answers — that level of instruction-following and
  arithmetic reliability is far beyond what base pretraining at this scale produces, with or without a working
  tokenizer.
- This repository has **no QA-grading harness, no benchmark question set, and no accuracy-scoring code** anywhere
  that could have produced this report from this checkpoint.

**This PDF does not appear to be a real evaluation of this checkpoint.** It may be an unrelated attachment sent by
mistake, output from a different project, or something else — ask directly rather than filing it as evidence of
this model's quality.

## What this means

The **training run itself** (metrics, checkpoint weights, architecture) looks genuine and healthy. But the
**tokenizer is unusable**, one **evaluation number doesn't reconcile** with the checkpoint's own log, and the
**benchmark report doesn't match what this model or codebase can produce**. This submission cannot be used or
trusted as-is — not because the person didn't train something real, but because several of the artifacts around
it don't hold together under a basic sanity check.

## Questions to ask Ravikant directly (not to assume an answer to)

1. What tokenizer/library actually produced `train.bin`/`val.bin` and the `vocab.tok` file? (It looks like a
   Hugging Face WordPiece/BERT tokenizer — please share the actual tokenizer files — e.g. `tokenizer.json`,
   `vocab.txt` — not just the repackaged `.tok`, the same way it was needed for Earth & Environment's tokenizer.)
2. How was `eval.json` produced — by re-loading `ckpt_last.pt` and running this repo's `evaluate()`, or something
   else? Why does its `val_loss` differ from the checkpoint's own recorded one?
3. Where does the perplexity 3.24 figure come from — was it computed fresh for this run?
4. What is `report.pdf` an evaluation *of*? Which checkpoint, which question set, which grading code? It doesn't
   match this project's base-pretraining checkpoint.

## §6 — file hashes (large files not committed)
- `vocab.tok`: `sha256:028de11639bf8900effa662b6bf7e620e8ab5bbfb576f3df95cf8fe7657d6101` (107 KB — could be committed,
  but deliberately isn't, so nobody mistakes it for a working vocabulary; see finding above)
- `cleaned.txt`: `sha256:f107292c490be069a2519f1fe47c3b94bdd4c9e2975ed6239f843a9c07a95972` (1.9 GB)
- `ckpt_last.pt`: `sha256:a9f3e9acb2c5b0ca38ba3660082f1b251accf35e3f16caddb247ac2122bc95bf` (618 MB — also **exceeds
  GitHub's 100 MB hard limit** for a normal push; would need Git LFS or external hosting even if it were otherwise
  ready to commit, which it isn't yet given the open questions above)
