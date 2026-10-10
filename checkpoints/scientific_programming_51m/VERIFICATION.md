# Independent verification of Aarav's Scientific Programming submission

**Status: data and reports fully verified and internally consistent. Checkpoint weights themselves could NOT
be independently strict-loaded or spectral-tested — see §Checkpoint gap. One finding needs the user's
attention directly: an ownership discrepancy flagged by Aarav's own report (see §Attribution).**

## ✅ What checks out

- **`train.bin` / `val.bin` are valid** in this repo's exact format: correct `TOK1` magic, correct 36-byte
  header, `vocab_size=8193` / `eos_id=8192` matching the shared team vocabulary, exact size match for both
  files (21,485,660 train tokens / 1,146,491 val tokens).
- **`vocab.tok`'s SHA-256 (`7a56c0…fd362`) is identical to the team's shared `final_vocab_51m.tok`** — same
  vocabulary used by Prabanjan's Biology+Earth run and Om's Quant fine-tune.
- **`reports/evaluation.json` and `evaluation_pretrained.json` had a minor export glitch** (a stray literal
  `\n` after the closing `}`, making them fail strict JSON parsing) — fixed in the copies here; not a sign of
  anything wrong, just a trailing-text artifact from however the files were written out.
- **Fully self-consistent numbers**: `reports/metrics.jsonl`'s last line (step 7020, val_loss 2.0545) matches
  `evaluation.json`'s val_loss (2.0537) and `RESULTS.md`'s stated 2.0537/PPL 7.80 closely; the pre-fine-tune
  checkpoint's `evaluation_pretrained.json` (val_loss 6.8352, PPL 930.03) matches `RESULTS.md` **exactly**.
  `training_history.json`'s three-stage breakdown (5000→5020→6020→7020, token counts per stage) is internally
  consistent with the final 22-repository, 22,632,151-token corpus described in `RESULTS.md`.
- **Honest, specific self-reporting**: explicitly notes sampled outputs still show "repetition and
  cross-language leakage," names exactly which languages (SQL, MATLAB, Bash, LaTeX) are far below their
  provisional token targets, and states plainly "it does not establish production-quality scientific code
  generation." This is not an inflated report.
- **References commit `5fb3edf`** (this repo's actual commit from earlier in this project) as the code version
  used — consistent, checkable provenance.

## ⚠️ Checkpoint gap — could not independently verify the weights

`models/base/ckpt_best.pt`, `models/scientific_programming/{ckpt_best,ckpt_last}.pt` in the delivered ZIP are
**Git LFS pointer files (134 bytes each), not the actual checkpoints** — this is a plain `git clone`/"Download
ZIP" of a GitHub repo, which never includes LFS-tracked binary content. The pointers do confirm real, distinct
SHA-256 identities (`60b3cb85…e06a` for the base model — see cross-check below; `5bbe6a61…a9b10` for
`ckpt_best.pt`; `cc53fc85…fe81cf4` for `ckpt_last.pt`), but the weights themselves were not strict-loaded or
spectral-tested here, unlike every other submission in this directory. **To close this gap, we'd need the
actual LFS objects** (a `git lfs pull` against the real repo, or the files sent directly).

**Cross-check that still gives confidence**: the base-model LFS pointer's hash (`60b3cb85…e06a`) is *exactly*
the same checkpoint Prabanjan separately delivered to complete `../51m_model_run/`, and the same one Om's
Quant fine-tune in `../quant_51m/` started from. Three independent people referencing byte-identical hashes
for the same shared base checkpoint is strong corroborating evidence, even without loading Aarav's own
fine-tuned weights directly.

## ⚠️ Attribution — raised by Aarav's own report, needs the user's decision

Aarav's `REPORT.md` states directly: *"The collection-plan README allocates 2% of the wider corpus to this
domain... That README lists **Pankaj** as owner, while the group chat lists **Aarav**; this repository
preserves both source documents so the ownership discrepancy is visible."* This names the user. Not something
to resolve on his behalf — flagging as-is for a decision on who owns the Scientific Programming domain going
forward.

## Not committed
`train.bin` (41 MB), `val.bin` (2.2 MB), `vocab.tok` (113 KB, identical to the team's shared vocabulary — not
duplicated here), and the three checkpoint LFS pointers/binaries are not committed to this repo. The 18
source-repository `.jsonl` dumps and their licenses (`dataset/github/`) are Aarav's own collection, not copied
here either — they live in his `scientific_testing` repo.
