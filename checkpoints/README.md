# Checkpoints — TOY placeholders, not the production model

> Status 24 Oct 2026: the 51M models below are the only real checkpoints; **no scaled (100M–1.5B) model has been trained yet**.
> The plan and decisions are in `docs/TRAINING_PLAN_1p5B.md`. The roster and "current as of" dates in this file are historical.

**These are not the real model. Do not treat their output as a quality signal, and expect them to be replaced.**

| File | Params | Trained on | Steps | Purpose |
|---|---|---|---|---|
| `toy_bio_small_6.3M.pt` | 6.3M (`small` preset) | Biology corpus only (5.39M tokens) | 1,400 | CPU smoke test — proves the pipeline trains end to end |
| `toy_earth_small_6.3M.pt` | 6.3M (`small` preset) | Earth & Environment corpus only (3.54M tokens) | 900 | Same, second dataset |

Results and how these were produced: `docs/BIOLOGY_SMOKE_TEST.md`, `docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`.
Each carries provenance in its own metadata (config, step, losses, vocabulary hash — `src/llm/checkpoint.py`);
`untrained=False` on both (they did train — this is a size/scope limitation, not the F-01 random-weights problem).

## Why these exist

Pushed so anyone integrating against the pipeline now (e.g. wiring up a parser) has *something* real to point at,
while the actual model is still being trained. **Use these only to test that your integration loads a checkpoint
and produces output of the right shape — not to judge output quality.**

## Real 51M runs received so far (8, as of 2026-10-10)

| Directory | Domain | Submitter | Status |
|---|---|---|---|
| `51m_model_run/` | Biology + Earth & Science | Prabanjan | **Complete & verified.** Checkpoint arrived 2026-10-10; strict load, spectral test, tokenizer all pass. Now the best checkpoint for these domains. |
| `chem_base_51m/` | Chemistry (original) | Ravikant | **Not usable.** Vocabulary is WordPiece, not BPE — unreadable by this repo's tokenizer. See F-31. Superseded by `chem_base_51m_v2/`. |
| `chem_base_51m_v2/` | Chemistry (resubmission) | Ravikant | **Verified, fixes the above.** Correct BPE vocabulary this time; checkpoint and data check out. |
| `physics_51m/` | Physics | Prabanjan | **Verified.** First fully-clean submission; see its own README. |
| `math_51m/` | Math | Ravikant | **Verified** (checkpoint, vocab, data). |
| `engineering_51m/` | Engineering | Ravikant | **Verified** (checkpoint, vocab, data). |
| `quant_51m/` | Quant Finance (51.5M, base + fine-tuned) | Om | **Verified**, extremely well self-documented. Use `submission_quant_finetune/ckpt_best.pt`. |
| `quant_finance_small_6.3M/` | Quant Finance (6.3M toy) | Om | **Verified.** Superseded by `quant_51m/`. |
| `scientific_programming_51m/` | Scientific Programming | Aarav (left the team 2026-10-15) | **Closed.** Data/reports were verified, but the checkpoint was only ever delivered as Git LFS pointers (never the weights) and cannot now be obtained. Domain now owned by Ravikant. |

Every checkpoint above over 100 MB is **not committed to this repo** (GitHub's push limit) — each directory's
README/VERIFICATION file records SHA-256 hashes for traceability. Full findings: `docs/REVIEW_FINDINGS.md`.

A cross-cutting finding worth flagging on its own: Prabanjan's Biology+Earth checkpoint, Om's Quant fine-tune,
and Aarav's Scientific Programming fine-tune all independently reference the **exact same shared base-checkpoint
SHA-256** (`60b3cb85…e06a`) as their starting point — strong corroboration across three unrelated submissions
that this shared model is real.

## What replaces these, and when

The team is training the **full 51.5M-parameter reference model** (`--preset base`) on a GPU, combining datasets
from everyone currently training in parallel:

| Person | Dataset(s) |
|---|---|
| Ravikant Kumar | Astronomy and Space, Chemistry |
| Prabanjan Jadav | Biology, Earth & Science |
| Om Ahire | Quant |
| Afnan | Engineering and Mathematics |
| ~~Aarav Jayalwal~~ (left 2026-10-15) | ~~Scientific Programming~~ — reassigned to Ravikant; his other domains went to Ravikant (Chemistry, Scientific Q&A) and Gokul (Scientific Discovery, Biology, Earth & Environment) |

Current as of 2026-09-29 — see `docs/HANDOFF_GPU.md` §4 if this has changed again.

Once every dataset above is ready and the combined GPU run finishes, **replace both files in this directory** with
the real checkpoint (or the relevant per-domain ones, if the team ships more than one). See `docs/HANDOFF_GPU.md`
for the training commands and `docs/REVIEW_FINDINGS.md` (F-01) for why the *original* delivered
`pretrain_model.pt` was never usable in the first place (random weights, not this size/scope issue).

**Until that replacement happens, anyone pulling this repo should assume these are placeholders — check this
file's own git history / the date below for how stale they are.**

Placed here: 2026-09-29.
