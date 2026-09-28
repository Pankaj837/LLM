# Checkpoints — TOY placeholders, not the production model

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

## In progress: a real 51M run

`51m_model_run/` tracks a real, non-toy training run (config + metrics, submitted by Prabanjan) — the run itself
completed 5,000/5,000 steps cleanly with healthy loss/perplexity convergence, but the `.pt` checkpoint file and
the vocabulary it was trained with haven't been submitted yet, so it can't replace the toy checkpoints below
until those arrive. Full verification and what's still needed: `checkpoints/51m_model_run/README.md`.

## What replaces these, and when

The team is training the **full 51.5M-parameter reference model** (`--preset base`) on a GPU, combining datasets
from everyone currently training in parallel:

| Person | Dataset(s) |
|---|---|
| Ravikant Kumar | Astronomy and Space, Chemistry |
| Prabanjan Jadav | Biology, Earth & Science |
| Om Ahire | Quant |
| Afnan | Engineering and Mathematics |
| Aarav Jayalwal | Scientific Programming |

Current as of 2026-09-29 — see `docs/HANDOFF_GPU.md` §4 if this has changed again.

Once every dataset above is ready and the combined GPU run finishes, **replace both files in this directory** with
the real checkpoint (or the relevant per-domain ones, if the team ships more than one). See `docs/HANDOFF_GPU.md`
for the training commands and `docs/REVIEW_FINDINGS.md` (F-01) for why the *original* delivered
`pretrain_model.pt` was never usable in the first place (random weights, not this size/scope issue).

**Until that replacement happens, anyone pulling this repo should assume these are placeholders — check this
file's own git history / the date below for how stale they are.**

Placed here: 2026-09-29.
