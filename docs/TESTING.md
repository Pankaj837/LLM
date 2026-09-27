# Testing

## Layers
1. **Unit / component** (`tests/test_c*.py`, `test_dynamic_ntk.py`, `test_data_bin.py`, `test_bpe_trainer.py`,
   `test_checkpoint.py`) — one file per component, each finding tagged `# noqa` style in its test name and
   cross-referenced in `docs/REVIEW_FINDINGS.md`.
2. **Legacy** (`tests/legacy/`) — the team's original `test_gqa.py` (10 tests) and `test_integration.py` (7 checks),
   ported to pytest with fixtures, **unmodified logic**. They must keep passing: if they don't, something changed
   the contract the team originally tested.
3. **Team smoke script** (`scripts/smoke_test.py`, 29 checks) — kept as delivered (one fixed expectation: the
   corrected YaRN sign) and run in CI as its own step.
4. **Integration** (`test_c11_integration.py`, `test_e2e.py`) — text → tokenizer → `.bin` → train → checkpoint →
   pipeline → reward model, and the LM/reward hand-off contracts.
5. **Real-data** (`test_bio_smoke.py` opt-in via `BIO_ZIP`; `test_prepare_earth.py` opt-in via `EARTH_ZIP`) —
   the biology and Earth & Environment pipelines at small scale (~1 min each); the full-scale runs and their
   results are `docs/BIOLOGY_SMOKE_TEST.md` and `docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`. `test_prepare_earth.py`'s
   synthetic-data tests (no zip needed) also pin the exact incompatible binary-header format found in that
   delivery (finding F-29), so a regression there is caught without the 24 MB dataset.
6. **GPU handoff** (`test_gpu_handoff.py`, opt-in, needs CUDA) — see `docs/HANDOFF_GPU.md`.
7. **C++ parity** (`test_cpp_parity.py`, opt-in via `TOK_CLI`/`TOK_VOCAB`) — skips until the team's binary and
   vocabulary are available; ready to run the moment they are.
8. **Release gate** (`test_checkpoint.py -k trained`, opt-in via `LLM_CKPT`) — fails on a checkpoint that looks
   like random initialisation. Verified to **pass** on our trained biology checkpoint and to **fail** on the
   delivered `pretrain_model.pt`.
9. **Mutation check** (`scripts/mutation_check.py`) — the ultimate check that the suite can fail. See below.

## Running

```bash
python -m pytest                                             # layers 1-4 (fast, ~45 s)
BIO_ZIP="Biology Dataset.zip" python -m pytest tests/test_bio_smoke.py -s
EARTH_ZIP="Earth and Environment Processed Dataset.zip" python -m pytest tests/test_prepare_earth.py -s
python -m pytest tests/test_gpu_handoff.py -s                 # on a CUDA machine
TOK_CLI=... TOK_VOCAB=... python -m pytest tests/test_cpp_parity.py
LLM_CKPT=runs/bio_small/ckpt_last.pt python -m pytest tests/test_checkpoint.py -k trained
python scripts/mutation_check.py
python scripts/lint.py
python scripts/smoke_test.py
```

## Current totals (CPU, this repository)
- Without `BIO_ZIP`/`EARTH_ZIP`: 265 passed, 15 skipped (GPU 7, C++ parity 2, real-data 4, release gate 1, legacy checkpoint 1).
- With both: 269 passed, 11 skipped.
- Lint: 0 problems. Team smoke script: 29/29.
- Mutation check: **25/25 mutants killed, 0 survived**.
- Release gate: **passes** on `runs/bio_small/ckpt_last.pt` (trained here), **fails** on the delivered `pretrain_model.pt`
  (as it should — that file is random weights).

## Mutation testing
`scripts/mutation_check.py` re-introduces each fixed defect (and a few policy violations found while building this
repository) into a temporary copy of the tree and re-runs the tests that should catch it. A **surviving** mutant is a
behaviour nothing tests; the script exits non-zero if any survive.

Result on this repository: **25/25 mutants killed, 0 survived** (two further source-level equivalent mutants are
documented in the script's `EQUIVALENT` list and never added as cases, since they cannot be killed by design).
Two mutants that looked killed on a first pass were not actually sensitive to the change and were caught and
rewritten (`test_planned_context_changes_logits_and_cached_uncached_logits_agree` needed a sharper attention
pattern before it could tell the mutant from the original) — this is why the check exists: a passing assertion is
not evidence unless it can also fail.

One mutant (`llm/bpe_trainer.py`, flipping the merge-priority sign) originally broke the trainer's internal
stale-cache-entry invariant instead of just its priority order, which sent it into an infinite loop that ran
for hours undetected before being found and killed manually. Fixed by mutating the sign and its paired
consistency check together, and by adding a 90-second per-mutant `subprocess` timeout (a `TIMEOUT` result now
counts as a survivor needing attention rather than hanging the whole check).

## Design notes
- `tests/conftest.py` provides `ckpt_path` (skips without `LLM_CKPT`) and `vocab_path` (a session-scoped stand-in
  vocabulary built once).
- `tests/helpers.py` centralises the small-model config, the legacy checkpoint-key format, and the reward-data
  surface-heuristic baseline so every file uses the same ones.
- Tests that assert a *quantity* (thresholds, ratios) always print the measured value in the assertion message and
  were calibrated against an actual run, not guessed.
