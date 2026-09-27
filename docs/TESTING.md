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
5. **Real-data** (`test_bio_smoke.py`, opt-in via `BIO_ZIP`) — the full biology pipeline at small scale (~1 min);
   the full-scale run and its results are `docs/BIOLOGY_SMOKE_TEST.md`.
6. **GPU handoff** (`test_gpu_handoff.py`, opt-in, needs CUDA) — see `docs/HANDOFF_GPU.md`.
7. **C++ parity** (`test_cpp_parity.py`, opt-in via `TOK_CLI`/`TOK_VOCAB`) — skips until the team's binary and
   vocabulary are available; ready to run the moment they are.
8. **Release gate** (`test_checkpoint.py -k trained`, opt-in via `LLM_CKPT`) — fails on a checkpoint that looks
   like random initialisation. Verified to **pass** on our trained biology checkpoint and to **fail** on the
   delivered `pretrain_model.pt`.
9. **Mutation check** (`scripts/mutation_check.py`) — the ultimate check that the suite can fail. See below.

## Running

```bash
python -m pytest                                             # layers 1-4 (fast, ~40 s)
BIO_ZIP="Biology Dataset.zip" python -m pytest tests/test_bio_smoke.py -s
python -m pytest tests/test_gpu_handoff.py -s                 # on a CUDA machine
TOK_CLI=... TOK_VOCAB=... python -m pytest tests/test_cpp_parity.py
LLM_CKPT=runs/bio_small/ckpt_last.pt python -m pytest tests/test_checkpoint.py -k trained
python scripts/mutation_check.py
python scripts/lint.py
python scripts/smoke_test.py
```

## Current totals (CPU, this repository)
- Without `BIO_ZIP`: 257 passed, 14 skipped (GPU 7, C++ parity 2, real-data 3, release gate 1, legacy checkpoint 1).
- With `BIO_ZIP`: 260 passed, 11 skipped.
- Lint: 0 problems. Team smoke script: 29/29.
- Release gate: **passes** on `runs/bio_small/ckpt_last.pt` (trained here), **fails** on the delivered `pretrain_model.pt`
  (as it should — that file is random weights).

## Mutation testing
`scripts/mutation_check.py` re-introduces each fixed defect (and a few policy violations found while building this
repository) into a temporary copy of the tree and re-runs the tests that should catch it. A **surviving** mutant is a
behaviour nothing tests; the script exits non-zero if any survive.

Result on this repository: **every mutant is killed** (two source-level equivalent mutants are documented in the
script and excluded — see its `EQUIVALENT` list). Two mutants that looked killed on a first pass but were not
actually sensitive to the change were caught and rewritten (`test_planned_context_changes_logits_and_cached_uncached_logits_agree`
needed a sharper attention pattern before it could tell the mutant from the original) — this is why the check exists:
a passing assertion is not evidence unless it can also fail.

## Design notes
- `tests/conftest.py` provides `ckpt_path` (skips without `LLM_CKPT`) and `vocab_path` (a session-scoped stand-in
  vocabulary built once).
- `tests/helpers.py` centralises the small-model config, the legacy checkpoint-key format, and the reward-data
  surface-heuristic baseline so every file uses the same ones.
- Tests that assert a *quantity* (thresholds, ratios) always print the measured value in the assertion message and
  were calibrated against an actual run, not guessed.
