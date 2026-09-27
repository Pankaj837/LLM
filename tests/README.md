# Tests

Test **code** lives here, one file per component (see `docs/COMPONENTS.md`). Narrative **results** — what was run,
against what data, with what numbers — live in `docs/` as reports (`docs/TESTING.md`, `docs/BIOLOGY_SMOKE_TEST.md`,
`docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`), because they need prose and tables, not just pass/fail. `results/` here
holds the raw, timestamped output of the last time this suite, lint and the mutation check were actually run.

## Layout
- `test_c1_config_blocks.py` … `test_c11_integration.py` — one file per component of the model/tokenizer stack.
- `test_dynamic_ntk.py`, `test_data_bin.py`, `test_bpe_trainer.py`, `test_checkpoint.py`, `test_train.py` — the
  supporting modules (RoPE/NTK, `.bin` I/O, BPE trainer, checkpoints, trainer) added during consolidation.
- `test_e2e.py` — text → tokenizer → `.bin` → train → checkpoint → pipeline → reward model, in one run.
- `test_bio_smoke.py`, `test_prepare_earth.py` — real-data smoke tests; opt-in via `BIO_ZIP` / `EARTH_ZIP` env vars
  (skip cleanly without them; `test_prepare_earth.py`'s format-parsing tests run always, no dataset needed).
- `test_cpp_parity.py` — opt-in via `TOK_CLI` / `TOK_VOCAB`; skips until the team's C++ tokenizer is available.
- `test_gpu_handoff.py` — opt-in, needs CUDA; skips on CPU-only machines.
- `test_checkpoint.py -k trained` — the release gate; opt-in via `LLM_CKPT`, fails on a checkpoint that looks
  like random initialisation.
- `legacy/` — the original team's `test_gqa.py` and `test_integration.py`, ported to pytest but **logic
  unchanged**; kept passing to prove nothing broke their original contracts.
- `conftest.py`, `helpers.py` — shared fixtures (a stand-in BPE vocabulary, a small model config, the legacy
  checkpoint key format).
- `results/` — raw output of the last real run of `pytest`, `scripts/lint.py` and `scripts/mutation_check.py`
  (see each file's own header line for the date). Regenerate any of them with the commands in `docs/TESTING.md`.

## Quick run
```bash
pip install -e .[dev]
python -m pytest                 # this directory; opt-in tests skip themselves without their env var
```
