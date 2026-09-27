# Changelog

## 0.1.0 — first consolidated release
Consolidates the delivered components (GQA integration, RoPE / Dynamic NTK / YaRN, tokenizer, conditional reward model)
into one package, after an independent component review. See `docs/REVIEW_FINDINGS.md` for every finding and
`docs/TEAM_INSIGHTS.md` for what was kept versus changed.

### Added
- `llm/` package with relative imports (replaces flat, generically named modules).
- `llm/dynamic_ntk.py`, `llm/rope_cache.py`: the team's Dynamic NTK and RoPE cache, integrated into `RotaryEmbedding`.
- `llm/checkpoint.py`: self-describing checkpoints, vocabulary / config verification, untrained-weights warning.
- `llm/data.py`: `.bin` token files (tokenizer report format), `TokenDataset`, document split, `NearDuplicateFilter`.
- `llm/train.py`: reference trainer (bit-exact resume, `stop_step`, bf16 autocast on CUDA).
- `llm/bpe_trainer.py`: fast pure-Python BPE trainer writing the `.tok` format (tooling/tests only).
- `TransformerModel.forward(return_hidden=…, rope_context_len=…)`, `LMBackboneAdapter` for the reward model.
- Scripts: `prepare_biology.py`, `eval_bio.py`, `make_scaffold_checkpoint.py`, `mutation_check.py`.
- Tests (component by component), the team's original tests kept in `tests/legacy`, GPU handoff tests, C++ parity test (skipped until provided).

### Fixed
- YaRN frequency ramp and attention-logit multiplier (now per the paper).
- Dynamic NTK now follows the team's report (was a static factor active below the trained length).
- `RopeCache` returned stale tables for a different θ.
- No weight initialisation (initial loss 573 → ≈ ln V).
- Silent random-weight / byte-tokenizer fallbacks in the pipeline; lenient checkpoint loading.
- Control-token injection through prompts and documents.
- `generate()` edge cases (zero tokens, empty prompt, negative temperature, per-row EOS, context overrun).
- Host-device synchronisation in every attention layer at every decode step.
- Reward model could not attach to the LM ((B,T) mask crash, no hidden-state API).
- Two diverging tokenizer copies → one.
- `ModelConfig` validation uses exceptions instead of `assert`.

### Removed
- `ntk_scale_factor` (a static factor in no specification); `yarn.py` is a deprecated alias shim.
- The 206 MB `pretrain_model.pt` is **not** shipped (random weights); use `scripts/make_scaffold_checkpoint.py` or train one.
