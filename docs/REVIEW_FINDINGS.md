# Review findings and their final status

Origin: independent component review of the delivered `llm/` code (owner-agnostic), then re-verification while building
this repository. Every *Resolved* item has a test that fails without the fix (checked by mutation testing, see
`docs/TESTING.md`). Severity: **Critical** blocks meaningful GPU testing/integration · **High** wrong results or security ·
**Medium** robustness · **Design** decision needed.

| ID | Sev | Component | Finding | Resolution here | Status |
|---|---|---|---|---|---|
| F-01 | Critical | C7 | `pretrain_model.pt` is statistically identical to random init (singular values on the random-matrix edge, kurtosis ≈ 0, norm weights 1.000 ± 0.0006, greedy output = one repeated token). The integration report calls the checkpoint files “dummy” | Loader warns on every use; `scripts/make_scaffold_checkpoint.py` makes an explicit `untrained=True` file; release gate `LLM_CKPT=… pytest -k trained`; trainer + data path proven on real text (`docs/BIOLOGY_SMOKE_TEST.md`) | **Open**: the 51.5M reference model still needs a GPU training run |
| F-02 | High | C7 | Checkpoint has no config / tokenizer identity / provenance | Self-describing format, config + vocabulary-hash checks, `weights_only` loading | Resolved |
| F-03 | High | C3 | YaRN logit multiplier inverted (×0.97 instead of ×1.14 at s=2) | (0.1·ln s + 1)² | Resolved |
| F-04 | High | C3 | YaRN ramp linear in wavelength, not dimension index (frequencies off 25–71%) | Reference ramp | Resolved |
| F-05 | High | C3 | “Dynamic NTK” in the unified `rope.py` was a static factor, active below the trained length, and ignored the team's Dynamic NTK report | Team's `DynamicNTK` / `RopeCache` integrated (strict threshold, `d/(d−2)`), report values are tests, planned-context policy (D1) | Resolved |
| F-06 | Critical | C2 | No weight init: tied head at N(0,1) → initial CE ≈ 573 | N(0, 0.02) init | Resolved |
| F-07 | Medium | C6 | `max_new_tokens=0` returned an extra token (cached path) | early return | Resolved |
| F-08 | Medium | C6 | empty prompt: opaque crash; negative temperature accepted | `ValueError` | Resolved |
| F-09 | High | C6 | finished rows kept generating in batches | per-row `finished` mask | Resolved |
| F-10 | Medium | C6 | no warning past the configured context | `UserWarning` | Resolved |
| F-11 | Medium | C1 | nonsensical scale factors / odd `head_dim` accepted | `ValueError`s (not `assert`) | Resolved |
| F-12 | Critical | C8 | wrong path silently gave random weights / a byte tokenizer; lenient loading; no vocabulary↔embedding check | fail loudly, `strict=True`, vocabulary sha256 + size checks | Resolved |
| F-13 | High | C5/C8 | user text `<|endoftext|>` became the real EOS token | `allowed_special=set()` for prompts and documents | Resolved |
| F-14 | Critical | C9/C2 | LM had no hidden-state API; reward model could not attach | `return_hidden`, `LMBackboneAdapter` | Resolved |
| F-15 | Critical | C2/C9 | `(B,T)` padding mask crashed the LM | normalised to a 4-D bool mask | Resolved |
| F-16 | Medium | C9 | all-padding row silently pooled position −1 | `ValueError` | Resolved |
| F-17 | Medium | C5 | two tokenizer copies disagreed on unknown-id decoding | one tokenizer (D4) | Resolved |
| F-18 | Critical | C9 | reward-model scripts failed on import; files missing | package, relative imports, stand-in vocab trainer | Resolved |
| F-19 | High | C9 | evidence weak: a 5-line rule scores 100% on validation, 48-pair validation set, seed noise (FiLM vs concat flips), swap test on a training prompt | baseline printed, held-out swap prompt, README caveats | Resolved in code/docs; **conclusions need real preference data** |
| F-20 | Design | C5 | duplicate tokenizer implementations | consolidated | Resolved |
| F-21 | Medium | C4 | GPU performance: `.item()` host sync in every layer at every step; `repeat_interleave`; fused-kernel masking; bf16 | `.item()` removed (test); the rest need a GPU | **Open**: `docs/HANDOFF_GPU.md` |
| F-22 | High | all | missing: C++ tokenizer + trained vocabulary, pruning, training code/data, docs | trainer, `.bin` I/O, docs added | **Open**: C++ tokenizer / `vocab.tok`, pruning |
| F-23 | Medium | docs | README layout / stale TODOs / inconsistent names | rewritten | Resolved |
| F-24 | Design | C9 | reward backbone architecture ≠ LM | adapter provided | **Decision**: standalone or shared backbone |
| F-25 | High | C10 | LibreTexts republishes chapters: after exact-hash dedupe **25.5%** of its validation windows also occurred in training (4.9% overall) — would silently inflate validation scores | `NearDuplicateFilter` (content-defined 12-word shingles); overall leakage now 0.3% | Resolved |
| F-26 | High | C7 | `torch.__version__` (a `TorchVersion` object) stored in checkpoint metadata made `weights_only=True` loading fail — found by the new tests | stored as `str` | Resolved |
| F-27 | Medium | C3 | delivered `RopeCache` keyed on sequence length only → stale tables for a different θ | key includes θ, head_dim, device, dtype | Resolved |
| F-28 | Medium | C11 | a run “stopped early” by lowering `max_steps` follows a different LR schedule and cannot resume bit-exactly | `stop_step` | Resolved |
| F-29 | High | C10 | the Earth & Environment delivery's `.bin` files use a **different, incompatible header** (24 bytes, magic `GNRP`) than this repo's tokenizer-report spec (36 bytes, magic `TOK1`) — `llm.data.read_header` correctly refuses to open them | `scripts/prepare_earth.py`: reads the delivered format, independently re-verifies every claim in its `PROCESSING_SUMMARY.md` (all matched), re-emits through `llm.data.write_bin` | Resolved (adapter); **the two header formats should be unified for future deliveries** |

## Not covered by this repository (needs input from the team)
1. **C++ tokenizer parity** (source, `encode_cli`, trained `vocab.tok`) — `tests/test_cpp_parity.py` is ready.
2. **Structured pruning** and its interaction with the reward model.
3. **A trained reference checkpoint** (51.5M) and the perplexity-vs-length evaluation of Dynamic NTK / YaRN on it.
4. **Real preference data** for the reward model.
5. **Dual Chunk Attention** (described in the RoPE report for >128K tokens; out of scope at 2048).
6. **Licences**: the biology data carries per-record licences (`pes2o` is marked “unknown” for all 6,499 records);
   check before sharing models or data outside the team.
