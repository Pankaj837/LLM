# Changelog

## 0.1.0 — first consolidated release
Consolidates the delivered components (GQA integration, RoPE / Dynamic NTK / YaRN, tokenizer, conditional reward model)
into one package, after an independent component review. See `docs/REVIEW_FINDINGS.md` for every finding and
`docs/TEAM_INSIGHTS.md` for what was kept versus changed.

### Added
- `scripts/tokenize_parquet.py` (+ tests): streams Parquet shards into TOK1 `.bin` files with a frozen tokenizer in low memory, in parallel and resumable; `--count-only` gives per-domain token estimates from `char_count` in seconds. Also reads v4.0 entity-schema files (no `text` column): body taken from the `properties` JSON (`code_content`, else question+answer, else `description`) -- a proposed rule the data owners must confirm.
- 24 Oct audit of the whole scale-up pipeline (see `docs/PROGRESS_REPORT.md`, "Failures and corrections"):
  - `scripts/tokenize_text.py`: tokenizer for `.txt` / `.txt.gz` corpora (the 34 GB math corpus, Multi-step, Astronomy, Finance, chemistry text); streams gzip, parallel, resumable; Windows `\r\n` and invalid bytes handled.
  - `scripts/tokenize_parquet.py`: big files are split into row-group jobs so one file uses every core; unique job names (same-named inputs no longer overwrite each other); lone surrogates repaired instead of crashing; inputs recorded in the manifest.
  - `scripts/merge_bins.py`, `scripts/check_bins.py`: one token file per domain; header / size / id-range / EOS / vocabulary integrity gate.
  - `llm.data.MixtureDataset` and `llm.train --train-mix PATH=WEIGHT ...`: weighted domain mix with a per-file epoch report and warning; `llm.train` also gets a 60-minute collective timeout for DDP and atomic checkpoint writes.
  - `scripts/eval_domains.py`: wildcard pools a domain's several validation files.
  - `llm.bpe_trainer.train_bpe(split_digits=True)` / `scripts/train_vocab.py --split-digits`: no token with two adjacent digits; second tokenizer candidate `tokenizers/vocab32k_digits_v2/` (numbers digit by digit, ~5–6% more tokens). `scripts/verify_vocab.py` now samples deterministically per domain.
  - `v4_text` rule rebuilt from a field census of the real files (Q&A options/explanations, `doc_content`, DeepCAD, structured QA pairs).
  - CI installs `pyarrow` (the pipeline tests no longer skip) and runs the repo lint; `.tok` files are stored byte-exact.
- `tokenize_text.py --eot-domains` / `build_dataset.py` key `"eot": true`: documents separated by a line reading `<|endoftext|>`. Found on 28 Oct in the first 20 MB of the engineering reasoning corpus (problem + `<think>` trace + answer, with a blank line between almost every line): the blank-line rule gave 221,904 "documents" of 26 tokens on average (an end-of-text token after every paragraph, reasoning traces cut into pieces), the right rule gives 386 documents of 15,073 tokens; both pass `check_bins`.
- `tokenize_text.py --start-domains D --start-prefix P` / `build_dataset.py` key `"start": "P"`: every document begins with a header line starting with `P` (the header line is dropped). Found on 29 Oct in Ravi's raw ChemPile text (`chempile-code|config=...|license=mit`, `chempile-education|...`, `chempile-paper|...`): code documents contain many blank lines, so the line rule cut them into fragments (and the header lines became tiny duplicate documents). On the first 20 MB the rule gives 2,715 whole documents of 7,639 characters on average, no duplicates, none under 60 characters.
- `scripts/sample_outputs.py` (+ tests): real held-out inputs and outputs per domain, next-token top-1 accuracy, ten fixed probes with automatic pass/fail, and failure flags (repetition loop, empty, invalid UTF-8, copies the prompt); written as `.json` and `.md`. First H100 session analysed in `docs/FIRST_RUN_ANALYSIS_2026-10-27.md`.
- `scripts/gpu_session.py`: the 3-hour GPU session as one command (`preflight` / `bench` / `smoke` / `train` / `eval` / `report` / `all`). Preflight checks the GPU, the exact git tag, the frozen tokenizer hash (also in every build manifest), `check_bins`, disk and permissions before any GPU time is spent; the benchmark picks the micro-batch; the 1.5B smoke test (60 steps, save, resume to 80) stops the session on a failure; a 30-step calibration on the real mix sets `max_steps` from the measured tokens/s and the run is killed if it overruns its budget by 10%; an interrupted run resumes with the same plan; `session_report.md` is the post for the group. Rehearsed end to end on CPU with the real frozen tokenizer (`tests/test_gpu_session.py`).
- `merge_bins.py --cap-per-input N` and `val_mix.bin`: `llm.train --val-bin` takes ONE file, but the builder only writes per-domain validation files, and a plain concatenation would have validated on the first domain only (the evaluation reads the first 20 batches). The balanced file takes the same number of whole documents from every domain.
- `build_dataset.py` writes `build_manifest.json` next to the bins (tokenizer SHA-256, code commit, config and its hash, tokens, input sizes) and honours `weights_sum` so a partial config reports epochs against the whole mix; `tokenize_text.py --max-tokens` estimates what is already in flight and no longer overshoots by the pipeline depth (a real run overshot 8M to 17M tokens).
- `tokenize_text.py --marker-domains` (documents between `<|document|>` / `<|endofdocument|>` lines, as in the physics file) and `--spread K` (with `--max-tokens` on a plain-text file: K evenly spaced slices of whole documents instead of a prefix; Windows line endings handled); `build_dataset.py` keys `marker` and `spread`.
- `scripts/benchmark.py` (throughput, peak memory, MFU and rupees per 1B tokens of the real training step on synthetic tokens; survives OOM), `scripts/build_dataset.py` + `configs/run_100m.example.json` (config → capped tokenization per domain → merge → integrity check → printed `--train-mix`), `tokenize_parquet.py --sample-rg` (evenly spaced row groups), `tokenize_text.py --max-tokens`, `peak_mem_gb` in the training metrics, `docs/CAPABILITIES.md` (the official 15 capabilities and data coverage).
- Tokenizer candidates renamed `vocab32k_v2` / `vocab32k_digits_v2` and rebuilt with **64 special tokens** (`train_vocab.py --reserved-specials 64`): EOS moved from id 31,999 to 31,936; chat / tool-use markers and 55 reserved slots are reserved up front (see `tokenizers/README.md`). Nothing had been tokenized with the earlier drafts.
- `llm.train` writes `run_info` (git commit, dirty flag, torch/CUDA versions, GPU, host) into `train_config.json`; `tests/test_golden.py` pins the single-process training numbers produced before the scale-up work (bit-identical weights verified once on commit 9f05cd8).
- `tokenize_parquet.py` / `tokenize_text.py` store a fingerprint (vocabulary hash, settings, input size and time, rule version) with every finished job and reuse it only if identical; job outputs are written to temporary names and renamed on success; `dedup_parquet.py` writes 64 MB row groups and keeps same-named inputs apart.
- `scripts/dedup_text.py` and `scripts/dedup_parquet.py` (+ tests): exact de-duplication across files (line-per-document text, `.gz` supported; Parquet text or v4.0), broken-encoding and short-document filters, counts report. `--preset 100m` (100.1M) and `300m` (303.4M) for budget-sized runs.
- `tokenizers/vocab32k_v2/` (+ `scripts/train_vocab.py`, `scripts/verify_vocab.py`): 32,000-token candidate vocabulary trained on a balanced 218 MB sample of the delivered data; lossless round-trip and 3003/3003 C++ parity lines verified; `.tok` files are now stored byte-exact (`.gitattributes`).
- `llm.train`: `1p5b` (1.509B) and `2b` (2.012B) presets (32 layers, d=2048, 16 query / 4 KV heads, 4K context; parameter counts checked against an exact formula), and multi-GPU training with DistributedDataParallel (`torchrun --nproc_per_node=N -m llm.train ...`; `--batch-size` is per rank; a 2-process run is tested to match one process with accumulation 2). Not tested on CUDA/NCCL.
- `scripts/pubmed_to_parquet.py` (+ tests): PubMed baseline XML (.xml.gz, loose or inside a ZIP) -> Parquet with a `text` column (title + abstract, English, one document per citation), streamed and resumable.
- `llm/` package with relative imports (replaces flat, generically named modules).
- `llm/dynamic_ntk.py`, `llm/rope_cache.py`: the team's Dynamic NTK and RoPE cache, integrated into `RotaryEmbedding`.
- `llm/checkpoint.py`: self-describing checkpoints, vocabulary / config verification, untrained-weights warning.
- `llm/data.py`: `.bin` token files (tokenizer report format), `TokenDataset`, document split, `NearDuplicateFilter`.
- `llm/train.py`: reference trainer (bit-exact resume, `stop_step`, bf16 autocast on CUDA).
- `llm/bpe_trainer.py`: fast pure-Python BPE trainer writing the `.tok` format (tooling/tests only).
- `TransformerModel.forward(return_hidden=…, rope_context_len=…)`, `LMBackboneAdapter` for the reward model.
- Scripts: `prepare_biology.py`, `eval_bio.py`, `prepare_earth.py`, `eval_earth.py`, `make_scaffold_checkpoint.py`, `mutation_check.py`.
- Tests (component by component), the team's original tests kept in `tests/legacy`, GPU handoff tests, C++ parity test (skipped until provided).
- Real-data smoke tests on two independent datasets (biology, Earth & Environment) — see `docs/BIOLOGY_SMOKE_TEST.md` and `docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`.

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
- Earth & Environment delivery used an incompatible `.bin` header (24-byte `GNRP` vs this repo's 36-byte `TOK1`); adapted in `prepare_earth.py` after independently re-verifying the delivery's own claims (finding F-29).
- Mutation check: a mutant that broke the BPE trainer's stale-cache-entry invariant caused an infinite loop instead of a clean test failure; fixed the mutant and added a 90s per-mutant timeout.
- `ModelConfig` validation uses exceptions instead of `assert`.

### Removed
- `ntk_scale_factor` (a static factor in no specification); `yarn.py` is a deprecated alias shim.
- The 206 MB `pretrain_model.pt` is **not** shipped (random weights); use `scripts/make_scaffold_checkpoint.py` or train one.
