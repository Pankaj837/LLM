# LLM — consolidated codebase

A from-scratch decoder-only language-model stack: **grouped-query attention** with KV cache, **RoPE with Dynamic NTK / YaRN**
context extension, a byte-level **BPE tokenizer** (compatible with the C++ `.tok` format), a text **pipeline**, a reference
**trainer** on the tokenizer team's `.bin` files, and a **conditional reward model**.

Organised by component, verified component by component. **Where things stand and what is open: `docs/PROGRESS_REPORT.md`
(top section) and `docs/INDEX.md` (which document is current and which is history).** The design notes are
`docs/TEAM_INSIGHTS.md` (what was kept, what changed) and `docs/HANDOFF_GPU.md` (what to run on a GPU).

## Status (5 Nov 2026)

| Component | State |
|---|---|
| Config, blocks, GQA + KV cache, generation | verified against independent references; the team's original tests pass unchanged |
| RoPE / Dynamic NTK / YaRN | per the team's reports and the YaRN paper; cached ≡ uncached at logit level |
| Tokenizer | Python port verified; **C++ parity verified** — see `docs/CPP_TOKENIZER_PARITY.md`. The 32K vocabulary `vocab32k_v2` is **frozen** (25 Oct; sha256 `10b3c5ae…b51cdc`, `tokenizers/FROZEN.md`, guarded by a test) |
| Checkpoint / pipeline | self-describing, fails loudly. Since 4 Nov: `ckpt_last.pt` always carries the latest validation loss, rolling backups, `--resume auto`, graceful stop (Ctrl-C / SIGTERM / `STOP` file) — `docs/CHECKPOINT_RELIABILITY_2026-11-04.md`. The only real trained checkpoint of the current line is the **100M run** (step 8,250, validation loss 2.166, 1.2 GB, **not in this repository**; hashes in `docs/A100_RUN_RESULT_2026-11-01.md`). The two 6.3M files in `checkpoints/` are toys |
| Trainer + `.bin` data | proven on 2 real datasets (biology, Earth & Environment), bit-exact resume (also demonstrated after a hard kill: `scripts/resume_demo.py`) |
| Scale-up presets `100m` / `300m` / `1p5b` / `2b`, multi-GPU DDP, weighted domain mix, atomic checkpoints | **GPU-tested on one GPU**: 100M run on an H100 (27 Oct, 63 min, owner-reported) and on an A100 20 GB slice (31 Oct to 1 Nov, 2.2B tokens); 1.5B smoke test on one H100 (25.4k tokens/s, 60.8 GB, owner-reported). **Multi-GPU (DDP/NCCL) and a full 1.5B run have not been run** — `docs/PILOT_PLAN_2026-11-04.md` |
| Data pipeline (count → dedup → tokenize → check → evaluate per domain) | tested on synthetic data and on the real bins of the team; state of the 30B data plan: `docs/DATA_PLAN_30B_2026-11-01.md` |
| Evaluation | per-domain validation loss and `sample_outputs.py` here; Namit's multiple-choice set and scorer (`scripts/eval_science.py`, on branch `namit/science-eval`) were reproduced exactly on CPU: `docs/EVAL_REPRODUCTION_2026-11-04.md`; `scripts/compare_eval.py` checks any new run against a reference |
| Reward model | mechanism verified; **synthetic data, not evidence of quality** |
| bf16 / torch.compile | bf16 training was used in the H100 run (owner-reported, `docs/FIRST_RUN_ANALYSIS_2026-10-27.md`); no recorded use of `torch.compile`; no numerical comparison against fp32 on a trained model |
| Structured pruning | not part of this repository |

Test totals and the mutation-check result are in `docs/TESTING.md`.

## Quickstart

Source lives under `src/llm/` (standard Python "src layout"). Install it in editable mode once — after that,
`import llm` and `python -m llm.<module>` work from anywhere, no `PYTHONPATH` needed:

```bash
pip install -e ".[dev]"                # installs torch, numpy, pytest and this package (editable)
python -m pytest                       # CPU suite (GPU / real-data tests skip themselves)
python scripts/smoke_test.py           # the team's 29-check smoke script
python scripts/lint.py                 # dependency-free lint
```
(pytest itself does not need the install — its `pythonpath = ["src"]` setting in `pyproject.toml` already finds
the package; the install is what makes `python -m llm.train` etc. work as plain shell commands.)

```python
import torch
from llm import ModelConfig, TransformerModel          # or: from llm.model import TransformerModel
from llm.pipeline import LLMPipeline

# 1. a model from config (random init) — 51.5M parameters by default
model = TransformerModel(ModelConfig())
logits = model(torch.randint(0, 8000, (1, 32)))         # (1, 32, 8000)

# 2. text in / text out with a trained checkpoint + its vocabulary
pipe = LLMPipeline.from_pretrained("runs/bio_small/ckpt_last.pt", "data/bio/vocab.tok")
print(pipe.generate("DNA replication begins when", max_new_tokens=40, temperature=0.8, top_p=0.9))
```

### Train on your own text
```bash
# text -> .bin: use the team's C++ prepare_dataset, or (Python stand-in) see scripts/prepare_biology.py
python -m llm.train --train-bin data/train.bin --val-bin data/val.bin --vocab data/vocab.tok \
    --out runs/mine --preset small --seq-len 256 --batch-size 16 --max-steps 1400
python scripts/make_scaffold_checkpoint.py --out checkpoints/scaffold_untrained.pt    # wiring tests only (flagged untrained)
```

### Real-data smoke test (biology)
```bash
python scripts/prepare_biology.py --zip "Biology Dataset.zip" --out data/bio --vocab-size 8000   # ~1.5 min
python -m llm.train --train-bin data/bio/train.bin --val-bin data/bio/val.bin --vocab data/bio/vocab.tok \
    --out runs/bio_small --preset small --seq-len 256 --batch-size 16 --max-steps 1400 --lr 2e-3       # ~25 min CPU
python scripts/eval_bio.py --run runs/bio_small/ckpt_last.pt --data data/bio --out runs/bio_small/eval.json
```
Results: `docs/BIOLOGY_SMOKE_TEST.md`.

### Real-data smoke test (Earth & Environment — a differently-tokenized third-party delivery)
```bash
python scripts/prepare_earth.py --zip "Earth and Environment Processed Dataset.zip" --out data/earth   # ~5 s
python -m llm.train --train-bin data/earth/train.bin --val-bin data/earth/val.bin \
    --out runs/earth_small --preset small --seq-len 256 --batch-size 16 --max-steps 900 --lr 2e-3   # ~13 min CPU
python scripts/eval_earth.py --run runs/earth_small/ckpt_last.pt --data data/earth --out runs/earth_small/eval.json
```
This delivery's `.bin` files use a different (incompatible) binary header than this repo's own — `prepare_earth.py`
verifies and adapts it (finding F-29). Results: `docs/EARTH_ENVIRONMENT_SMOKE_TEST.md`.

### Data pipeline and scale-up (Oct 2026)
```bash
pip install -e ".[dev]"                                               # includes pyarrow for the Parquet tools
python scripts/tokenize_parquet.py "data/*.parquet" --count-only --out runs/count      # seconds: rows, chars, ~tokens per domain
python scripts/dedup_parquet.py "biology/*.parquet" --out cleaned --expect-files 63     # exact dedup across files
python scripts/dedup_text.py corpus.txt.gz --out cleaned_text                           # text corpora (--records blank for Q&A blocks)
python scripts/tokenize_parquet.py "cleaned/*.parquet" --vocab tokenizers/vocab32k_v2/vocab32k.tok --out runs/bins --no-merge
python scripts/tokenize_text.py math=cleaned_text/corpus.txt.gz --vocab tokenizers/vocab32k_v2/vocab32k.tok --out runs/bins_text
python scripts/merge_bins.py "runs/bins/code*.train.bin" --out bins/code.train.bin      # one train and one val file per domain (big inputs are split into jobs)
python scripts/check_bins.py bins/*.bin --vocab tokenizers/vocab32k_v2/vocab32k.tok              # integrity gate
torchrun --nproc_per_node=8 -m llm.train --train-mix bins/code.train.bin=0.6 bins/math.train.bin=0.4 --val-bin val.bin \
    --vocab tokenizers/vocab32k_v2/vocab32k.tok --out runs/1p5b --preset 1p5b --seq-len 2048 --batch-size 4 --grad-accum 8
python scripts/eval_domains.py --ckpt runs/1p5b/ckpt_last.pt --val math="runs/bins/math*.val.bin" code=runs/bins/code.val.bin
```
One command from raw files to a checked, mixed training set (capped per domain), and a measured GPU benchmark:
```bash
python scripts/build_dataset.py configs/run_100m.example.json      # tokenize + merge per domain + integrity check + prints --train-mix
python scripts/benchmark.py --presets 100m 300m 1p5b --batch-sizes 1 2 4 8 16 --seq-len 2048 --out bench.json     # tokens/s, memory, rupees per 1B tokens
```
Recipe, budget, mix proposal, the 3-hour GPU session, smoke test and the pre-launch checklist: `docs/TRAINING_PLAN_1p5B.md`;
the 15 capabilities and what data exists for each: `docs/CAPABILITIES.md`.

### C++ tokenizer parity (verified)
```bash
cd native/tokenizer && cmake -B build && cmake --build build && cd ../..   # needs a C++17 compiler
./native/tokenizer/build/prepare_dataset <a text corpus> vocab.tok 8000    # train a real vocabulary
TOK_CLI=native/tokenizer/build/encode_cli TOK_VOCAB=vocab.tok python -m pytest tests/test_cpp_parity.py -v
```
Results: `docs/CPP_TOKENIZER_PARITY.md` (46/46 on the team's own C++ test suite; identical token ids on every
parity case between the C++ and Python tokenizers, on a real trained vocabulary).

## Layout

```
src/llm/
  model_config.py  modules.py  dynamic_ntk.py  rope_cache.py  rope.py  gqa.py  model.py     the model
  tokenizer.py  bpe_trainer.py  data.py                                                     text ↔ ids ↔ .bin
  checkpoint.py  pipeline.py  train.py                                                      persistence, inference, training
  reward/                                                                                   conditional reward model
native/tokenizer/  the team's C++ tokenizer, as delivered (+ one added test harness) — see native/README.md
checkpoints/  the 51M submissions of September/October and two TOY 6.3M files (integration work only) — see checkpoints/README.md
tokenizers/   the frozen 32K vocabulary (vocab32k_v2) and the unused digits candidate — see tokenizers/FROZEN.md
configs/   example and draft run configs (build_dataset.py / gpu_session.py input)
data/      README only (datasets are external/generated, never committed — see data/README.md)
tests/     one file per component  +  legacy/ (the team's original tests, unchanged)  +  results/ (last real run's output)
scripts/   data:   tokenize_parquet  tokenize_text  dedup_parquet  dedup_text  merge_bins  check_bins  build_dataset
                   train_vocab  verify_vocab  pubmed_to_parquet  prepare_biology  prepare_earth
           run:    gpu_session  benchmark  resume_demo
           eval:   eval_domains  sample_outputs  compare_eval  eval_bio  eval_earth
           checks: smoke_test  mutation_check  lint  make_scaffold_checkpoint
docs/      start with INDEX.md (current documents versus history); status: PROGRESS_REPORT, PREPILOT_TASKS_STATUS
archive/   deprecated/uncertain files land here if any are found later — see archive/README.md for what
           was considered and why nothing is in it yet
```

## Ground rules (enforced by tests)
1. **One vocabulary end to end.** Model `vocab_size` = tokenizer `n_vocab` = `.bin` header; the vocabulary hash is stored in checkpoints.
2. **Hand off text, not ids, between models with different tokenizers.**
3. **Right-pad** everything; left padding is unsupported.
4. **User text can never inject control tokens.**
5. **Fail loudly**: missing files, mismatched configs or vocabularies raise; untrained weights warn on every load.
6. **Dynamic NTK frequencies are fixed per call** from the planned context (`docs/DECISIONS.md` D1).

## Known limitations
See `docs/REVIEW_FINDINGS.md` (“Not covered”) and `docs/HANDOFF_GPU.md` §6.
