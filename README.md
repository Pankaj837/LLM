# LLM Squad-1 — consolidated codebase

A from-scratch decoder-only language-model stack: **grouped-query attention** with KV cache, **RoPE with Dynamic NTK / YaRN**
context extension, a byte-level **BPE tokenizer** (compatible with the C++ `.tok` format), a text **pipeline**, a reference
**trainer** on the tokenizer team's `.bin` files, and a **conditional reward model**.

Organised by component, verified component by component. Start with `docs/TEAM_INSIGHTS.md` (what was kept, what changed, what
is still open) and `docs/HANDOFF_GPU.md` (what to run on a GPU).

## Status

| Component | State |
|---|---|
| Config, blocks, GQA + KV cache, generation | verified against independent references; the team's original tests pass unchanged |
| RoPE / Dynamic NTK / YaRN | per the team's reports and the YaRN paper; cached ≡ uncached at logit level |
| Tokenizer | Python port verified; **C++ parity verified** — built from the real source and tested, see `docs/CPP_TOKENIZER_PARITY.md` |
| Checkpoint / pipeline | self-describing, fails loudly; **no trained reference checkpoint exists yet** |
| Trainer + `.bin` data | proven on 2 real datasets (biology, Earth & Environment), bit-exact resume |
| Reward model | mechanism verified; **synthetic data, not evidence of quality** |
| GPU / bf16 / torch.compile | **untested** — see `docs/HANDOFF_GPU.md` |
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
data/      README only (datasets are external/generated, never committed — see data/README.md)
tests/     one file per component  +  legacy/ (the team's original tests, unchanged)  +  results/ (last real run's output)
scripts/   smoke_test.py  prepare_biology.py  eval_bio.py  prepare_earth.py  eval_earth.py
           make_scaffold_checkpoint.py  mutation_check.py  lint.py
docs/      ARCHITECTURE  COMPONENTS  DECISIONS  REVIEW_FINDINGS  TESTING  HANDOFF_GPU  TEAM_INSIGHTS  REWARD_MODEL
           BIOLOGY_SMOKE_TEST  EARTH_ENVIRONMENT_SMOKE_TEST  CPP_TOKENIZER_PARITY
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
