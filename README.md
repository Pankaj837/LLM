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
| Tokenizer | Python port verified; **C++ parity unverified** (binary and real `vocab.tok` not available) |
| Checkpoint / pipeline | self-describing, fails loudly; **no trained reference checkpoint exists yet** |
| Trainer + `.bin` data | proven on real text (biology corpus), bit-exact resume |
| Reward model | mechanism verified; **synthetic data, not evidence of quality** |
| GPU / bf16 / torch.compile | **untested** — see `docs/HANDOFF_GPU.md` |
| Structured pruning | not part of this repository |

Test totals and the mutation-check result are in `docs/TESTING.md`.

## Quickstart

```bash
pip install "torch>=2.4" numpy pytest
python -m pytest                       # CPU suite (GPU / real-data tests skip themselves)
python scripts/smoke_test.py           # the team's 29-check smoke script
python scripts/lint.py                 # dependency-free lint
```

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

## Layout

```
llm/
  model_config.py  modules.py  dynamic_ntk.py  rope_cache.py  rope.py  gqa.py  model.py     the model
  tokenizer.py  bpe_trainer.py  data.py                                                     text ↔ ids ↔ .bin
  checkpoint.py  pipeline.py  train.py                                                      persistence, inference, training
  reward/                                                                                   conditional reward model
scripts/   smoke_test.py  prepare_biology.py  eval_bio.py  make_scaffold_checkpoint.py  mutation_check.py  lint.py
tests/     one file per component  +  legacy/ (the team's original tests, unchanged)  +  test_gpu_handoff.py
docs/      ARCHITECTURE  COMPONENTS  DECISIONS  REVIEW_FINDINGS  TESTING  HANDOFF_GPU  TEAM_INSIGHTS  REWARD_MODEL  BIOLOGY_SMOKE_TEST
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
