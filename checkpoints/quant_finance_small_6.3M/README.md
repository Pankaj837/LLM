# Quant Finance model — small preset (6.3M parameters)

Trained with the Squad-1 LLM stack (github.com/Pankaj837/LLM) on the Quant Finance domain from the
PranagAI/LLMs-Datasets registry. See `Quant_Finance_Model_Guide.pdf` for the full explanation.

## Files

| File | What it is | Needed to use the model? |
|---|---|---|
| `quant_finance_small_6.3M.pt` | Trained weights + config + training state (step 900) | **Yes** |
| `vocab.tok` | The 8,000-token vocabulary the model was trained with | **Yes — the .pt will not load without this exact file** |
| `train_config.json` | Every training and model setting used | No (record) |
| `metrics.jsonl` | Loss curve, one line per logged step | No (record) |
| `eval.json` | Final scores on held-out data, baselines, sample outputs | No (record) |
| `data_manifest.json` | What data went in, what was excluded, integrity checks | No (record) |
| `scripts/` | `prepare_quant.py`, `eval_quant.py` — to reproduce the run | No |

## Use it

```python
from llm.pipeline import LLMPipeline          # after `pip install -e .` in the LLM repo
pipe = LLMPipeline.from_pretrained("quant_finance_small_6.3M.pt", "vocab.tok")
print(pipe.generate("The price of a European call option", max_new_tokens=40, temperature=0.8, top_p=0.9))
```

## Headline numbers (held-out documents)

Validation loss 4.955 · perplexity 142 · 1.754 bits per byte · beats the unigram baseline by 2.34 nats ·
1.35 nats worse on non-finance text than on finance text (it specialised to the domain).

## Reproduce

```bash
python scripts/prepare_quant.py --src "Domains/Quant Finance/domain_dataset" --out data/quant --vocab-size 8000
python -m llm.train --train-bin data/quant/train.bin --val-bin data/quant/val.bin --vocab data/quant/vocab.tok \
    --out runs/quant_small --preset small --seq-len 256 --batch-size 16 --max-steps 900 --lr 2e-3 \
    --eval-interval 50 --eval-batches 20 --seed 0
python scripts/eval_quant.py --run runs/quant_small/ckpt_best.pt --data data/quant --out runs/quant_small/eval.json
```
(`prepare_quant.py` imports one helper from the repo's `scripts/prepare_biology.py`, so run it from the LLM repo's `scripts/` folder.)
