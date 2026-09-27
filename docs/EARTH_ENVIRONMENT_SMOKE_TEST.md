# Earth & Environment real-data smoke test

**Purpose.** Same goal as `docs/BIOLOGY_SMOKE_TEST.md`: prove the whole path works on real, externally-prepared
data. This dataset was delivered **already tokenized** by another contributor, so it also exercises a case
`docs/BIOLOGY_SMOKE_TEST.md` did not: **consuming someone else's preprocessing output**, not just raw text.

## 0. What this data actually is

Not natural text originally — **71 CSV files (580 MB)** of agriculture, climate, earthquake and volcano
records. Each row was converted to a sentence by a per-file template (a "verbalization" step) before
tokenizing, e.g. *"On 1979-03-01, the average land temperature in Guizhou, China was 11.44 degrees C
(uncertainty +/-0.18)."* This makes the corpus **highly templated**: a small number of sentence shapes with
different names/dates/numbers, not free-form prose like the biology corpus. That difference explains the very
different numbers below — see §4.

## 1. Integration finding: two incompatible binary header formats

**This is exactly the kind of interface mismatch this whole review exercise exists to catch, and it was real.**

The tokenizer team's `.bin` spec (`docs/COMPONENTS.md` C10, reproduced in `src/llm/data.py`) is a **36-byte** header,
magic `b"TOK1"`. This dataset's `train.bin`/`val.bin` use a **24-byte** header, magic `b"GNRP"`, with no
`split`/`reserved` fields:

| Field | This repo (`src/llm/data.py`) | Earth & Environment delivery |
|---|---|---|
| Header size | 36 bytes | 24 bytes |
| Magic | `TOK1` (`0x544F4B31`) | `GNRP` (`0x50524E47`) |
| Struct | `<IIIIQIQ>` (+ split, reserved) | `<IIIIQ>` |

`llm.data.read_header` raises `ValueError: bad magic` on these files — **not a bug, correct behaviour** (a
silent wrong-length read would misinterpret the token stream). `scripts/prepare_earth.py` reads the delivered
format directly, verifies it independently, and re-emits the same token ids through `llm.data.write_bin` so
the rest of the repo (`llm.train`, evaluation) needs no further changes. Recorded as finding **F-29** in
`docs/REVIEW_FINDINGS.md`.

## 2. Independent verification (not just trusting the delivery's own summary)

| Check | Train | Val |
|---|---|---|
| Header-declared vs actual token count | 3,544,700 / 3,544,700 ✓ | 394,189 / 394,189 ✓ |
| File size matches header | ✓ | ✓ |
| All ids < vocab (8192) and < 65536 | ✓ | ✓ |
| EOS count (independently counted) vs delivery's claimed document count | 99,828 vs 99,828 ✓ | 11,092 vs 11,092 ✓ |
| Decode round-trip via the delivered `tokenizer.json` (Hugging Face byte-level BPE) | 3/3 sample documents decode to clean, readable sentences | |
| Val 32-token windows also seen in train | **1.07%** | |

Every number the delivery's `PROCESSING_SUMMARY.md` claimed was reproduced exactly by our own, independent
re-parse. The 1.07% window overlap is higher than biology's post-fix 0.3% but far below its pre-fix 4.9% — expected
here, since many rows share an identical sentence *template* and only the embedded numbers differ, so template
scaffolding tokens legitimately recur across train and val without being duplicate documents.

## 3. Training

`small` preset (6.33M params) on 3.54M train tokens, 900 steps × 16 × 256, seed 0 — same protocol as biology.

| Step | Val loss | Val perplexity |
|---:|---:|---:|
| 0 (uniform) | 9.01 | 8192 |
| 150 | 1.40 | 4.04 |
| 300 | 1.33 | 3.78 |
| 450 | 1.28 | 3.58 |
| 600 | 1.23 | 3.41 |
| **900 (final)** | **1.18** | **3.24** |

| Metric | Value |
|---|---:|
| Unigram baseline (fitted on train, evaluated on val) | 5.40 |
| **Model beats unigram by** | **4.22 nats** |

## 4. Why this looks so much "better" than the biology run — read before comparing

| | Biology | Earth & Environment |
|---|---:|---:|
| Val loss after training | 4.93 | **1.18** |
| Unigram baseline | 7.40 | 5.40 |
| Gap over unigram | 2.44 nats | **4.22 nats** |

This is **not** evidence that the earth/environment model is "better." The two corpora have very different
entropy: biology is free-form scientific prose (high per-token uncertainty even for a good language model);
this corpus is machine-generated from **a few dozen sentence templates** with slots for names, dates and
numbers. A model can memorise the templates almost perfectly (that's most of the 900-step drop) and its
remaining loss is dominated by the genuinely unpredictable parts — the actual numbers and place names, which
still take a large fraction of the vocabulary's probability mass, but far less than modelling open-ended prose.
**Compare loss/perplexity only within a domain, never across domains** (`docs/BIOLOGY_SMOKE_TEST.md` §9 makes the
same point about bits-per-byte).

## 5. Samples (qualitative)

```
prompt:  "On 2001-06-01, the average land temperature in"
greedy:  "...New York, United States was 20.80 degrees C (uncertainty +/-0.20).
          On 04/03/2019, the Potato (Other) market in Pune(Hadapsar), Kangra, Himachal..."

prompt:  "In 2010,"
greedy:  "...during the Kharif season, Andhra Pradesh produced Rice over an area of
          5444000.0 hectares, yielding 5200 tonnes (yield ratio 0.6), using 4026000.0
          units of fertilizer..."

prompt:  "The earthquake"
greedy:  "...(mww) occurred at Port-Vila, Vanuatu on 16-03-2022 14:03, at a depth of
          10.0km. Alert level: nan. Tsunami risk flag: 0."
```
The model reproduces template **structure** correctly (field order, units, formatting) after only 900 steps —
consistent with §4's explanation — while the specific place names, dates and numbers are invented (as expected;
6.3M parameters, ~1 epoch).

## 6. What this does not show
- Nothing about the 51.5M reference model (needs a GPU, `docs/HANDOFF_GPU.md`).
- This corpus's own tokenizer (Hugging Face byte-level BPE, vocab 8192) is used purely as delivered; it is a
  **different vocabulary** from both this repo's `.tok` format and the biology run's — do not mix ids across runs.
- No held-out non-templated control set exists for this domain (unlike biology's Gutenberg control).
- Single seed.

## 7. Reproduce
```bash
python scripts/prepare_earth.py --zip "Earth and Environment Processed Dataset.zip" --out data/earth
python -m llm.train --train-bin data/earth/train.bin --val-bin data/earth/val.bin \
    --out runs/earth_small --preset small --seq-len 256 --batch-size 16 --max-steps 900 \
    --warmup-steps 80 --lr 2e-3 --eval-interval 150 --eval-batches 40 --seed 0
python scripts/eval_earth.py --run runs/earth_small/ckpt_last.pt --data data/earth --out runs/earth_small/eval.json
EARTH_ZIP="Earth and Environment Processed Dataset.zip" python -m pytest tests/test_prepare_earth.py -q
```
