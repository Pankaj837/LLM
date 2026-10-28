# First H100 session: what it shows and what it does not (27 Oct 2026)

**Evidence level: owner-reported.** Gokul ran `scripts/gpu_session.py all` on an H100 on commit `bc1788f` (tag `scale-run-rc6`) and posted the generated
`session_report.md` (kept unchanged next to this file: `FIRST_H100_SESSION_REPORT_2026-10-27.md`) and the three validation-file checks. The checkpoint,
`session.json`, `train.log` and `metrics.jsonl` have **not** been seen by Pankaj yet; everything below is read from that report.

## What was measured
| | |
|---|---|
| Hardware | NVIDIA H100 80GB HBM3, torch 2.14.1+cu130 |
| Data | 13 domains in the mix (math, code, sciqa, bio, sciprog, earth, scidisc, chem, physics, cad, multistep, astro, finance): the session script refuses to start unless every token file passes `check_bins` (including the decode and re-encode vocabulary check) and the tokenizer hash equals the frozen one, so a wrong vocabulary cannot have been in the run |
| Smoke test (1.5B) | PASS: loss 8.70 → 7.94, 25,373 tokens/s, peak 60.8 GB, 4.5 min; checkpoint and resume worked |
| 100M run | 3,215 steps × 262,144 tokens = **842.8M tokens**, 63.1 min, 224,031 tokens/s (the benchmark predicted 225,631: agreement within 1%) |
| Loss | training 9.81 (step 10) → 2.216; validation 4.295 → **2.162** (best = last step: still falling) |
| Per-domain validation (mean 2.118) | code 1.28, cad 0.15, finance 1.40, multistep 1.63, scidisc 1.75, sciprog 1.80, sciqa 1.99, bio 2.59, chem 2.77, physics 2.96, earth 2.92, math 2.98, astro 3.30 |
| Cost | 69.0 min session clock ≈ ₹379 (at ₹330/h) |

Benchmark (synthetic tokens, best micro-batch that fits): 100M 225,631 tokens/s (₹406 per 1B tokens, MFU 18.9%); 300M 88,475 tokens/s (₹1,036 per 1B); 1.5B 24,645 tokens/s
(₹3,719 per 1B, 55 GB at micro-batch 4; micro-batch 8 runs out of memory). So **30B tokens on the 1.5B model would take about 338 H100-hours (≈ ₹112,000)**: out of reach, as estimated;
100M for 2B tokens ≈ 2.5 h (≈ ₹820); 300M for 6B tokens ≈ 19 h (≈ ₹6,200).

## What it proves
The whole chain works on real hardware: all owners' token files, one frozen vocabulary, the weighted mix, the training loop in bf16, checkpoint and resume, per-domain evaluation, and the
cost model (measured speed = benchmark speed). The loss curve is healthy (no NaN, falling to the end). It is the first model trained on the combined data, and a baseline other runs can be compared with.

## What it does not prove (read before quoting any number)
- **One run, 0.84B tokens, 100M parameters.** That is under half of the ≈ 2B tokens (20 per parameter) that size would take; the model is clearly under-trained, and the loss was still falling.
- **A low loss is not a capability.** `cad` 0.146 (perplexity 1.2) is almost certainly templated or near-duplicate text (40.5% of the CAD data is procedural): it says nothing about CAD understanding. `finance` and `multistep` are generated tables and arithmetic; `code` is highly regular. Treat these three as "the model learned the format".
- **Math may not be representative:** the report does not say which math file was used. If it is Prabanjan's 25 Oct build, it is the first 5% of the corpus (a prefix; the rerun with `spread` is pending) and its validation file comes from the same prefix. To be confirmed with `session.json` / the bins' `build_manifest.json`.
- **The chemistry data is poor** (checked 28 Oct in Ravi's `bins.zip`, which may or may not be the version used): about 22% exact duplicates, 17% of documents under 16 tokens, and in a sample roughly 43% code fragments cut at line level. Chemistry's loss (2.77) therefore measures partly code lines; the same applies to `physics`/`math` only if their files turn out to have similar defects (not checked). Details: `docs/PROGRESS_REPORT.md`.
- **No generation was shown.** Loss does not tell whether the model answers questions. `scripts/sample_outputs.py` (added 27 Oct, `scale-run-rc7`) prints real held-out inputs and outputs, next-token accuracy per domain, fixed probes with automatic pass/fail, and failure flags (repetition loops, empty or invalid output, copying the prompt). Expect most probes to fail at this size and token count: those are the failure cases Raju asked for (10:07 pm, 27 Oct).
- **No comparison point yet.** Without a baseline (the same data with a unigram or smaller model) "2.16" has no scale; the per-domain table is the baseline for the next run.

## Where the remaining budget goes (a recommendation, not a decision)
If the ₹1,000 credit is the only budget, about ₹620 (≈ 1.9 H100-hours) remains. At 224k tokens/s that is ≈ 1.5B more tokens: **continuing the same 100M model to about 2.4B tokens in total**, which is the size the plan was written for, and compare the per-domain table with this one. A 300M or 1.5B run does not fit; they stay a plan for when compute exists. Before it: upload the checkpoint (a notebook workspace can be deleted when the session ends), run `sample_outputs.py`, rebuild math with `spread`, add Ravi's and Prabanjan's missing domains if any are still partial.
