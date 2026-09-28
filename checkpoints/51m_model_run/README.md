# 51M model run — training config and metrics (submitted by Prabanjan)

**Status: training run itself looks genuinely complete and healthy. The `.pt` checkpoint file has not been
submitted yet — only `train_config.json` and `metrics.jsonl` are here.** Everything below is independently
verified against the raw `metrics.jsonl`, not taken on the submitter's word.

## Is it done?

**Yes, this specific run (5,000 steps) completed cleanly.**
- `metrics.jsonl` has exactly 500 lines, steps 10 → 5000 in strict, unbroken order (`log_interval=10`), no
  duplicate or missing steps.
- Loss and perplexity fall monotonically with no NaN/Inf anywhere: val loss 6.68 → 3.84, val perplexity
  **794.5 → 46.4**.
- The LR schedule behaves exactly as configured: peaks at `0.0006` (the configured `lr`), and floors at
  `6.00e-05` precisely at step 5000 — that's `min_lr_ratio (0.1) × peak (0.0006)`, confirming the cosine decay
  is correctly stretched across the *full* 5,000-step run rather than floor-decaying early. This is the exact
  fix Raju sir asked for (§2 below) — it's applied and it worked.

## Cross-checked against Raju sir's Sept 2026 feedback (verified where possible from this data alone)

| His point | Verified? | What the data actually shows |
|---|---|---|
| Increase to 5,000–10,000 steps; fix scheduler so decay doesn't hit the floor early | ✅ Done | `max_steps: 5000`, `resume: null` in the config; LR floors exactly at step 5000, not earlier |
| `seq_len` too low vs `max_position_embeddings` | ✅ Done | `seq_len: 512` in this config (was 256) |
| Enable bf16 + `torch.compile()`, target >2,000 tok/s (from ~540) | ✅ **Exceeded** | `dtype: bf16`, `compile: true`; measured throughput **~5,245 tok/s sustained** — 2.6× his target, ~9.7× the old baseline |
| "Perplexity (127.45) verifies loss convergence" | ⚠️ **Doesn't match this file** | No step in this run has ppl 127.45. Closest is step 1400 (ppl 126.69) — an **interim** point. By step 5000 (this run's actual end) perplexity is **46.4**, far better. Either his comment was made on an earlier/partial view of this run, or on a different checkpoint — ask which, don't assume 127.45 is the final number. |
| Duplicate print logs between steps 1010–1250 | ❓ **Can't verify from this file** | `metrics.jsonl` itself has no duplicate step entries in that range. If the duplication was in console/stdout output, that log wasn't included here — ask for it if it still matters. |
| Run generation quality checks (T=0.7, top-p 0.9) at step 2,000 | ❌ Not included | No sample generations in either file submitted |
| Acquire compiled C++ binary + 32k/64k vocab for parity testing | ❌ Separate, unrelated task | Not something this training run produces; still outstanding (see `docs/CPP_TOKENIZER_PARITY.md` for the 8k-vocab parity work already done — this asks for a *larger* vocabulary, not yet acquired) |

## Config as submitted

`vocab_size: 8193` (`../final_vocab_51m.tok`, not one of the vocabularies verified so far in this repo — its
provenance/hash isn't recorded here; ask for it alongside the checkpoint). Model architecture otherwise matches
this repo's `base` preset exactly (11 layers, d=640, 8 query/2 KV heads, SwiGLU 1664, `max_position_embeddings=2048`).

## What's still needed before this can replace the toy checkpoints in `checkpoints/`

1. **The actual `.pt` file** (`ckpt_last.pt` or `ckpt_best.pt`) — not submitted yet.
2. **`final_vocab_51m.tok`** — the checkpoint is only usable with the exact vocabulary it was trained on.
3. Confirmation on the 127.45-perplexity discrepancy above.
4. Ideally: which dataset(s) this was trained on, and whether it's the combined corpus across the team
   (`docs/HANDOFF_GPU.md` §4 roster) or one person's alone.

Once all four arrive, this replaces `checkpoints/toy_bio_small_6.3M.pt` and `checkpoints/toy_earth_small_6.3M.pt`.
