# Independent verification of Om's original 6.3M Quant toy model

Checkpoint strict-loads cleanly into this repo's `small` preset (6,278,400 params, 0 missing/unexpected keys).
`vocab.tok`'s SHA-256 matches the checkpoint's own recorded `vocab_sha256` exactly. Tokenizer round-trips
correctly. Spectral test ratios 2.1–3.2× the random-matrix edge — genuinely trained. `metrics.jsonl`'s last
line (train_loss 4.5983, val_loss 4.9555 at step 900) matches the checkpoint's internal meta exactly. This was
Om's first, small-scale announcement ("trained quant finance dataset on 6.3M parameters Model") ahead of the
full 51.5M submission in `../quant_51m/`. Checkpoint (`quant_finance_small_6.3M.pt`, 72 MB) not committed;
SHA-256: `a154829295a6842e55e9b44755b76d07cb9669b1b5bc4dbdc45bc1996504589d`.
