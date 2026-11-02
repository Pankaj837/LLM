#!/usr/bin/env bash
# Rebuild every result and the PDF of reports/analysis_100m from the checkpoint and the data files.
#
#   RUN=path/to/aikosh BINS=path/to/bins bash scripts/reproduce_analysis.sh
#
# RUN holds ckpt_best.pt, ckpt_last.pt, metrics.jsonl and train_config.json (from the training run);
# BINS holds <domain>.train.bin and <domain>.val.bin for the domains below (team Drive folder).
# The SHA-256 of every input is written to results/inputs_sha256.txt; compare it with the one in the report.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RUN:?set RUN to the folder with the checkpoints}"
: "${BINS:?set BINS to the folder with the .bin files}"
PY=${PY:-python}
DEVICE=${DEVICE:-auto}
VOCAB=tokenizers/vocab32k_v2/vocab32k.tok
DOMAINS="math physics chem bio earth scidisc sciqa sciprog code cad multistep astro finance engineering gen_eng exp_val tech_comm"
METRICS=$(ls "$RUN"/metrics*.jsonl | head -1)
CONFIG=$(ls "$RUN"/train_config*.json | head -1)

mkdir -p results
{
  echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo none)"
  echo "python: $($PY -c 'import sys; print(sys.version.split()[0])')  torch: $($PY -c 'import torch; print(torch.__version__)')"
  shasum -a 256 "$RUN"/ckpt_best.pt "$RUN"/ckpt_last.pt "$METRICS" "$CONFIG" "$VOCAB" evals/MANIFEST.json
  for d in $DOMAINS; do shasum -a 256 "$BINS/$d.train.bin" "$BINS/$d.val.bin"; done
} > results/inputs_sha256.txt

VALS=""
for d in $DOMAINS; do VALS="$VALS $d=$BINS/$d.val.bin"; done
TRAIN=""
for d in $DOMAINS; do TRAIN="$TRAIN $BINS/$d.train.bin"; done

# shellcheck disable=SC2086
$PY scripts/check_leakage.py --vocab $VOCAB --bins $TRAIN --out results/leakage_all
for c in best:8250 last:8200; do
  $PY scripts/eval_science.py --ckpt "$RUN/ckpt_${c%%:*}.pt" --vocab $VOCAB --exclude results/leakage_all.json \
      --out "results/eval_science_100m_step${c##*:}" --device "$DEVICE"
done
# shellcheck disable=SC2086
$PY scripts/analyze_checkpoint.py --ckpt "$RUN/ckpt_best.pt" --compare-ckpt "$RUN/ckpt_last.pt" --metrics "$METRICS" \
    --config "$CONFIG" --vocab $VOCAB --val $VALS --out results/analysis_100m_step8250 --device "$DEVICE"
$PY scripts/eval_report_tex.py --results results/eval_science_100m_step8250.json --leakage results/leakage_all.json \
    --compare results/eval_science_100m_step8200.json --out reports/analysis_100m/generated_eval
$PY scripts/analysis_report_tex.py --analysis results/analysis_100m_step8250.json \
    --curve results/analysis_100m_step8250_curve.csv --eval results/eval_science_100m_step8250.json \
    --config "$CONFIG" --out reports/analysis_100m/generated
(cd reports/analysis_100m && tectonic main.tex)
echo "done: reports/analysis_100m/main.pdf; inputs in results/inputs_sha256.txt"
