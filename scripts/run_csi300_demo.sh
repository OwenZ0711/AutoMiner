#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/autollm311/bin/python}"
LIMIT_SYMBOLS="${LIMIT_SYMBOLS:-10}"
START="${START:-2020-01-01}"
END="${END:-2024-12-31}"
TRAIN_END="${TRAIN_END:-2022-12-31}"
VAL_END="${VAL_END:-2023-12-31}"
SLEEP_S="${SLEEP_S:-5.0}"
MAX_RETRIES="${MAX_RETRIES:-8}"
BASE_BACKOFF_S="${BASE_BACKOFF_S:-1.0}"
POPULATION_SIZE="${POPULATION_SIZE:-40}"
GENERATIONS="${GENERATIONS:-3}"
TOP_K="${TOP_K:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gp_cta}"
PULL_AKSHARE="${PULL_AKSHARE:-1}"

pull_arg=""
if [[ "$PULL_AKSHARE" == "1" || "$PULL_AKSHARE" == "true" ]]; then
  pull_arg="--pull-akshare"
fi

"$PYTHON_BIN" scripts/run_gp_cta_demo.py \
  --data-mode cache \
  --universe csi300 \
  --limit-symbols "$LIMIT_SYMBOLS" \
  --start "$START" \
  --end "$END" \
  --train-end "$TRAIN_END" \
  --val-end "$VAL_END" \
  $pull_arg \
  --sleep-s "$SLEEP_S" \
  --max-retries "$MAX_RETRIES" \
  --base-backoff-s "$BASE_BACKOFF_S" \
  --population-size "$POPULATION_SIZE" \
  --generations "$GENERATIONS" \
  --top-k "$TOP_K" \
  --output-dir "$OUTPUT_DIR"
