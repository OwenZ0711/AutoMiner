# GP-CTA baseline

This is a small genetic-programming baseline for A-share long/flat CTA-style
strategy mining. It is intentionally separate from `STL pipelines/`, because
Pipeline A's planned v1 remains STL+MCTS.

## How it works

1. `features.py` builds simple per-symbol price/volume features from OHLCV.
2. `expressions.py` defines safe expression trees such as
   `ts_mean(div(close, ts_mean(close, 20)), 5)`.
3. `backtest.py` converts a formula's continuous signal into long/flat
   positions using a lagged rolling threshold, then scores next-day returns.
4. `evaluator.py` aggregates Sharpe, drawdown, turnover, trades, and time in
   market across symbols.
5. `gp.py` evolves formulas with selection, crossover, mutation, and elitism.

The demo script supports two data modes:

- `fixture`: reads the bundled 5-symbol parquet for network-free review.
- `cache`: reads `~/AutoLLM_data` through `AKshareParquetAdapter`; optionally
  pulls selected symbols with AKShare first.

Experiment outputs are written under `results/gp_cta/`. That directory is
ignored by git.

## Run the fixture demo

Install the demo dependencies into the shared venv:

```bash
~/venvs/autollm311/bin/python -m pip install -r requirements-gp-cta.txt
```

```bash
~/venvs/autollm311/bin/python scripts/run_gp_cta_demo.py \
  --population-size 40 \
  --generations 3 \
  --top-k 10
```

The default input is the existing offline fixture:

```text
data_fetcher/tests/fixtures/ohlcv_5sym_2018_2024.parquet
```

Outputs:

- `results/gp_cta/top_formulas.csv`
- `results/gp_cta/metrics.csv`
- `results/gp_cta/equity_curve.csv`
- `results/gp_cta/run_config.json`

## Run a larger CSI300 subset

The shortest command is:

```bash
./scripts/run_csi300_demo.sh
```

This pulls the first 10 current CSI300 symbols into `~/AutoLLM_data`, then runs
the same GP-CTA search on that larger local cache:

```bash
~/venvs/autollm311/bin/python scripts/run_gp_cta_demo.py \
  --data-mode cache \
  --universe csi300 \
  --limit-symbols 10 \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --train-end 2022-12-31 \
  --val-end 2023-12-31 \
  --pull-akshare \
  --sleep-s 5.0 \
  --max-retries 8 \
  --base-backoff-s 1.0 \
  --population-size 40 \
  --generations 3 \
  --top-k 10
```

After the first pull, omit `--pull-akshare` to reuse the local cache.

The shell wrapper supports simple overrides without editing code:

```bash
START=2019-01-01 END=2024-12-31 LIMIT_SYMBOLS=50 ./scripts/run_csi300_demo.sh
```

If AKShare disconnects frequently, slow down the pull further or reduce the
symbol count:

```bash
LIMIT_SYMBOLS=5 SLEEP_S=10 MAX_RETRIES=10 ./scripts/run_csi300_demo.sh
```

Reuse local cache without pulling AKShare again:

```bash
PULL_AKSHARE=0 ./scripts/run_csi300_demo.sh
```
