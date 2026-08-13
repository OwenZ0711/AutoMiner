"""Band 3 — admission gates (DEFERRED PHASE, deliberately not built).

Everything this band needs is ALREADY being produced:

- ``results/trial_ledger/part-*.parquet`` — every hypothesis examined: every
  (pair, window) the lead-lag scan looked at (kind=``ll_pair_scan``), every
  GP formula backtested (``gp_formula``), every one-time sign flip
  (``gp_sign_flip``). ``gp_cta.ledger.scan_ledger(paths)`` reads it; the
  total row count is the Deflated-Sharpe trial denominator.
- Per-trial full metrics (``metrics_json``): Sharpe, Sortino, maxDD, Calmar,
  hit rate, profit factor, skewness, kurtosis, PSR, n_days — the DSR inputs.
- ``results/pool/*.parquet`` — candidate formulas with train/validation
  fitness and the mutual-correlation gate verdict.

To build here later: Deflated Sharpe Ratio (Bailey & López de Prado, using
ledger count + metric moments), BHY FDR across the candidate pool, and CPCV
(combinatorially-purged cross-validation) — then a holdout scoring path that
runs EXACTLY once per mining round (`paths.holdout_marker`).
"""
