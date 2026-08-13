# `gp_cta/` — mining band (lead-lag detection + warm-start GP)

Consumes the data band's panel artifacts (`~/AutoLLM_data/futures/panel/`)
exclusively through `panel_io.PanelReader` — the bands share disk contracts,
never imports. Spec: [proposal.md](proposal.md); math:
`docs/LEADLAG_DESIGN.md`.

## Layers

| Layer | Modules | What it does |
|---|---|---|
| L3 scan | `leadlag/scan.py` | segment-blocked GEMM lagged-correlation tensor (zero-pad trick, 3-Gram joint-valid normalization) |
| L3 validate | `leadlag/validate.py` | asymmetry statistics (bounce-immune), leader-shifted stratified permutation nulls, BH/BY FDR, PIT trailing stability |
| L4 edges | `leadlag/edges.py` | append-only as-of edge table — the ONLY path by which GP touches foreign series |
| ledger | `ledger.py` | append-only trial ledger (the future DSR denominator) |
| GP | `gp/expressions.py`, `gp/ops.py`, `gp/terminals.py` | typed frozen-shape AST, vectorized eval kernels, as-of cross terminals |
| GP | `gp/backtest.py`, `gp/metrics.py`, `gp/fitness.py` | long/short next-open backtest (the execution-timing chokepoint), full metrics suite, fitness + one-time sign flip |
| GP | `gp/warmstart.py`, `gp/engine.py` | Ren–Qin–Li frozen-structure warm start, sequential seed runs, correlation-gated candidate pool |
| gates | `gates/` | band 3 stub — see its docstring for the ledger contract |

## CLI

```bash
python -m gp_cta scan     --slice acceptance     # < 1 s on the 3-product slice
python -m gp_cta validate --slice acceptance     # full null (200 draws) + PIT monthly edge history
python -m gp_cta mine     --slice acceptance --init warmstart
python -m gp_cta mine     --slice acceptance --init random     # A/B baseline
python -m gp_cta check leadlag --slice acceptance
python -m gp_cta check gp      --slice acceptance
```

Look-ahead gates wired in and tested:
- edge `as_of` never returns same-day or future rows (`tests/test_validate.py`);
- cross terminals resolve month-by-month, frozen at m−1, lags clamped to the
  validated ℓ*;
- the `+1-bar kill-switch` re-scores the best formula on shifted terminals —
  reported in every mining manifest;
- backtest thresholds are shifted one bar; fills at t+1 open; limit-locked
  bars defer fills; the 2025 hole forces flat.

Empirical note (2019–2021 acceptance slice): the black-chain link validates
as **RB→I at lag 1** (q ≈ 0.015) — the more liquid rebar leg leads iron ore
at minute scale, opposite to the naive supply-chain prior.

Tests: `pytest gp_cta -q`; the flagship full-pipeline regression is
`tests/e2e/test_pipeline_planted.py`.
