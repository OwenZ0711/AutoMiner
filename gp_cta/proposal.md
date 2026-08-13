# GP-CTA futures factor mining — build guide

> **Read this first.** This file is the complete context for the GP-CTA part of
> this project. A Claude Code agent given this file should know enough to
> build any part of the system without asking for background. The
> memory-safety and look-ahead-safety sections override everything else,
> including speed and convenience. Deeper math lives in
> `docs/LEADLAG_DESIGN.md`; this file is the buildable spec — every default an
> agent needs is pinned here.

## Mission

Build a factor-mining factory on Chinese futures 1-minute data. Three bands:

1. **Prepare once** (per data drop): clean the raw CSVs, build our own
   continuous contract prices, build a time-aligned feature panel, and scan
   for validated lead-lag pairs (product i's moves precede product j's).
2. **Mining loop**: a genetic-programming (GP) engine evolves trading
   formulas from those features and lead-lag pairs, warm-started from proven
   structures, scored by cost-adjusted long/short backtests.
3. **Judge once per round**: out-of-sample tests and multiple-testing gates
   admit survivors into a strategy pool.

Target alpha horizon: minutes to a few days (mid/high frequency CTA).

**Search method: genetic programming ONLY.** No reinforcement learning in
this build — RL miners are explicitly a later version (v2). Where AlphaGen
is cited below, we borrow its engineering (typed AST, cached panel fitness,
correlation-gated pool), which is method-neutral — NOT its RL search.

---

## Memory safety — TOP PRIORITY

The machine is a MacBook M1, 8 cores, **16 GB RAM**; the raw data is 68 GB. A
careless full load can freeze the user's machine. Non-negotiable rules:

1. **Process in the smallest natural snippet**: one contract CSV, one
   product-year, or one day folder at a time. Write results to disk, release,
   move on. Never "read the whole folder first."
2. **Hard budget: < 4 GB peak RSS per stage.** If a step could exceed it,
   split the batch. When in doubt, use the smaller batch.
3. **Lazy reads only**: Polars `scan_csv`/`scan_parquet` with column pushdown
   and streaming collect. Eager `read_*` of more than one snippet is a bug.
4. **Small types**: float32 for prices/features, boolean/uint8 masks,
   categorical symbols. Drop unused columns at scan time.
5. **Disk over RAM**: intermediates go to parquet; big arrays are
   memory-mapped (`np.memmap`), not held.
6. **Parallelism is capped by memory, not cores**: 8 workers × 2 GB = dead
   machine. Compute per-worker peak first. Default to sequential.
7. **Test tiny, measure, then scale**: every stage runs on 1 product × 1 year
   first with peak RSS measured (`psutil.Process().memory_info().rss` or
   `/usr/bin/time -l`) and printed. Full-history passes are opt-in flags.
8. **Subsets are fine.** Development and validation use ~10 liquid products ×
   3 years. You do not have to touch all the data.

---

## Look-ahead safety — TOP PRIORITY (co-equal with memory)

No artifact at time t may depend on information revealed after t. Enforced by
construction plus automated gates, never by review alone:

1. **Point-in-time continuous prices.** The roll adjustment is
   **segment-START anchored (forward/PIT)**: `adj_factor(bar)` = product of
   the factors of rolls **at or before** that bar (= 1.0 at segment start).
   Every adjusted level is computable from data available at that bar.
   (Conventional segment-END anchoring is anticausal in *levels* — a future
   roll rescales the entire past — and would break gate (a) below. Returns
   are identical under either anchoring.)
2. **One execution-timing chokepoint.** Decide at bar-t close → fill at
   t+1 open (traded/limit veto) → PnL open(t+1)→open(t+2). This lives in
   exactly one function in `gp_cta/gp/backtest.py`; nothing else times trades.
3. **Causal features only**: trailing windows over the product's own traded
   bars; daily statistics (σ60, seasonality profile) from strictly-prior
   trading days; EWMA vol shifted one bar inside `x_std`; backtest threshold
   quantiles computed on the signal shifted one bar; limit-lock detection
   uses *running* (causal) session extremes, never the full day's extreme.
4. **As-of edge discipline**: GP touches foreign series only via the edge
   table filtered `asof_date ≤ t − 1 trading day`, terminal resolution frozen
   at month m−1, lags clamped to validated values.
5. **Automated leakage gates (build gates, not optional tests)**:
   (a) *truncation-prefix* — rebuild any artifact with inputs truncated at
   date d → the prefix is bitwise-identical;
   (b) *+1-bar kill-switch* — shift all features forward one bar → |fitness|
   collapses to ≈ 0;
   (c) *future-edge injection* — adding an edge with `asof_date ≥` the
   resolution month leaves cross-terminal arrays byte-identical.
6. **Sanctioned non-causal metadata** (descriptive facts, not signals):
   session-calendar inference, era flags, contract multipliers/ticks, trap
   filters. Dev-universe selection by full-period ADV is a development
   convenience; production universe selection must be PIT.
7. Every NEW feature or terminal must ship with gate (a) coverage.

---

## Environment and conventions

- Python 3.11 venv: `~/venvs/autollm311` (exists; polars 1.40.1, numpy 2.4,
  pandas 3.0, pyarrow, loguru, pytest installed).
- Engine: **Polars** for data work; NumPy for the matrix scan; pandas only at
  library boundaries. No deep learning. numba allowed for hot kernels.
- Style: `ruff format` + `ruff check`; `mypy --strict` on public APIs;
  `pytest -q`; every random path takes `seed: int` (default 42); loguru.
- Data on disk lives **outside the repo** at `~/AutoLLM_data/futures/`
  (create it). Raw inputs: repo folder `Future_minute_data/` (read-only).
- Work directly in `/Users/zhangziyao/Desktop/AutoMiner/` (the main folder),
  not in `.claude/worktrees/`.
- **Layout (revised 2026-07-13, in-place build):** three top-level packages —
  `futures_common/` (shared spine: paths, config + config_hash, memory guard,
  run manifests, named-stream RNG, umbrella CLI), `data_fetcher/` (data band
  L0–L2, local CSVs only, zero network I/O), `gp_cta/` (mining band: lead-lag
  + GP). Shared config files in top-level `configs/`. The old A-share daily
  baseline was removed (recoverable from git history); the old AKshare equity
  fetcher lives untouched in `data_fetcher_akshare/`; the vendored `qlib/`
  clone was deleted (nothing used it).
- Backtester: NumPy-vectorized with numba JIT kernels for the sequential
  parts (position state machine, rolling quantiles); every kernel has a pure
  NumPy/pandas reference twin used in tests and as runtime fallback. No C++.
- Results go under `results/` (git-ignored); every run writes a json with
  config + seed + git hash + peak RSS.

---

## The data (already profiled and verified — trust these facts)

Location: `Future_minute_data/` — 1-minute bars, ~95 products, 6 exchanges.
Product list and folder tree: `Future_minute_data/TICKERS.md` (note: its
description of 8888 as "main contract" is wrong; the verified semantics are
below).

**Two layouts:**

```
Future_minute_data/2005-202506/EXCHANGE/PRODUCT/XXYYMM.csv   ← one file per contract month
Future_minute_data/2026/YYYYMM/YYYYMMDD/*.csv                ← one folder per calendar day
```

**Historical tree (2005-01-04 → 2025-07-04):** 9,412 files, 68 GB. Columns
(12): `exchange,symbol,open,close,high,low,amount,volume,position,bob,eob,type`.
**2026 tree (2026-01-05 → 2026-02-28):** same + `sequence` (13 cols), ~885 MB.

Column facts (all verified against real files):
- `volume` = contracts; `amount` = CNY turnover **including the contract
  multiplier** (multiplier ≈ amount / (volume × price) — use to infer and
  check a per-product multiplier table); `position` = open interest.
- `bob`/`eob` = bar begin/end, ISO-8601 with `+08:00`; eob = bob + 1 min.
- `type` is constant (14) — drop. `sequence` is candle color — drop.
- Each YYMM file holds one real contract with **raw, unadjusted prices**.

Contract-code semantics (verified empirically):
- `XXYYMM` = real contract. **The historical tree contains ONLY these.**
- `XX8888` (2026 tree only) = open-interest-weighted **index across all
  listed months** — fractional prices, not tradeable.
- `XX9999` = raw-stitched dominant with **unadjusted roll gaps** (verified:
  eggs jumped +9.5% on a 2026-02 roll). `XX9998` = sub-dominant.
- 2026 filenames are mixed-case (`rb2605.csv` vs `RB8888.csv`) — normalize
  symbols to upper case on ingest.

Sessions (all commodity products share 10:15–10:30 and 11:30–13:30 breaks):
| Group | Bars/day | Sessions |
|---|---|---|
| No night (CZCE AP/PK/UR…, DCE jd/lh/fb/bb, all GFEX) | 225 | 09:00–10:15, 10:30–11:30, 13:30–15:00 |
| Night → 23:00 (RB, HC, I, M, TA, MA, most) | 345 | + 21:00–23:00 |
| Night → 01:00 (SHFE base metals, INE BC) | 465 | + 21:00–01:00 |
| Night → 02:30 (AU, AG, AO, INE SC/LU/NR) | 555 | + 21:00–02:30 |
| CFFEX index (IF/IH/IC/IM) | 240 | 09:30–11:30, 13:00–15:00 |
| CFFEX bonds (T/TF/TS/TL) | 255 | 09:30–11:30, 13:00–15:15 |

**Traps every band must handle:**
1. **Session calendars are time-varying**: night sessions launched
   product-by-product from 2013-07; all nights suspended 2020-02 → 2020-05;
   some end-times changed (RB night 01:00→23:00 in 2016-05); CFFEX index
   hours changed in 2016 (old files show 09:15 opens). Infer the calendar
   from the data (L1a below); never hard-code today's hours.
2. **Hard data hole**: 2025-07-05 → 2026-01-04 exists in neither tree.
3. **2026 day folders are calendar-keyed**: post-midnight night bars sit in
   the NEXT day's folder; Saturday folders hold only Friday-night tails.
4. **Zero-volume padded bars** in the historical tree (far months up to 84%
   padding); the 2026 tree writes only traded minutes.
5. **Sporadic all-NaN rows** (e.g. `SC2009.csv`: 3,930 literal-"nan" rows).
6. **Volume/OI counting break**: SHFE/DCE/CZCE switched double→single-sided
   counting around 2020-01 (CFFEX always single). Exact date per exchange is
   configurable in `configs/volume_counting.yaml` (default 2020-01-01);
   verify empirically during L1 by checking for a ~2× volume discontinuity.
7. **Daily price limits (涨跌停)** freeze prices; unlock causes synchronized
   jumps; locked bars are untradeable. Detection heuristic below (L2a).
8. **CFFEX 2015-09 → ~2017**: index-futures restrictions collapsed volume
   ~99%; flag era untradeable for IF/IH/IC.
9. **Late listings** (INE SC 2018, BC 2020, LH 2021, GFEX SI 2022 / LC 2023,
   PL/OP/PD/PT/BZ 2026-only) and delisted CZCE codes (WS, WT, GN, ER, RO,
   ME, TC, RI, PM, JR, LR) exist. universe.py owns these dates.

---

## Prior baseline (removed 2026-07-13)

The daily A-share GP baseline that used to live in `gp_cta/*.py` was deleted
after its ideas were ported (immutable `Expr` tree surgery, `_clean` safety
contract, lagged-threshold lookahead discipline, tournament/elitism engine
shape, fitness structure). Recover it from git history (`git log -- gp_cta`)
or see the old AKshare fetcher idioms in `data_fetcher_akshare/`.

---

## Target architecture (in-place, revised 2026-07-13)

```
configs/            defaults.toml, costs.yaml, volume_counting.yaml, limits.yaml,
                    calendar.yaml, universe.yaml, leadlag_priors.yaml, scan.yaml, gp.yaml
futures_common/     paths.py, config.py, memory.py, manifest.py, log.py, rng.py,
                    cli.py (umbrella: subprocess-per-stage runner), tests/
data_fetcher/       readers/{base,historical,daily2026}.py, ingest.py, calendar.py,
                    continuous.py, universe.py, panel/{memmap_io,align,features}.py,
                    cli.py, exceptions.py, tests/
gp_cta/             panel_io.py, ledger.py, leadlag/{scan,validate,edges}.py,
                    gp/{expressions,ops,terminals,backtest,metrics,fitness,
                    warmstart,engine}.py, gates/ (band 3 stub), cli.py, tests/
tests/e2e/          planted-lead-lag full-pipeline regression test
```

Import DAG: `futures_common ← data_fetcher ← (disk artifacts) ← gp_cta`.
Bands never import each other; gp_cta reads panel artifacts only through
`gp_cta/panel_io.py`. Stage CLIs: `python -m data_fetcher {ingest,calendar,
continuous,universe,panel,features,check}` and `python -m gp_cta
{scan,validate,mine,check}`; the umbrella `python -m futures_common run
--slice acceptance [--check]` runs each stage as its own subprocess under a
memory watchdog.

### L0 — `data_fetcher/ingest.py`

Input: both raw trees. Output: one parquet per real contract at
`~/AutoLLM_data/futures/raw/exchange=SHFE/product=RB/RB2001.parquet`.

Schema: `product (cat, e.g. "RB"), contract (str, e.g. "RB2001"), open/high/
low/close (f32), amount (f64), volume/position (i64), bob (datetime
Asia/Shanghai), trading_date (date)`.

Rules:
- **Skip pseudo-contracts entirely**: any file/symbol ending 8888/9999/9998
  is not ingested (they would corrupt dominant-contract selection).
- Drop `type`, `sequence`, `eob`; drop rows with bob == 'nan'; drop
  volume == 0 rows unconditionally (padded bars carry no trade info;
  limit-locked minutes still print volume at the limit price, so they
  survive); uppercase symbols; `symbol` column is split into `product` +
  `contract`.
- **trading_date rule** (precise): bob in [21:00, 24:00) → the next trading
  day strictly after bob's calendar date; bob in [00:00, 03:00) → the first
  trading day ≥ bob's calendar date; otherwise bob's own date. Trading days
  = the set of dates observed with day-session bars (see L1a); this also
  absorbs the 2026 Saturday-tail folders.
- **Idempotency**: each contract's parquet is fully rebuilt from all its
  source rows on re-run (rewrite-per-contract, dedupe key = contract + bob).
  A contract spanning both trees (e.g. RB2605) gets one merged file.
- Loop unit: one product at a time (historical tree); one day folder at a
  time (2026 tree), buffered per contract and flushed per product.
- CLI: `--products RB,I,TA --start 2019-01-01 --end 2021-12-31` (filters on
  trading_date), `--tree historical|2026|both`.

### L1a — `data_fetcher/calendar.py`

Infer, from ingested bars of the **dominant or near-dominant contracts**, a
session table: `product, date_start, date_end, template_id` with a template
table `template_id → list of (start_time, end_time) intervals` (the 6 base
templates above plus discovered variants, e.g. RB-night-to-01:00 pre-2016).
Also emit `trading_days.parquet` (all dates with any day-session bar across
liquid products). Output path: `~/AutoLLM_data/futures/calendar/`.
Consumers: trading_date rule, panel masks, segment boundaries, null strata.

### L1b — `data_fetcher/continuous.py`

Dominant contract per product per trading day: highest day-volume, with
**switch-only-forward** (never back) and **2-consecutive-day confirmation**
(anti-whipsaw). Bootstrap: first listed day = that day's max volume, no
confirmation. Ties: larger open interest, then nearer delivery month.

Roll mechanics: decision uses day t's volumes; the switch happens at the
first bar of trading day t+1. Roll factor = `close_new(t, 15:00 day-session
close) / close_old(t, 15:00)`. Multiplicative chain, stored as a cumulative
`adj_factor (f64)` column — prices stay raw; adjusted price = raw ×
adj_factor, so raw is always recoverable. **Anchoring is segment-START (PIT,
look-ahead safety §1): adj_factor(bar) = ∏ factors of rolls at-or-before the
bar, so adj_factor = 1.0 at the segment start and every adjusted level is
knowable at that bar's time.** Positions held across a roll are charged one
round-trip transaction cost in backtests.

**The 2025 hole**: the adjustment chain restarts at 2026-01-05. Output
carries `segment_id (0 = 2005→2025-07, 1 = 2026→)`; returns and features
never cross segments.

Output: minute-level stitched bars,
`~/AutoLLM_data/futures/continuous/product=RB/year=YYYY.parquet`:
`bob, trading_date, contract, open/high/low/close (f32, raw), adj_factor
(f64), volume, position (i64), roll_flag (bool), segment_id (i8)`.
Plus a roll calendar: `product, trading_date, old_contract, new_contract,
factor`.

### L1c — `data_fetcher/universe.py`

Per product: listing date, delisting date (if any), era flags (CFFEX
2015-09→2017-12 restricted; pre/post volume-counting break), and a liquidity
table (rolling 60-day median daily volume × multiplier × price = ADV proxy).
v1 dev universe = top ~10 by ADV among: RB, HC, I, J, JM, CU, AL, ZN, AU,
AG, TA, MA, PP, M, Y, P, SR, CF, SC, IF.

### L2a — `data_fetcher/panel/align.py`

Per year and per channel, a time-major float32 `np.memmap` of shape
(T_year_minutes × N_products) on the union minute grid (chronological
order), plus:
- `bob_index.parquet` (row → timestamp, trading_date, segment boundaries);
- masks as uint8 memmaps: `m_traded` (bar exists after L0 filters),
  `m_limit` (limit-locked heuristic: ≥ 30 consecutive bars where close ==
  running session high (or low) AND 1-bar return == 0; plus optional
  `configs/limits.yaml` per-product limit rates for exact detection later).
Session-segment boundaries = union of ALL products' break points including
the 10:15–10:30 mini-break. Output: `~/AutoLLM_data/futures/panel/year=YYYY/`.

### L2b — `data_fetcher/panel/features.py`

All causal (trailing data only), float32, one memmap per feature per year.
Pinned v1 feature list (names are the GP terminal names):
- `x_std` — the canonical standardized return used everywhere:
  1-bar log return of adjusted close, winsorized at ±5σ over trailing 60
  trading days, divided by EWMA vol (halflife = 1 trading day of bars),
  divided by the minute-of-day median-|return| profile (trailing 60 trading
  days, per session template).
- `ret_5m, ret_15m, ret_60m, ret_1d` — cumulative log returns.
- `vol_ewma` — the EWMA vol above.
- `z_60, z_240, z_1200` — close z-scores over {60, 240, 1200} bars.
- `oi_chg_1d` — 1-day open-interest change ratio (era-adjusted per
  volume-counting config).
- `vol_ratio` — vol_ewma(60 bars)/vol_ewma(1200 bars) of volume.
- `range_1, range_15, range_60` — (high−low)/close over {1,15,60} bars.

### L3 — `leadlag/scan.py` + `validate.py`

Input: `x_std` panels + masks. For lag ℓ ∈ [1, 60] bars:
- **Segment-blocked GEMMs** (never across any segment boundary): within each
  segment, uncentered cross-products `Z[:-ℓ]ᵀ Z[ℓ:]` (uncentered is
  deliberate — x_std is pre-standardized, means ≈ 0), summed over segments,
  with per-pair joint-valid normalization from 3 Gram products
  (x·x, x²·mask, mask·x²).
- Cells with joint-valid count n[i,j,ℓ] < 20,000 bars are excluded.
- Statistics per ordered pair: detection = sup_ℓ |ρ(ℓ)| (reported);
  **inference runs on the asymmetry statistic** A_ij = Σ_{ℓ=1..60}
  (|ρ_ij(ℓ)| − |ρ_ji(ℓ)|) (absolute values — signed sums confuse direction
  with relationship sign). `lag_bars` = argmax_ℓ |ρ_ij(ℓ)|; `sign` =
  sign(ρ at that lag).
- **Null**: 200 draws; circularly shift the LEADER's series (values + mask
  together) by a uniform whole-trading-day offset in [5, D−5], within strata
  of identical session templates; day blocks keyed by trading_date;
  p = (1 + #{|A_null| ≥ |A_obs|}) / 201.
- **FDR**, two families: (A) economic priors from
  `configs/leadlag_priors.yaml` → Benjamini–Hochberg, q = 0.05;
  (B) all remaining pairs → Benjamini–Yekutieli, q = 0.10.
  Starter prior list (family A): I→RB, I→HC, J→RB, JM→J, RB↔HC, ZC→methanol
  chain, SC→FU, SC→BU, SC→PX? (PX 2026-only: use TA), SC→TA, TA→PF, MA→PP,
  L↔PP↔V, M→RM, Y→P, OI→Y, A→M, CU→ZN, CU→AL, CU↔BC, AU↔AG, IF↔IH↔IC,
  IF→CU, RU↔NR, AG→SI? (no: SI is industrial silicon — skip), LC→SI.
  Both directions of each are registered.
- **Rolling protocol**: full-history scan once, then monthly re-scans on a
  trailing 24-month window. `stability` = share of the last 12 monthly
  re-scans with same sign and q < 0.10. `half_life` = ln 2 / λ from an AR(1)
  fit to the monthly A_ij series. `regime_flags` = strings, e.g.
  "night-only", "pre-2020", "cffex-restricted-era".

### L4 — `leadlag/edges.py`

Append-only parquet:
`asof_date, leader, follower, lag_bars, strength (ρ at lag), sign, a_stat,
p_value, q_value, family, n_eff, stability, half_life, regime_flags,
config_hash`. Consumers only read rows with asof_date ≤ t − 1 trading day.
Re-scans append under a new config_hash; nothing is overwritten. Every
(pair, window) examined — significant or not — is also counted into
`results/trial_ledger.parquet` for band 3.

### GP band — `gp_cta/gp/`

**Operator set** (typed; windows are discrete choices, never free integers):
- Arithmetic: add, sub, mul, div (safe), neg, abs, log1p, sign, min, max,
  gt, lt (→ 0/1), if_then_else.
- Time-series (window W ∈ {5, 15, 30, 60, 120, 240, 480, 1200} bars):
  ts_mean, ts_std, ts_min, ts_max, ts_delta, ts_sum, ts_rank, ts_zscore,
  ema, decay_linear, ts_corr(x, y, W).
- Terminals: the L2b features, constants {−2, −1, −0.5, 0.5, 1, 2}, and
  cross-asset terminals (below).

**Cross-asset terminals** (resolved via the edge table, as-of t − 1 day):
- `LEAD_RET(rank)` — the rank-th leader of this follower (by |a_stat| among
  validated edges); value = leader's cumulative `x_std` over the validated
  `lag_bars` bars ending strictly at t.
- `LEAD_OI_CHG(rank)` — that leader's `oi_chg_1d`.
- `LEAD_STALE(rank)` — minutes since that leader last printed a traded bar.
- `LL_STRENGTH(rank)` — the edge's current `strength`.
GP never searches lags; they are clamped to the validated values.

**Warm start** (Ren–Qin–Li, arXiv:2412.00896): population starts as ONE seed
formula; generation 1 = point mutations only (swap feature / window /
same-arity operator at a node — tree shape frozen); generations ≥ 2 mix 50%
restricted crossover (swap subtrees only at identical tree positions between
same-shape parents), 40% point mutation, 10% copy; tournament size 4;
elitism 1; duplicates (canonical-AST hash) rejected; per-seed population 64,
10 generations, early stop after 3 stagnant generations. Seeds run
sequentially (memory rule 6). Each seed run yields its best formula; the
cross-seed pool then rejects any candidate with |signal corr| > 0.7 against
an already-pooled one (AlphaGen pattern).

**Seed library** (exact ASTs, defaults in parentheses):
- MA cross: `sign(ts_mean(close_adj, Wf=60) − ts_mean(close_adj, Ws=480))`
- Channel breakout: `gt(close_adj, ts_max(ref(close_adj,1), W=240)) −
  lt(close_adj, ts_min(ref(close_adj,1), W=240))`
- RSI-style: `ts_mean(max(ts_delta(close_adj,1),0), W=240) /
  (ts_mean(abs(ts_delta(close_adj,1)), W=240) + ε) − 0.5`
- Vol-scaled momentum: `ts_delta(close_adj, W=240) / (ts_std(x_std, W=240)·√W + ε)`
- Lead-lag (one per validated edge): `sign(LEAD_RET(1)) ·
  gt(LL_STRENGTH(1), 0.02)`

**Backtest** (`gp/backtest.py`):
- Position ∈ {−1, 0, +1} per product from two-sided lagged rolling-quantile
  bands on the signal: long if signal > trailing q0.70, short if < q0.30
  (lookback 1200 bars, thresholds shifted 1 bar); otherwise hold previous
  position; forced flat after max_hold bars (default = 1 trading day of
  bars, config).
- Decided at bar t close, executed at bar t+1 open; PnL marked open-to-open
  on adjusted prices; position persists until flipped/expired (NOT a fixed
  1-bar hold).
- **No entry or exit on limit-locked bars** (m_limit); forced flat across
  the 2025 segment boundary; roll bars charge one round-trip cost.
- Costs from `configs/costs.yaml`: per-product `fee_bps` per side +
  `slippage_ticks` (default 1) × tick size. Starter defaults: commodities
  fee 1.0 bps/side; CFFEX index 0.5 bps/side (note: real close-today CFFEX
  fees are punitive — refine from a broker schedule later); tick sizes
  inferred per product as the minimum positive price increment observed in
  the data (mode-of-diffs), stored in the same yaml.

**Fitness** (`gp/fitness.py`): aggregate bar PnL → daily returns per
product → per-product Sharpe = mean/std × √252 → fitness = equal-weight mean
across products − 0.5 × mean maxDD − 0.05 × mean daily turnover. Validity:
≥ 20 trades per product-year on average AND time-in-market ≥ 2%, else −10.
A negative-Sharpe formula may be sign-flipped once **before** the validity
gate; the flip counts as one extra trial in the ledger.

**Splits**: train 2016-01 → 2022-12; validation 2023-01 → 2024-06; holdout
2024-07 → 2025-07-04 plus 2026-01/02. Holdout is scored exactly once per
mining round.

### Band 3 — `gates/` (later phase)

Deflated Sharpe (trial count = ledger total: every formula tried, every
sign-flip, every lead-lag pair×window examined), BHY FDR, CPCV. Not needed
to start; the ledger makes it possible later.

---

## Build order and acceptance checks

| Step | Done when |
|---|---|
| 1. `data_fetcher/ingest.py` | RB+I+TA 2019–2021 slice ingests; pseudo-contracts skipped; row counts match raw minus filters; peak RSS < 1 GB printed; re-run byte-identical |
| 2. `data_fetcher/calendar.py` | Inferred table shows RB night ending 01:00 before 2016-05, 23:00 after; COVID night gap (2020-02→05) detected; trading_days covers Golden Week closures |
| 3. `data_fetcher/continuous.py` | RB dominant follows the 01/05/10 pattern; zero fake return spikes at rolls on adjusted series; 2026 egg-roll gap absent; segment restart at 2026-01-05 |
| 4. `panel/` | 10-product × 3-year panel builds < 4 GB peak; masks agree with calendar; x_std passes causality test (recompute with truncated data → identical prefix) |
| 5. `leadlag/` | 10×10×60 tensor on 3 years < 60 s; the I↔RB pair validates in one direction (2019–21 outcome: **RB→I at lag 1**, q≈0.015 — the liquid leg leads; the naive I→RB prior was backwards); null p-values uniform on shuffled data; edge table round-trips |
| 6. `gp/` rework | No lookahead: +1-bar shift kills fitness (2019-21 outcome: best train 1.20 → shifted **−2.11** ✓). Warm-start-beats-random is gated at DEV-slice scale (≥ 6 products, mean validation fitness across seeds — the paper's claim is an average over many seeds on a wide cross-section); on the 3-product acceptance slice the 4-seed A/B is regime noise and is reported, not gated (2019-21 draw: random won, ws mean −0.24 vs rnd 0.64) |
| 7. `gates/` | Later phase |

Every step: `pytest -q` green; small-slice peak memory measured and printed;
run json written.

## References

- Warm-start GP: Ren, Qin, Li (2024), arXiv:2412.00896 — the method summary
  above is complete enough to implement.
- Lead-lag math, statistics, pitfalls in depth: `docs/LEADLAG_DESIGN.md`.
- Literature research prompt: `docs/RESEARCH_PROMPT_leadlag.md`.
- Product list & folder details: `Future_minute_data/TICKERS.md`.
- Patterns studied: AlphaGen (KDD'23) — typed AST, cached panel fitness,
  correlation-gated pool. AlphaGen's search is RL; we do NOT use that. Its
  repo's own GP baseline (`gp.py` + vendored gplearn) is the relevant part.
