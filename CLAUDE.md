# `AutoLLM/` — project conventions

> **Source of truth:** `proposal_version1.md` is the spec. This file holds the *durable conventions* for the build session — things that don't belong in the proposal but every phase needs to know. If a future phase needs a convention you can't find here, add it before using it. **Never** hand-carry a convention as session state.

---

## ⚠ ACTIVE WORK (2026-07): futures GP-CTA mining — read `gp_cta/proposal.md` first

The current build effort is **GP-based factor mining on Chinese futures minute
data** (`Future_minute_data/`, 68 GB), *not* the CSI 300 STL pipeline below.
The guide for this work is **`gp_cta/proposal.md`** — read it before writing
any code. Deeper design: `docs/LEADLAG_DESIGN.md`.

**TWO CO-EQUAL TOP PRIORITIES:**

1. **MEMORY SAFETY.** The machine has 16 GB RAM. All code processes data in
   the smallest possible snippets (one contract file / one product-year / one
   day folder at a time), stays under ~4 GB peak per stage (enforced by
   `futures_common.memory.PeakTracker` + the umbrella watchdog), uses Polars
   lazy scans + float32, writes intermediates to disk, and is tested on
   1 product × 1 year with measured peak memory before any larger run. Full
   rules: `gp_cta/proposal.md` § "Memory safety". When in doubt, use a
   smaller batch. Speed never wins over memory.
2. **LOOK-AHEAD SAFETY.** No artifact at time t may use information revealed
   after t: PIT segment-START-anchored roll adjustment, one execution-timing
   chokepoint (decide t close → fill t+1 open), strictly causal features,
   as-of edge discipline, and three automated leakage gates
   (truncation-prefix, +1-bar kill-switch, future-edge injection). Full
   rules: `gp_cta/proposal.md` § "Look-ahead safety".

**Layout (since 2026-07-13):** `futures_common/` (shared spine),
`data_fetcher/` (futures data band, local CSVs only — the old AKshare equity
fetcher is preserved untouched in `data_fetcher_akshare/`), `gp_cta/` (mining
band), top-level `configs/`. The old A-share GP baseline and the vendored
`qlib/` clone were deleted 2026-07-13 (git history has them).

**Build status (2026-07-13):** the full framework is built and
acceptance-verified on the RB/I/TA 2019–2021 slice (proposal build-order
steps 1–6): ingest → inferred calendar → PIT continuous → panel/features →
lead-lag scan/validation/edges → warm-start GP with structured backtest +
full metrics + trial ledger. Band 3 (`gp_cta/gates/`) is a stub by design.
Package READMEs document each band; `python -m futures_common run --slice
acceptance --check` drives the whole chain.

Also: edit files directly in this main folder, not in `.claude/worktrees/`.

---

## What this project is

A four-pipeline auto-mining factory for CSI 300 trading signals. Mining Factory has four independent miners (Pipeline A: STL+MCTS per-asset CTA; B: RL-CTA; C: RL cross-sectional alpha; D: LLM alpha) feeding a shared strategy pool. Estimation Factory has two parallel admission gates (CTA for A∪B; alpha for C∪D) with explicit multiple-testing correction (DSR, BHY haircut, CPCV, monthly CSCV-PBO). v1 builds Pipeline A end-to-end on an M1 16 GB MacBook with **no deep learning**; B/C/D are explicitly v2.

This file's session covers Phases 0 → 3.6 (Pipeline A baseline, multi-horizon, BMA combiner, decay monitor). Phase 4 (`signal_miner/` shared services + alpha-family gate + MTC) and Pipelines B/C/D are separate prompts later.

---

## Folder map (updated 2026-07-13)

```
AutoMiner/
├── CLAUDE.md                       ← this file
├── configs/                        ← shared pipeline configs (defaults.toml, costs.yaml, …)
├── futures_common/                 ← ACTIVE — shared spine (paths, config, memory, manifest, rng, umbrella CLI)
├── data_fetcher/                   ← ACTIVE — futures data band L0–L2 (local CSVs, zero network)
├── gp_cta/                         ← ACTIVE — mining band (proposal.md + leadlag/ + gp/)
├── docs/                           ← LEADLAG_DESIGN.md (lead-lag math)
├── Future_minute_data/             ← read-only raw 1-minute futures CSVs (68 GB, git-ignored)
├── data_fetcher_akshare/           ← dormant — old AKshare A-share fetcher (reference, untouched)
├── STL pipelines/                  ← dormant — CSI 300 Pipeline A
└── NLP and LLM automatic signal Miner/
    └── relative papers/            ← read-only paper PDFs + cloned repos
```

Read-only folders: `Future_minute_data/`, `data_fetcher_akshare/`, and
`NLP and LLM automatic signal Miner/`. Do not edit anything inside them.
(`qlib/` was deleted 2026-07-13 — nothing used it.)

> **Everything below this line documents the DORMANT CSI 300 effort** (kept as
> historical record; its conventions still apply where they don't conflict
> with `gp_cta/proposal.md`).

---

## Conventions

| Knob | Value | Why |
|---|---|---|
| Python | 3.11+ | Polars + AKshare + pandas-3.x compatibility floor |
| Shared venv | `~/venvs/autollm311` | Outside repo (R6); shared across phases |
| Throwaway venv | `~/venvs/qlib_xref311` | Phase 2 cross-validation only; deleted after use |
| Formatter | `ruff format` | Single tool for format + lint |
| Linter | `ruff check` | Same |
| Type checker | `mypy --strict` on public APIs only | Don't `--strict` test files |
| Test runner | `pytest -q` | Default; `--network` marker gates AKshare smoke |
| DataFrame engine | **Polars** for new writes; **pandas** only when interfacing with libraries that demand it (AKshare, sklearn, scipy in places) | 5–20× faster on rolling-window ops |
| Data sources | **Pluggable `Source` interface**, AKshare is the v1 implementation. Future sources (paid tabular data, tushare, Wind dumps) drop in as new `Source` subclasses without touching downstream code. | User plans to buy paid data later; design for it now |
| Qlib at runtime | **Never.** Qlib is a read-only reference for the Alpha158 formula list (and only the formula list). The Phase 2 cross-validation uses a *throwaway* venv that gets deleted after the diff matches. | Heavyweight abstractions, fragile install (decision-log row 1 in proposal §0) |
| Storage format | parquet, partitioned `year=YYYY/symbol=XXX.parquet` | Portable, columnar, plays with everything |
| Ad-hoc query | DuckDB over the parquet store | SQL without spinning up a DB |
| Experiment tracking | MLflow standalone (no Qlib wrapper) | Pinned later in Phase 5 |
| Logger | `loguru` | Cheap dependency; better defaults than stdlib `logging` |
| Random seeds | Every randomness path takes a `seed: int` argument; default `42` for tests | R4 reproducibility |

### Data on disk (outside the repo)

```
~/AutoLLM_data/
├── raw/                 ← AKshare→parquet output, year/symbol partitioned
│   └── year=YYYY/symbol=SH600519.parquet
├── universe/            ← CSI 300 add/drop event log (PIT source of truth)
│   └── csi300/membership_history.parquet  ← from ak.index_stock_hist; columns: stock_code, in_date, out_date
│   └── csi300/snapshots/YYYY-MM-DD.parquet ← derived per-date snapshots (cached, optional)
├── alpha158/            ← cached Alpha158 panel
│   └── csi300/YYYY-MM-DD/symbol=SH600519.parquet
├── labels/              ← triple-barrier label panels, rule-hash keyed
│   └── {label_rule_hash}/{primary_rule_hash}/csi300/H={H}/symbol=SH600519.parquet
├── sample_weights/      ← uniqueness weights, rule-hash keyed
│   └── {rule_hash}/csi300/symbol=SH600519.parquet
├── _qlib_xref/          ← Phase 2 throwaway xref output (kept; venv deleted)
│   └── alpha158_5sym_1y.parquet
└── pool/                ← admitted strategies + sqlite metadata
    ├── strategies.parquet
    └── meta.sqlite
```

**Never** put data inside the repo (R6). The `data_fetcher/tests/fixtures/` parquet (5 symbols × 100 days, baked from a Phase 1 smoke pull) is the **only** exception — it's small, version-controlled, and lets Phases 2+ run network-free.

### Symbol normalisation

| Raw input | Normalised | Source |
|---|---|---|
| `600519` (Kweichow Moutai) | `SH600519` | numeric prefix `6` → SH |
| `000001` (PA Bank) | `SZ000001` | prefix `0` → SZ |
| `300750` (CATL) | `SZ300750` | prefix `3` → SZ ChiNext |
| `688981` (SMIC) | `SH688981` | prefix `68` → SH STAR |
| `SH600519` / `SZ000001` | passthrough | already normalised |
| `600519.SH` (Wind/Tushare) | `SH600519` | reverse the dot suffix |

The full table + edge cases land in `data_fetcher/symbols.py`.

---

## DSL syntax

Two distinct DSLs in this project — **don't mix them**.

### Alpha158 DSL (Phase 2)

Operator vocabulary (port from Qlib's `qlib/contrib/data/loader.py:72-310` — see "Research notes" below):

```
Ref(x, d)              Mean(x, d)             Std(x, d)
Max(x, d)              Min(x, d)              IdxMax(x, d)
IdxMin(x, d)           Slope(x, d)            Rsquare(x, d)
Resi(x, d)             Quantile(x, d, p)      Rank(x, d)        # time-series rank
Corr(x, y, d)          Log(x)                 Abs(x)
Greater(x, y)          Less(x, y)             If(cond, x, y)
Sum(x, d)              Add  Sub  Mul  Div
Delta(x, d) := x - Ref(x, d)
```

All operators are **per-asset time-series** — Alpha158 has no cross-sectional ops (the subagent confirmed; see "Research notes"). Cross-sectional operators (`CSRank`, `CSZScore`, `CSMean`) come later for Pipelines C/D.

Variable convention: `$close`, `$open`, `$high`, `$low`, `$volume`, `$vwap` (with the `$` prefix, matching Qlib's syntax) for raw OHLCV; computed features get unprefixed names like `alpha158_RSI_14` or `ret_5d`.

### STL DSL (Phase 3)

Per proposal §11.1:

```
Formula  := Atomic | Not Formula | Formula And Formula | Formula Or Formula
          | G[a,b] Formula | F[a,b] Formula | Formula U[a,b] Formula

Atomic   := Feature CMP Threshold
CMP      := > | <
Feature  := RawFeature | DerivedFeature
RawFeature     := $close | $open | $high | $low | $volume
                | ret_1d | ret_5d | vol_20d
DerivedFeature := <any of the 158 named features in alpha158/alpha158.py>
Threshold:= float (initialised at the rolling-window quantile of the feature)
[a,b]    := temporal interval, integer days, 0 ≤ a ≤ b ≤ 60
```

**Bounded:** `max_depth ≤ 4`, `0 ≤ a ≤ b ≤ 60`, `max_formula_size = 32` AST nodes.

**Operators have signed-distance robustness semantics** (proposal §11.2). All implemented as Polars rolling reductions — **no per-row Python loops**.

**Pre-processing:** Winsorize raw features at 1st/99th percentile per asset per rolling 252-day window before robustness — otherwise `min`/`max` over a 60-day temporal operator gets dominated by a single outlier print.

---

## Subagent dispatch policy (R3)

Use a fresh subagent for:
- **Reading any paper PDF** in `relative papers/`. Subagent extracts ≤300-500 words. Never load full PDFs into main context.
- **Reading any large existing module** (Qlib's loader.py, AlphaGen's expression.py). Subagent summarises interface + algorithm + code skeleton.
- **Long-running cross-validations** (Alpha158 158-column diff vs Qlib reference; AKshare smoke pulls).

**Never** delegate the *implementation* of the four modules to subagents. The implementation is the durable artifact and lives in the main thread where the user can see every edit.

When dispatching, use `subagent_type="Explore"` for read-only file/code lookups, `general-purpose` for multi-step research, `Plan` only when an implementation plan is the goal.

---

## Stop conditions (R9)

Stop and report at:
- End of Phase 0 (now — after this CLAUDE.md is written).
- After each phase's plan (R1) before implementing.
- After each phase's tests pass (or after R4's twice-failed condition fires).
- Any time R7 (ambiguity) fires.

---

## Open questions (R7 flags — resolve before they harden)

Resolved at end of Phase 0:

1. ✅ **Python 3.11 via Homebrew** — confirmed. `python3.11 -m venv ~/venvs/autollm311` is the canonical setup.
2. ✅ **AKshare for v1, source layer pluggable.** No tushare fallback wired in v1; the `Source` interface keeps the door open for paid data drops later.
3. ⏸ **MILP solver** — defer to Phase 3. Default to CBC (free, via PuLP); explained to user.
4. ⚠ **CSI 300 PIT — partial only in v1** (the Phase-0 subagent was wrong about `index_stock_hist`; corrected during Phase 1). v1 uses `ak.index_stock_cons` which returns today's 300 members + their `in_date` only — captures additions, NOT removals. Survivorship bias for pre-2022 backtests is real and documented; Phase 1.5 swap-in path is via Tushare or paid data. See "Research notes — CSI 300 membership" below for the full story.
5. ✅ **Qlib runtime** — banned. Read-only reference for Alpha158 formulas only. Phase 2 cross-validation runs in a throwaway venv that gets deleted after the diff passes.

Outstanding (resolve at the relevant phase boundary):

- **Per-source rate-limit defaults.** AKshare's `stock_zh_a_hist` throttles past ~5–10 req/s. v1 default: 0.5s base + exponential backoff to 30s, configurable per-`Source`. Revisit if Phase 1 smoke pull is too slow.
- **Universe parquet retention.** Per-date PIT snapshots are derivable from the event log + a date — should we cache them eagerly (one .parquet per trading day per universe = ~2k files for 8 years) or compute lazily on read? Default: lazy — compute on `Fetcher.universe(name, on_date)`, cache only if Phase 1 profiling shows it's a hot path.

---

## Research notes (filled by subagent dispatches; never re-read in main context)

### Alpha158 — formulas (subagent, Phase 0)

**Source location:** `qlib/qlib/contrib/data/loader.py`, class `Alpha158DL`, method `get_feature_config()`, **lines 72–310**. (The build prompt referenced `handler.py`; that file just imports from `loader.py`. Use `loader.py` as the truth.)

**~158 features in 10 categories, all per-asset time-series, no cross-sectional ops:**

| Category | Count | Windows | Examples |
|---|---|---|---|
| K-line shape | 9 | 0 | `(close-open)/open`, `(high-low)/open`, body/wick ratios |
| Price/VWAP normalised | ~20 | [0..4] | lagged `OHLCV/close` |
| Volume normalised | ~5 | [0..4] | lagged `vol/vol_norm` |
| Rolling mean/std | 10 | {5,10,20,30,60} | `MA`, `STD`, `VMA`, `VSTD` |
| Rolling extrema/rank | 15 | {5,10,20,30,60} | `MAX`, `MIN`, `RANK`, `RSV`, `IMAX`, `IMIN`, `IMXD` |
| Momentum / returns | 5 | {5,10,20,30,60} | `ROC`, `CNTP`, `CNTN`, `CNTD` |
| Regression | 15 | {5,10,20,30,60} | `BETA` (slope), `RSQR`, `RESI` |
| Quantiles | 10 | {5,10,20,30,60} | `QTLU` (80th), `QTLD` (20th) |
| Gain/loss asymmetry | 15 | {5,10,20,30,60} | `SUMP`, `SUMN`, `SUMD`, `VSUMP`, `VSUMN`, `VSUMD` |
| Correlation & vol | 10 | {5,10,20,30,60} | `CORR(close, log(vol))`, `CORD(ret, vol_chg)`, `WVMA` |

**Operator vocabulary** (full list in DSL syntax section above).

**Polars idiom for the bulk** (rolling-window features):
```python
# MA20: pl.col("close").rolling_mean(20).truediv(pl.col("close"))
# STD20: pl.col("close").rolling_std(20).truediv(pl.col("close"))
# CORR20: pl.rolling_corr(pl.col("close"), pl.col("volume").log1p(), window_size=20)
```

**The handful of features that are NOT pure rolling reductions** (need attention in Phase 2):
- `BETA(close, d)` = OLS slope of close vs `[0..d-1]` — Polars has no built-in; use `rolling_apply` or compute closed-form via `Sum(x*t) - Sum(x)*Sum(t)/d` over `Sum(t²) - Sum(t)²/d`.
- `RESI(close, d)` = `close - (slope * (d-1) + intercept)` — same.
- `IMXD = IdxMax - IdxMin` — composite; trivial.

These three are the most likely places the cross-validation diff (≤ 1e-6 on 5 sym × 252 days) will fail; add hand-test fixtures for them first.

### OHLCV column conventions (settled in Phase 1)

- **`volume`** — number of **shares** (not 手/lots). AKshare returns 成交量 in 手 (1 lot = 100 shares); `_normalise_ohlcv` multiplies by 100 on ingest. Unit test `test_pull_ohlcv_happy_path` pins this.
- **`turnover`** — yuan, exactly as AKshare reports.
- **`vwap`** — **TYP proxy** = (high + low + close) / 3, **not** turnover/volume.
  - Why: when `adjust ∈ {qfq, hfq}`, AKshare adjusts OHLC but leaves turnover/volume raw. `turnover/volume` then yields the *unadjusted* VWAP, which doesn't sit inside the adjusted [low, high] range — incompatible unit-systems. TYP is in the same unit as the price columns by construction, well-known in TA, and preserves the cross-day shape Alpha158 cares about.
  - Trade-off: TYP weights H, L, C equally regardless of intraday volume distribution. For the v1 use case (Pipeline A's STL miner consuming `$vwap` as one feature among 158) this is fine. If a future use case needs true volume-weighted VWAP, we'd pull `adjust=""` separately and apply the per-day adjustment factor `qfq_close/raw_close` — costs 2× AKshare calls per symbol.

### CSI 300 membership — AKshare API (Phase 0 subagent **was wrong**; corrected Phase 1)

**Phase-0 retraction.** A subagent claimed `ak.index_stock_hist(symbol="sh000300")` returns the full add/drop event log for CSI 300 going back to 2005. **That function does not exist** in AKshare 1.18.60. I empirically verified before wiring it. Recording the corrected reality here so no future phase makes the same mistake.

**Functions that *do* exist for CSI 300 membership** (all return *current* members only — none accept a date arg):

| Function | Returns | Use |
|---|---|---|
| `ak.index_stock_cons(symbol="000300")` | 300 rows: `品种代码 (stock_code), 品种名称 (name), 纳入日期 (in_date)` | **v1 source of truth** for membership |
| `ak.index_stock_cons_csindex(symbol="000300")` | 300 rows from csindex.com.cn (English/Chinese mix) | sanity-check only |
| `ak.index_stock_cons_sina(symbol="000300")` | 300 rows from Sina | sanity-check only |
| `ak.index_stock_cons_weight_csindex(symbol="000300")` | 300 rows + current weights | for weight queries (not membership) |

**v1 universe semantics — known limitation, accepted (Owen 2026-05-03).**

`Fetcher.universe("csi300", on_date)` returns today's 300 members **filtered to those whose `in_date ≤ on_date`**. This captures index *additions* correctly: a stock that joined CSI 300 in 2020 will not appear in a 2018 universe query. But it cannot capture *removals* — stocks that were in CSI 300 in 2018 but have since been kicked out are silently absent.

Empirical universe sizes from this approach:

| `on_date` | Universe size returned | Stocks silently missing (true ~300 − returned) |
|---|---|---|
| 2018-01-01 | ~130 | ~170 |
| 2020-01-01 | ~167 | ~133 |
| 2024-01-01 | ~290 | ~10 |

**This is severe survivorship bias for early years.** Backtests pre-2022 should be interpreted with that caveat. Downstream code MUST log a one-line warning the first time `Fetcher.universe("csi300", on_date < today − 1y)` is called.

**Mitigation path (Phase 1.5 / when paid data lands):**
- Replace `AKshareSource.list_universe_event_log` with a Source that exposes proper add/drop history (Tushare's `index_member` is one option; paid-tabular-data drop is another).
- The `Source.list_universe_event_log` return shape (`stock_code, in_date, out_date`) already accommodates a real event log — we just always return `out_date = NULL` for now.
- The downstream filter logic `(in_date ≤ D) AND (out_date IS NULL OR out_date > D)` already handles both cases; no Fetcher rewrite needed.

**Code shape (what we actually wrap):**

```python
import akshare as ak
df = ak.index_stock_cons(symbol="000300")
# df columns: 品种代码, 品种名称, 纳入日期 — 300 rows
# We translate to: stock_code (str, 6-digit), in_date (date), out_date (always None in v1)
```
