# `data_fetcher/` — AKshare → parquet pipeline

Shared service for the four mining pipelines. Pulls A-share OHLCV from AKshare, normalises into a partitioned parquet store on disk, and exposes a typed Python API for downstream consumers.

**Status:** v1 complete (Phase 1). 92/92 offline tests + 11/11 network smoke tests pass.

See `../proposal_version1.md` §8 and `../CLAUDE.md` for design notes; `../CLAUDE_CODE_BUILD_PROMPT.md` Phase 1 was the build spec.

---

## Files in this folder

The package is layered: each layer owns one concern and depends only on the layers below it (`exceptions` / `symbols` → `storage` / `trade_calendar` → `source` / `akshare_source` → `universe` → `fetcher` → `adapter` / `cli`).

| File | What it does |
|---|---|
| `__init__.py` | Public API surface. Re-exports `Fetcher`, `AKshareParquetAdapter`, `DataAdapter`, `Source`, `AKshareSource`, `normalise` / `to_akshare_bare` / `to_akshare_with_market`, plus the `exceptions` and `trade_calendar` modules. Nothing not listed here is part of the contract. |
| `__main__.py` | Two-line shim so `python -m data_fetcher ...` resolves to `cli.main` and exits with its return code. |
| `cli.py` | `argparse`-based CLI for the four user-facing commands — `pull` (backfill OHLCV), `panel` (print a slice for smoke / debug), `universe` (PIT membership for a date), `calendar` (first/last trading dates). Each handler just builds a `Fetcher` and dispatches; no logic of its own. |
| `fetcher.py` | `Fetcher` — the top-level orchestrator. Composes `Source` + `ParquetStore` + `UniverseCache` + trade-calendar cache and exposes `pull / panel / universe / calendar`. Translates user inputs (universe name vs. explicit symbol list, str/`Timestamp` dates, `"today"`) into Source calls and partition reads, paces per-symbol pulls with `sleep_s`, and surfaces a `{pulled, skipped, failed}` tally. Intentionally a thin façade. |
| `adapter.py` | The typed `DataAdapter` Protocol every downstream pipeline programs against (`panel / calendar / universe / windows / context_bundle` — proposal §8.3), plus `AKshareParquetAdapter`, the v1 implementation that forwards to `Fetcher` and reads only from cached parquet (no network at adapter level). `windows(asset, start, end, length)` yields rolling `(length, n_fields)` numpy arrays for STL/CTA window encoders; `context_bundle(on_date)` is a stub for Pipeline D's LLM Idea Agent. Pipelines depend on `DataAdapter`, never `Fetcher` directly, so a v1.5 paid-data adapter drops in unchanged. |
| `source.py` | The `Source` Protocol — the contract any upstream wrapper must satisfy: `pull_ohlcv / list_universe_event_log / trade_calendar`, return Polars frames in the canonical schemas, raise typed `data_fetcher.exceptions.*` instead of leaking `requests.HTTPError` etc. Adding a new source (Tushare, paid feed, …) is ~30 lines against this Protocol. |
| `akshare_source.py` | v1's `Source` implementation. Wraps three AKshare endpoints: `stock_zh_a_hist` (daily OHLCV), `index_stock_cons` (today's index roster + each member's `in_date`), `tool_trade_date_hist_sina` (calendar). Owns the Chinese→English column rename (the single place upstream drift surfaces), bare-vs-prefixed code translation, VWAP via TYP-proxy `(H+L+C)/3` (rationale in CLAUDE.md — `turnover/volume` mixes adjusted prices with raw volume under qfq/hfq), volume unit fix (AKshare's 手 × 100 → shares), exponential-backoff retry on `requests` connection errors, and surfacing of typed exceptions. Synthesises `out_date = NULL` for every membership row since AKshare exposes no add/drop event log — that's where the survivorship caveat enters. |
| `storage.py` | `ParquetStore` — the on-disk layer. Defines `OHLCV_SCHEMA` (the canonical column set + dtypes; the same schema is the contract for `Source.pull_ohlcv`). Layout: `root/raw/source={src}/adjust={adj}/year={Y}/symbol={SYM}.parquet`, so multiple sources and adjust modes coexist without collision. `write_ohlcv` is idempotent — merges with any existing partition, dedupes on `(symbol, date)`, and reports `{written, skipped}`. `read_ohlcv(symbols, start, end)` is cheap because it touches only the year × symbol partitions that actually intersect the request. |
| `universe.py` | `UniverseCache` — caches a `Source`'s membership event log under `root/universe/source={src}/{name}/membership_history.parquet`, refreshes weekly. Exposes `members_on(name, on_date)` with the PIT filter `in_date <= d AND (out_date IS NULL OR out_date > d)` (a no-op for v1 where `out_date` is always NULL, correct for v1.5+ Sources with real removal events) and `ever_members(name)` (the union, used by `Fetcher.pull` to decide what to backfill). Logs a one-time survivorship-bias warning per universe when `on_date` is more than a year in the past. |
| `trade_calendar.py` | Pure functions over a `pd.DatetimeIndex`: `save_cache` / `load_cached` (parquet round-trip with `max_age_days`-based staleness), plus stdlib-style queries `is_trading_day / next_trading_day / previous_trading_day / trading_days_between`. The actual calendar fetch is the `Source`'s job; this module only persists and queries. Named `trade_calendar` so it doesn't shadow stdlib `calendar`. |
| `symbols.py` | Symbol-normalisation helpers. Internal canonical form is `SH600519` / `SZ000001` (uppercase prefix, 6 digits). `normalise()` accepts bare digits, prefixed (any case), Wind dot-suffix (`600519.SH`), Bloomberg dot-suffix (`600519.XSHG` / `.XSHE`); rejects anything else with `InvalidSymbolError`. `to_akshare_bare()` strips the prefix for `stock_zh_a_hist`; `to_akshare_with_market()` lowercases for `index_stock_hist` (`sh000300`). Prefix rule is leading-digit-based: `6/9 → SH`, `0/2/3 → SZ`. Every external entry point passes through `normalise()` exactly once on entry — internal code never sees raw forms. |
| `exceptions.py` | Typed exception hierarchy rooted at `DataFetcherError`: `InvalidSymbolError`, `SymbolNotFound`, `EmptyResponseError`, `RateLimitError`, `FetchTimeoutError`, `StorageError`, `UniverseUnavailableError`. Sources raise these instead of leaking `requests.HTTPError` / `pandas.errors.EmptyDataError` / `KeyError`, so `Fetcher.pull` can decide retry/skip/abort policy without inspecting upstream tracebacks. |

### `tests/`

| File | What it covers |
|---|---|
| `conftest.py` | Shared pytest fixtures and the `--network` flag wiring (offline by default; `--network` enables the real-AKshare smoke tests). |
| `fixtures/ohlcv_5sym_2018_2024.parquet` | 5-symbol × 7-year baked OHLCV (232 KB, 7775 rows). What Phases 2–3 read offline. |
| `test_symbols.py` | `normalise()` round-trips across every accepted form; rejection of bad inputs. |
| `test_calendar.py` | `is_trading_day` / `next_trading_day` / `previous_trading_day` / `trading_days_between` + cache freshness logic. |
| `test_storage.py` | `ParquetStore` write idempotency, `(symbol, date)` dedupe, `read_ohlcv` partition pruning, schema validation. |
| `test_universe.py` | `UniverseCache` PIT filter, cache staleness, survivorship warning fires exactly once per universe. |
| `test_akshare_source.py` | Column rename, VWAP formula, volume scaling, retry policy — mostly mocked; `--network` switches to real AKshare. |
| `test_fetcher.py` | End-to-end pull → panel idempotency, `universe=` vs. `symbols=` mode switching, error tallying. |
| `test_adapter.py` | `AKshareParquetAdapter.windows()` shape + chronology, `context_bundle` v1 stub. |

---

## Public API

```python
from data_fetcher import Fetcher, AKshareParquetAdapter

# Backfill OHLCV for the explicit symbol list, 2018→2024.
f = Fetcher()  # defaults: AKshareSource, ~/AutoLLM_data, adjust="qfq"
f.pull(symbols=["SH600519", "SZ000001"], start="2018-01-01", end="today")
# → {'pulled': N_new_rows, 'skipped': N_existing, 'failed': N_errors}

# Or by index name (PIT-aware, see survivorship caveat below):
f.pull(universe="csi300", start="2018-01-01", end="today")

# Read a panel slice as a Polars LazyFrame.
lf = f.panel(
    universe=["SH600519", "SZ000001"],
    fields=["open", "close", "vwap"],
    start="2024-01-02",
    end="2024-01-31",
)
df = lf.collect()  # columns: date, symbol, open, close, vwap

# PIT membership snapshot.
members = f.universe("csi300", on_date="2020-06-30")

# Trade calendar (cached on disk, refreshes weekly).
cal = f.calendar()  # → pd.DatetimeIndex
```

For pipelines that program against the typed `DataAdapter` protocol (proposal §8.3):

```python
from data_fetcher import AKshareParquetAdapter, Fetcher
adapter = AKshareParquetAdapter(Fetcher())
# adapter.panel / .calendar / .universe / .windows / .context_bundle
```

---

## CLI

```bash
# Backfill (idempotent — re-runs report 'skipped').
python -m data_fetcher pull --symbols SH600519,SZ000001 --start 2018-01-01 --end today
python -m data_fetcher pull --universe csi300 --start 2018-01-01 --end today

# Read panel slice (smoke).
python -m data_fetcher panel --symbols SH600519 --fields close,vwap --start 2024-01-02 --end 2024-01-31

# PIT membership.
python -m data_fetcher universe csi300 --on-date 2020-06-30

# Trade calendar summary.
python -m data_fetcher calendar
```

---

## Storage layout

```
~/AutoLLM_data/
├── raw/source=akshare/adjust=qfq/year=YYYY/symbol=SH600519.parquet
├── universe/source=akshare/csi300/membership_history.parquet
└── calendar/source=akshare/trade_dates.parquet
```

`source=…/adjust=…` partitioning keeps multiple sources (future paid data) and adjustment modes (qfq vs hfq) on disk side-by-side without collisions.

---

## OHLCV column conventions

| Column | Type | Unit / semantics |
|---|---|---|
| `date` | date32 | Trading day |
| `symbol` | utf8 | Canonical SH/SZ + 6-digit form |
| `open`, `high`, `low`, `close` | f64 | Yuan; forward-adjusted under `adjust="qfq"` (default) |
| `volume` | f64 | **Shares** (AKshare's 手 × 100; conversion baked into `_normalise_ohlcv`) |
| `turnover` | f64 | Yuan, raw |
| `vwap` | f64 | **TYP proxy**: `(high + low + close) / 3`. See CLAUDE.md "OHLCV column conventions" for why |
| `return_pct` | f64 | AKshare's 涨跌幅 in % (NOT a fraction) |
| `turnover_rate` | f64 | AKshare's 换手率 in % |

---

## v1 limitation: CSI 300 PIT is one-sided

`Fetcher.universe("csi300", on_date)` filters today's roster by `in_date <= on_date`. This captures additions correctly (a stock that joined CSI 300 in 2020 won't appear in a 2018 universe) but cannot capture removals (stocks that left between then and now are silently absent). For dates more than a year in the past this triggers a one-time survivorship-bias warning. Backtests pre-2022 should be interpreted with the caveat. Mitigation path (paid data or Tushare for membership only) is documented in CLAUDE.md.

---

## Tests

```bash
# Offline (default; <1 s).
~/venvs/autollm311/bin/python -m pytest data_fetcher/tests/ -q

# With network smoke (hits real AKshare; ~10 s).
~/venvs/autollm311/bin/python -m pytest data_fetcher/tests/ -q --network
```

The 5-symbol × 7-year baked fixture at `tests/fixtures/ohlcv_5sym_2018_2024.parquet` (232 KB, 7775 rows) is what Phases 2–3 read offline.
