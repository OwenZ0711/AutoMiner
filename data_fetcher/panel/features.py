"""L2b: causal per-product features → one (T, N) f32 memmap per feature.

CAUSALITY CONTRACT (look-ahead safety §3) — every feature at bar t uses only:
- the product's own TRADED bars at or before t (windows count traded bars);
- daily statistics (σ60, seasonality profile) from trading days STRICTLY
  before t's day;
- the EWMA vol inside x_std shifted one bar.
Windows never cross the 2025 data hole (seg_macro): the 1-bar return is NaN
at the boundary and NaN poisons any window that spans it.

Feature definitions (r_t = log close_adj diff over consecutive traded bars,
B_d = the product's active-template bars/day):

x_std     clip(r, ±5·σ60_d) / vol_ewma_{t-1} / m̂_d(τ_t); σ60_d = std of r over
          the 60 trading days before d; m̂_d(τ) = median |r| at rel-minute τ
          over the previous 60 same-template days, normalized to mean 1,
          floored at 0.25.
ret_{5m,15m,60m}  rolling sum of r over {5,15,60} traded bars (all-finite).
ret_1d    rolling sum of r over B_d traded bars.
vol_ewma  sqrt(EWMA(r², halflife = B_d bars)) — unshifted.
z_{60,240,1200}   (close_adj − mean_W) / std_W, trailing inclusive.
oi_chg_1d position_t / position_{t−B_d} − 1; NaN within ±B_d bars of the
          exchange's volume-counting break date (trap 6) or across segments.
vol_ratio EWMA(volume, hl 60 bars) / EWMA(volume, hl 1200 bars).
range_{1,15,60}   (max(high_adj, W) − min(low_adj, W)) / close_adj.

Cross-year warmup: building year Y prepends the trailing WARMUP_DAYS of prior
years' bars so per-year builds match a single full pass (EWMA memory decays
below f32 resolution within the warmup).
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from data_fetcher.calendar import ANCHOR_MIN
from data_fetcher.panel import memmap_io
from futures_common.numba_compat import njit_kernel
from futures_common.paths import FuturesPaths

FEATURES = (
    "x_std",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "ret_1d",
    "vol_ewma",
    "z_60",
    "z_240",
    "z_1200",
    "oi_chg_1d",
    "vol_ratio",
    "range_1",
    "range_15",
    "range_60",
)

WARMUP_DAYS = 130  # trading days of prior-year context (≥ 26 halflives of hl=1200)
SIGMA_DAYS = 60
SEASONAL_DAYS = 60
SEASONAL_FLOOR = 0.25
WINSOR_SIGMAS = 5.0
_MIN_PER_DAY = 1440


@dataclass(slots=True)
class FeatureReport:
    years: list[int]
    products: list[str]
    features: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"years": self.years, "products": self.products, "features": self.features}


# ------------------------------------------------------------ rolling helpers


def rolling_sum_strict(x: np.ndarray, w: int) -> np.ndarray:
    """Sum of the last w values; NaN unless ALL w values are finite."""
    out = np.full(x.size, np.nan)
    if x.size < w:
        return out
    xf = np.where(np.isfinite(x), x, 0.0)
    bad = (~np.isfinite(x)).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(xf)])
    cb = np.concatenate([[0], np.cumsum(bad)])
    sums = cs[w:] - cs[:-w]
    nbad = cb[w:] - cb[:-w]
    valid = nbad == 0
    out[w - 1 :] = np.where(valid, sums, np.nan)
    return out


def rolling_mean_std(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing inclusive mean/std (ddof=0), NaN until w values, NaN-strict."""
    mean = np.full(x.size, np.nan)
    std = np.full(x.size, np.nan)
    if x.size < w:
        return mean, std
    xf = np.where(np.isfinite(x), x, 0.0)
    bad = (~np.isfinite(x)).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(xf)])
    cs2 = np.concatenate([[0.0], np.cumsum(xf * xf)])
    cb = np.concatenate([[0], np.cumsum(bad)])
    s = cs[w:] - cs[:-w]
    s2 = cs2[w:] - cs2[:-w]
    nbad = cb[w:] - cb[:-w]
    m = s / w
    v = np.maximum(s2 / w - m * m, 0.0)
    ok = nbad == 0
    mean[w - 1 :] = np.where(ok, m, np.nan)
    std[w - 1 :] = np.where(ok, np.sqrt(v), np.nan)
    return mean, std


def rolling_extreme(x: np.ndarray, w: int, mode: str) -> np.ndarray:
    """Trailing inclusive max/min over w values (NaN-strict)."""
    out = np.full(x.size, np.nan)
    if x.size < w:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, w)
    agg = view.max(axis=1) if mode == "max" else view.min(axis=1)
    out[w - 1 :] = agg  # NaN propagates through max/min naturally
    return out


@njit_kernel(cache=True)
def _ewma_scan(x: np.ndarray, alphas: np.ndarray) -> np.ndarray:  # pragma: no cover
    out = np.empty(x.size)
    state = np.nan
    for t in range(x.size):
        v = x[t]
        if np.isnan(v):
            out[t] = state
            continue
        if np.isnan(state):
            state = v
        else:
            a = alphas[t]
            state = a * v + (1.0 - a) * state
        out[t] = state
    return out


def ewma(x: np.ndarray, halflife_bars: np.ndarray) -> np.ndarray:
    """EWMA with per-bar halflife (bars/day varies by template run)."""
    alphas = 1.0 - np.exp(-np.log(2.0) / np.maximum(halflife_bars, 1.0))
    out: np.ndarray = np.asarray(_ewma_scan(np.ascontiguousarray(x, dtype=np.float64), alphas))
    return out


# --------------------------------------------------------------- core compute


def _seasonal_profile(
    abs_r: np.ndarray,
    rel: np.ndarray,
    day_index: np.ndarray,
    template_ids: np.ndarray,
) -> np.ndarray:
    """m̂ per bar: trailing-SEASONAL_DAYS same-template median |r| at the bar's
    rel-minute, normalized to mean 1 over covered minutes, floored.

    Strictly causal: day d's profile uses only days BEFORE d.
    """
    out = np.full(abs_r.size, np.nan)
    n_days = int(day_index.max()) + 1 if day_index.size else 0
    buffers: dict[int, np.ndarray] = {}
    fill_count: dict[int, int] = {}
    day_starts = np.flatnonzero(np.diff(np.concatenate([[-1], day_index])) > 0)

    for k, s in enumerate(day_starts):
        e = day_starts[k + 1] if k + 1 < day_starts.size else abs_r.size
        tid = int(template_ids[s])
        buf = buffers.get(tid)
        if buf is not None and fill_count[tid] >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN minutes
                med = np.nanmedian(buf, axis=0)
            covered = np.isfinite(med)
            if covered.any():
                norm = np.nanmean(med[covered])
                if norm > 0:
                    prof = np.maximum(med / norm, SEASONAL_FLOOR)
                    out[s:e] = prof[rel[s:e]]
        # append THIS day into the buffer (after computing the profile → causal)
        if buf is None:
            buf = np.full((SEASONAL_DAYS, _MIN_PER_DAY * 2), np.nan)
            buffers[tid] = buf
            fill_count[tid] = 0
        row = np.full(_MIN_PER_DAY * 2, np.nan)
        row[rel[s:e]] = abs_r[s:e]
        buf[fill_count[tid] % SEASONAL_DAYS] = row
        fill_count[tid] += 1
    del n_days
    return out


def _daily_sigma(r: np.ndarray, day_index: np.ndarray) -> np.ndarray:
    """σ60_d per bar: std of r over the SIGMA_DAYS trading days strictly before d."""
    out = np.full(r.size, np.nan)
    day_starts = np.flatnonzero(np.diff(np.concatenate([[-1], day_index])) > 0)
    sums: list[tuple[float, float, int]] = []  # per past day: (Σr, Σr², n)
    for k, s in enumerate(day_starts):
        e = day_starts[k + 1] if k + 1 < day_starts.size else r.size
        window = sums[-SIGMA_DAYS:]
        n = sum(w[2] for w in window)
        if n >= 100:
            tot = sum(w[0] for w in window)
            tot2 = sum(w[1] for w in window)
            var = max(tot2 / n - (tot / n) ** 2, 0.0)
            out[s:e] = np.sqrt(var)
        rd = r[s:e]
        finite = np.isfinite(rd)
        sums.append((float(rd[finite].sum()), float((rd[finite] ** 2).sum()), int(finite.sum())))
    return out


def compute_product_features(
    bars: pl.DataFrame,
    *,
    break_date: dt.date | None,
) -> dict[str, np.ndarray]:
    """All features on one product's compressed traded-bar series (f64)."""
    close = bars["close_adj"].to_numpy().astype(np.float64)
    high = bars["high_adj"].to_numpy().astype(np.float64)
    low = bars["low_adj"].to_numpy().astype(np.float64)
    volume = bars["volume"].to_numpy().astype(np.float64)
    position = bars["position"].to_numpy().astype(np.float64)
    seg = bars["seg_macro"].to_numpy()
    rel = bars["rel_min"].to_numpy().astype(np.int64)
    td = bars["trading_date"].cast(pl.Int32).to_numpy()
    tmpl = bars["template_id"].to_numpy().astype(np.int64)

    n = close.size
    day_index = np.cumsum(np.diff(np.concatenate([[td[0] - 1], td])) > 0) - 1

    r = np.full(n, np.nan)
    if n > 1:
        same_seg = seg[1:] == seg[:-1]
        r[1:] = np.where(same_seg, np.log(close[1:]) - np.log(close[:-1]), np.nan)

    # bars/day of the active template, per bar (fallback: median observed)
    bars_per_day = np.bincount(day_index)
    b_by_day = bars_per_day[day_index].astype(np.float64)
    b_med = float(np.median(bars_per_day)) if bars_per_day.size else 345.0
    b_day = np.where(b_by_day > 30, b_by_day, b_med)

    vol_ewma_sq = ewma(np.where(np.isfinite(r), r * r, np.nan), b_day)
    vol_ewma = np.sqrt(vol_ewma_sq)
    vol_ewma_lag = np.concatenate([[np.nan], vol_ewma[:-1]])

    sigma_d = _daily_sigma(r, day_index)
    seasonal = _seasonal_profile(np.abs(r), rel, day_index, tmpl)

    r_wins = np.clip(r, -WINSOR_SIGMAS * sigma_d, WINSOR_SIGMAS * sigma_d)
    x_std = r_wins / vol_ewma_lag / seasonal

    out: dict[str, np.ndarray] = {"x_std": x_std, "vol_ewma": vol_ewma}
    for name, w in (("ret_5m", 5), ("ret_15m", 15), ("ret_60m", 60)):
        out[name] = rolling_sum_strict(r, w)

    # variable-window (B_d) rolling sum for ret_1d + oi_chg_1d
    idx = np.arange(n)
    lag = np.maximum(idx - b_day.astype(np.int64), -1)
    csum = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(r), r, 0.0))])
    cbad = np.concatenate([[0], np.cumsum(~np.isfinite(r))])
    valid = lag >= 0
    lag_c = np.maximum(lag, 0)
    ret_1d = np.where(
        valid & ((cbad[idx + 1] - cbad[lag_c + 1]) == 0),
        csum[idx + 1] - csum[lag_c + 1],
        np.nan,
    )
    out["ret_1d"] = ret_1d

    oi = np.where(
        valid & (seg[lag_c] == seg) & (position[lag_c] > 0),
        position / np.maximum(position[lag_c], 1e-12) - 1.0,
        np.nan,
    )
    if break_date is not None:
        bd = (break_date - dt.date(1970, 1, 1)).days
        near = np.abs(td - bd) <= 3  # calendar slack; window is ~1 trading day
        near_win = near | np.concatenate([[False], near[:-1]])
        oi = np.where(near_win, np.nan, oi)
    out["oi_chg_1d"] = oi

    for name, w in (("z_60", 60), ("z_240", 240), ("z_1200", 1200)):
        mean, std = rolling_mean_std(close, w)
        out[name] = (close - mean) / np.where(std > 0, std, np.nan)

    v60 = ewma(volume, np.full(n, 60.0))
    v1200 = ewma(volume, np.full(n, 1200.0))
    out["vol_ratio"] = v60 / np.where(v1200 > 0, v1200, np.nan)

    for name, w in (("range_1", 1), ("range_15", 15), ("range_60", 60)):
        hi = rolling_extreme(high, w, "max")
        lo = rolling_extreme(low, w, "min")
        out[name] = (hi - lo) / close

    return out


# ------------------------------------------------------------------ build stage


def build_features(
    paths: FuturesPaths,
    *,
    products: Sequence[str] | None = None,
    years: Sequence[int],
    volume_cfg: Mapping[str, Any] | None = None,
) -> FeatureReport:
    years = sorted(years)
    meta0 = memmap_io.read_meta(paths, years[0])
    products = list(meta0["products"]) if products is None else sorted(p.upper() for p in products)

    # per-exchange counting-break dates for oi_chg_1d masking
    break_by_exchange: dict[str, dt.date] = {}
    for exch, section in ((volume_cfg or {}).get("exchanges") or {}).items():
        bd = section.get("break_date")
        if bd is not None:
            break_by_exchange[str(exch)] = (
                bd if isinstance(bd, dt.date) else dt.date.fromisoformat(str(bd))
            )

    # grid keys + template ids per year (template via session table lookup)
    from data_fetcher.calendar import load_session_table

    table = load_session_table(paths)

    grids: dict[int, pl.DataFrame] = {}
    grid_keys: dict[int, np.ndarray] = {}
    writers: dict[int, dict[str, np.memmap]] = {}
    for year in years:
        meta = memmap_io.read_meta(paths, year)
        assert meta["products"] == list(products), (
            f"panel {year} products {meta['products']} != requested {products}"
        )
        grid = pl.read_parquet(paths.bob_index(year))
        grids[year] = grid
        grid_keys[year] = (
            grid["trading_date"].cast(pl.Int32).to_numpy().astype(np.int64) * _MIN_PER_DAY * 2
            + grid["rel_min"].to_numpy()
        )
        writers[year] = {
            name: memmap_io.create_array(paths, year, "feature", name, (meta["T"], meta["N"]))
            for name in FEATURES
        }

    for i, product in enumerate(products):
        bars = _load_product_bars(paths, product, years, table)
        if bars.is_empty():
            logger.warning("features: no bars for {} — skipped", product)
            continue
        exchange = _exchange_of(paths, product)
        feats = compute_product_features(
            bars, break_date=break_by_exchange.get(exchange or "", None)
        )
        key = (
            bars["trading_date"].cast(pl.Int32).to_numpy().astype(np.int64) * _MIN_PER_DAY * 2
            + bars["rel_min"].to_numpy()
        )
        for year in years:
            gk = grid_keys[year]
            pos = np.searchsorted(gk, key)
            ok = (pos < gk.size) & (gk[np.minimum(pos, gk.size - 1)] == key)
            rows = pos[ok]
            for name in FEATURES:
                writers[year][name][rows, i] = feats[name][ok].astype(np.float32)
        logger.info("features {}: {} bars", product, bars.height)

    for year in years:
        for arr in writers[year].values():
            arr.flush()
        meta = memmap_io.read_meta(paths, year)
        meta["features"] = list(FEATURES)
        memmap_io.write_meta(paths, meta)

    return FeatureReport(years=list(years), products=list(products), features=list(FEATURES))


def _load_product_bars(
    paths: FuturesPaths,
    product: str,
    years: Sequence[int],
    table: Any,
) -> pl.DataFrame:
    """Compressed traded-bar series: warmup years + requested years, annotated."""
    from data_fetcher.continuous import HOLE_LAST_DAY

    candidates = [years[0] - 1, *years]
    frames = []
    for year in candidates:
        path = paths.continuous_year(product, year)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="vertical").sort("bob")
    # warmup restriction: keep last WARMUP_DAYS trading days before years[0]
    first_needed = dt.date(years[0], 1, 1)
    warm_days = (
        df.filter(pl.col("trading_date") < first_needed)["trading_date"]
        .unique()
        .sort()
        .tail(WARMUP_DAYS)
    )
    keep_from = warm_days[0] if len(warm_days) else first_needed
    df = df.filter(pl.col("trading_date") >= keep_from)

    df = df.with_columns(
        (
            (
                pl.col("bob").dt.hour().cast(pl.Int32) * 60
                + pl.col("bob").dt.minute().cast(pl.Int32)
                - ANCHOR_MIN
            ).mod(_MIN_PER_DAY)
        ).alias("rel_min"),
        (pl.col("close").cast(pl.Float64) * pl.col("adj_factor"))
        .cast(pl.Float32)
        .alias("close_adj"),
        (pl.col("high").cast(pl.Float64) * pl.col("adj_factor")).cast(pl.Float32).alias("high_adj"),
        (pl.col("low").cast(pl.Float64) * pl.col("adj_factor")).cast(pl.Float32).alias("low_adj"),
        (pl.col("trading_date") > HOLE_LAST_DAY).cast(pl.Int8).alias("seg_macro"),
    )
    # per-day template id (session table lookup, distinct days only)
    days = df["trading_date"].unique().sort().to_list()
    tid_map = {d: (table.template_for(product, d) or -1) for d in days}
    return df.with_columns(
        pl.col("trading_date").replace_strict(tid_map, return_dtype=pl.Int64).alias("template_id")
    )


def _exchange_of(paths: FuturesPaths, product: str) -> str | None:
    hits = list(paths.raw_root.glob(f"exchange=*/product={product}"))
    return hits[0].parent.name.removeprefix("exchange=") if hits else None
