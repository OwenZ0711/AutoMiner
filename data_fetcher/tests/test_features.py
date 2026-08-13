"""Feature tests: rolling-helper references, causality (truncation gate), warmup."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest

from data_fetcher.panel.features import (
    compute_product_features,
    ewma,
    rolling_extreme,
    rolling_mean_std,
    rolling_sum_strict,
)

RNG = np.random.default_rng(42)


# ------------------------------------------------------ helpers vs pandas


def test_rolling_sum_strict_vs_pandas() -> None:
    x = RNG.normal(size=200)
    x[[5, 50]] = np.nan
    ours = rolling_sum_strict(x, 10)
    ref = pd.Series(x).rolling(10, min_periods=10).sum().to_numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-12, equal_nan=True)


def test_rolling_mean_std_vs_pandas() -> None:
    x = RNG.normal(size=300)
    x[7] = np.nan
    mean, std = rolling_mean_std(x, 20)
    ref_m = pd.Series(x).rolling(20, min_periods=20).mean().to_numpy()
    ref_s = pd.Series(x).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    np.testing.assert_allclose(mean, ref_m, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(std, ref_s, atol=1e-10, equal_nan=True)


def test_rolling_extreme_vs_pandas() -> None:
    x = RNG.normal(size=150)
    ours = rolling_extreme(x, 15, "max")
    ref = pd.Series(x).rolling(15, min_periods=15).max().to_numpy()
    np.testing.assert_allclose(ours, ref, equal_nan=True)


def test_ewma_constant_halflife_vs_pandas() -> None:
    x = RNG.normal(size=400) ** 2
    ours = ewma(x, np.full(x.size, 30.0))
    ref = pd.Series(x).ewm(halflife=30, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(ours, ref, rtol=1e-10)


# ------------------------------------------------------ synthetic bar frames


def _bars(n_days: int = 80, bars_per_day: int = 120, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = []
    d = dt.date(2020, 1, 6)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    price = 100.0
    for day in days:
        for k in range(bars_per_day):
            price *= float(np.exp(rng.normal(0, 0.0005)))
            rows.append(
                {
                    "trading_date": day,
                    "rel_min": 750 + k,  # day-session rel block
                    "close_adj": price,
                    "high_adj": price * 1.001,
                    "low_adj": price * 0.999,
                    "volume": float(rng.integers(1, 100)),
                    "position": 1000.0 + k,
                    "seg_macro": 0,
                    "template_id": 1,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("close_adj").cast(pl.Float32),
        pl.col("high_adj").cast(pl.Float32),
        pl.col("low_adj").cast(pl.Float32),
        pl.col("rel_min").cast(pl.Int32),
        pl.col("seg_macro").cast(pl.Int8),
    )


def test_truncation_prefix_gate() -> None:
    """LOOK-AHEAD GATE (a): truncate input at day d → bitwise-equal prefix."""
    bars = _bars(n_days=80)
    full = compute_product_features(bars, break_date=None)
    cut_day = bars["trading_date"].unique().sort()[60]
    truncated = bars.filter(pl.col("trading_date") <= cut_day)
    trunc = compute_product_features(truncated, break_date=None)
    n = truncated.height
    for name, arr in trunc.items():
        f32_full = full[name][:n].astype(np.float32)
        f32_trunc = arr.astype(np.float32)
        same = (f32_full == f32_trunc) | (np.isnan(f32_full) & np.isnan(f32_trunc))
        assert same.all(), f"{name}: {np.flatnonzero(~same)[:5]} differ after truncation"


def test_x_std_uses_lagged_vol() -> None:
    """x_std at t must not depend on r_t's own contribution to the EWMA."""
    bars = _bars(n_days=70)
    feats = compute_product_features(bars, break_date=None)
    x = feats["x_std"]
    finite = np.isfinite(x)
    assert finite.sum() > 1000
    # a huge return at the last bar must NOT shrink its own x_std via vol:
    bars2 = bars.with_columns(
        pl.when(pl.arange(0, bars.height) == bars.height - 1)
        .then(pl.col("close_adj") * 1.05)
        .otherwise(pl.col("close_adj"))
        .cast(pl.Float32)
        .alias("close_adj")
    )
    feats2 = compute_product_features(bars2, break_date=None)
    # the spiked bar reads large but is winsorized to ~5σ (the cap is the design)
    assert abs(feats2["x_std"][-1]) > 3.0
    # and NO earlier value may change — the no-backward-leakage property
    for name, arr in feats2.items():
        a, b = feats[name][:-1], arr[:-1]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        assert same.all(), f"{name}: modifying the last bar changed earlier values"


def test_ret_windows_and_nan_warmup() -> None:
    bars = _bars(n_days=10)
    feats = compute_product_features(bars, break_date=None)
    r5 = feats["ret_5m"]
    assert np.isnan(r5[:5]).all()  # first bar has NaN return → window of 5 needs 6 bars
    assert np.isfinite(r5[6:120]).all()
    # ret_5m == sum of last 5 one-bar log returns
    close = bars["close_adj"].to_numpy().astype(np.float64)
    manual = np.log(close[100]) - np.log(close[95])
    np.testing.assert_allclose(r5[100], manual, rtol=1e-6)


def test_oi_chg_masked_around_break_date() -> None:
    bars = _bars(n_days=30)
    days = bars["trading_date"].unique().sort().to_list()
    break_day = days[15]
    feats = compute_product_features(bars, break_date=break_day)
    oi = feats["oi_chg_1d"]
    td = bars["trading_date"].to_numpy()
    near = np.abs((td - np.datetime64(break_day)).astype("timedelta64[D]").astype(int)) <= 3
    assert np.isnan(oi[near]).all()
    far_after = td > np.datetime64(break_day + dt.timedelta(days=10))
    assert np.isfinite(oi[far_after]).sum() > 0


def test_seg_macro_boundary_poisons_windows() -> None:
    bars = _bars(n_days=20)
    n = bars.height
    seg = np.zeros(n, dtype=np.int8)
    seg[n // 2 :] = 1  # artificial hole in the middle
    bars = bars.with_columns(pl.Series("seg_macro", seg))
    feats = compute_product_features(bars, break_date=None)
    b = n // 2
    assert np.isnan(feats["ret_5m"][b : b + 5]).all()  # windows crossing the hole die
    assert np.isfinite(feats["ret_5m"][b + 6 : b + 120]).all()


def test_cross_year_warmup_equals_single_pass() -> None:
    """Building with a warmup prefix must reproduce the single-pass values."""
    bars = _bars(n_days=200)
    full = compute_product_features(bars, break_date=None)
    days = bars["trading_date"].unique().sort()
    warm_start = days[len(days) - 130 - 40]  # 130-day warmup before the last 40 days
    subset = bars.filter(pl.col("trading_date") >= warm_start)
    part = compute_product_features(subset, break_date=None)
    tail = bars.filter(pl.col("trading_date") >= days[len(days) - 40]).height
    for name in ("vol_ewma", "vol_ratio", "x_std", "z_1200", "ret_1d"):
        a = full[name][-tail:].astype(np.float32)
        b = part[name][-tail:].astype(np.float32)
        both = np.isfinite(a) & np.isfinite(b)
        # EWMA memory beyond 130 trading days is below f32 resolution
        np.testing.assert_allclose(a[both], b[both], rtol=2e-5, atol=1e-7, err_msg=name)
        nan_mismatch = np.isnan(a) != np.isnan(b)
        assert not nan_mismatch.any(), f"{name}: NaN pattern differs"


@pytest.mark.realdata
def test_real_panel_and_features_acceptance() -> None:
    from futures_common.paths import default_paths

    paths = default_paths()
    if not paths.panel_meta(2020).exists():
        pytest.skip("panel not built")
    from data_fetcher.panel import memmap_io

    meta = memmap_io.read_meta(paths, 2020)
    x = memmap_io.open_array(paths, 2020, "feature", "x_std")
    m = memmap_io.open_array(paths, 2020, "mask", "m_traded")
    coverage = np.isfinite(x[m.astype(bool)]).mean()
    assert coverage > 0.9, f"x_std coverage {coverage:.2%}"
    assert meta["T"] > 50_000
