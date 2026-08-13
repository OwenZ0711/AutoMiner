"""Flagship regression net: a planted lead-lag relationship must survive the
ENTIRE pipeline — raw CSVs → ingest → calendar → continuous → panel →
features → scan → validation → edge table → GP cross terminals → backtest.

Fixture: two fake products, RB (leader) and HC (follower), ~150 trading days
of day-session minute bars, with

    r_HC(t) = 0.4 · r_RB(t − 7) + noise

and one planted dominant-contract roll per product. Assertions:
- the scan recovers lag ≈ 7 with the right direction ordering;
- validation admits RB→HC (and NOT HC→RB);
- the GP lead-lag seed earns positive train fitness through the edge
  terminals, and the +1-bar kill-switch destroys most of it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from data_fetcher.calendar import CalendarConfig, build_calendar
from data_fetcher.continuous import build_continuous
from data_fetcher.ingest import ingest
from data_fetcher.panel.align import build_panel
from data_fetcher.panel.features import build_features
from data_fetcher.tests.conftest import (
    Bar,
    ContractSpec,
    write_daily_tree,
    write_historical_tree,
)
from futures_common.paths import FuturesPaths
from futures_common.rng import make_rng
from gp_cta.gp.backtest import BacktestConfig, ProductCostArrays
from gp_cta.gp.engine import run_mining_round
from gp_cta.gp.expressions import TreeSpace
from gp_cta.gp.terminals import build_context
from gp_cta.gp.warmstart import EngineConfig, seed_from_edge
from gp_cta.leadlag.edges import EdgeStore
from gp_cta.leadlag.scan import ScanConfig, scan_lagged_corr
from gp_cta.leadlag.validate import (
    PriorSet,
    apply_fdr,
    pair_statistics,
    permutation_null,
)
from gp_cta.ledger import TrialLedger
from gp_cta.panel_io import PanelReader

LAG_TRUE = 7
BETA = 0.4
LAG_GRID = (1, 2, 3, 5, 7, 9, 12, 15)


def _weekdays(start: dt.date, n: int) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _minutes_of_day() -> list[dt.time]:
    """Compressed day session: 09:00–10:15 + 10:30–11:30 (135 bars/day)."""
    out = []
    for h, m0, m1 in ((9, 0, 60), (10, 0, 15), (10, 30, 60), (11, 0, 30)):
        for m in range(m0, m1):
            out.append(dt.time(h, m))
    return out


def _build_raw_tree(tmp_path):  # type: ignore[no-untyped-def]
    rng = make_rng(42, "e2e.planted")
    days = _weekdays(dt.date(2020, 1, 6), 150)
    minutes = _minutes_of_day()
    bars_per_day = len(minutes)
    t_total = len(days) * bars_per_day

    r_rb = rng.normal(0, 0.0015, size=t_total)
    noise = rng.normal(0, 0.0015, size=t_total)
    r_hc = np.empty(t_total)
    r_hc[:LAG_TRUE] = noise[:LAG_TRUE]
    r_hc[LAG_TRUE:] = BETA * r_rb[:-LAG_TRUE] + noise[LAG_TRUE:]
    prices = {
        "RB": 3000.0 * np.exp(np.cumsum(r_rb)),
        "HC": 3500.0 * np.exp(np.cumsum(r_hc)),
    }

    specs = []
    roll_day = 100  # planted dominant switch around day 100
    for product, px in prices.items():
        near_bars, far_bars = [], []
        k = 0
        for di, day in enumerate(days):
            near_vol = 1000 if di < roll_day else 50
            far_vol = 60 if di < roll_day - 2 else 1500  # crossover 2 days early
            for minute in minutes:
                p = float(px[k])
                bob = dt.datetime.combine(day, minute)
                near_bars.append(
                    Bar(
                        bob=bob,
                        open=p,
                        close=p,
                        high=p * 1.0001,
                        low=p * 0.9999,
                        volume=float(near_vol),
                        position=5000.0,
                    )
                )
                far_bars.append(
                    Bar(
                        bob=bob,
                        open=p * 1.01,
                        close=p * 1.01,
                        high=p * 1.0101,
                        low=p * 1.0099,
                        volume=float(far_vol),
                        position=3000.0,
                    )
                )
                k += 1
        specs.append(ContractSpec("SHFE", f"{product}2005", near_bars))
        specs.append(ContractSpec("SHFE", f"{product}2010", far_bars))
    hist = write_historical_tree(tmp_path, specs)
    daily = write_daily_tree(tmp_path, [])  # empty 2026 side
    return hist, daily, days


@pytest.mark.slow
def test_planted_leadlag_survives_full_pipeline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = FuturesPaths(data_root=tmp_path / "data", results_root=tmp_path / "results")
    hist, daily, days = _build_raw_tree(tmp_path / "raw")

    # ---------- data band
    report = ingest(
        paths,
        raw_historical=hist,
        raw_2026=daily,
        products=None,
        start=days[0],
        end=days[-1],
        anchors=["RB", "HC"],
        anchors_required=2,
    )
    assert report.contracts_written == 4

    build_calendar(paths, products=["RB", "HC"], cfg=CalendarConfig(anchors_required=2))
    cont = build_continuous(paths, products=["RB", "HC"], start=days[0], end=days[-1])
    assert cont.rolls == 2  # one planted roll per product

    years = [2020]
    build_panel(paths, products=["RB", "HC"], years=years)
    build_features(paths, years=years)

    panel = PanelReader(paths, years)
    assert panel.products == ("HC", "RB")

    # ---------- lead-lag band
    cfg = ScanConfig(max_lag=15, min_joint_bars=3000)
    scan = scan_lagged_corr(panel, cfg, lags=LAG_GRID)
    stats = pair_statistics(scan, lag_grid=LAG_GRID)
    rb_hc = stats.filter((pl.col("leader") == "RB") & (pl.col("follower") == "HC"))
    assert rb_hc["lag_bars"][0] == LAG_TRUE, f"recovered lag {rb_hc['lag_bars'][0]} != 7"
    assert rb_hc["sign"][0] == 1
    assert rb_hc["a_stat"][0] > 0.1

    stats = permutation_null(
        panel, stats, cfg, lag_grid=LAG_GRID, n_draws=99, min_shift_days=5, seed=42
    )
    stats = apply_fdr(stats, PriorSet.from_config([["RB", "HC"]]), q_a=0.05, q_b=0.10)
    validated = {
        (r["leader"], r["follower"])
        for r in stats.filter(pl.col("validated")).iter_rows(named=True)
    }
    assert ("RB", "HC") in validated
    assert ("HC", "RB") not in validated  # direction, not just association

    # ---------- edge table (asof before the panel: synthetic plumbing test)
    edges_frame = stats.filter(pl.col("validated")).with_columns(
        pl.lit("").alias("regime_flags"),
        pl.lit(days[0]).alias("window_start"),
        pl.lit(days[-1]).alias("window_end"),
        pl.lit(1.0, dtype=pl.Float32).alias("stability"),
        pl.lit(6.0, dtype=pl.Float32).alias("half_life"),
    )
    store = EdgeStore(paths)
    store.append(edges_frame, asof=days[0] - dt.timedelta(days=1), config_hash="e2e" * 4)
    leaders = store.leaders_for("HC", days[10], max_staleness_days=None)
    assert leaders and leaders[0].leader == "RB" and leaders[0].lag_bars == LAG_TRUE

    # ---------- GP band through the edge terminals
    ctx = build_context(panel, edges=store, max_staleness_days=None, min_stability=0.0)
    assert np.abs(ctx.arrays["lead_ret_1"]).sum() > 0  # terminals resolved
    ctx_shifted = build_context(
        panel, edges=store, max_staleness_days=None, min_stability=0.0, shift_bars=1
    )
    costs = {
        p: ProductCostArrays(fee_bps=0.2, slippage_ticks=0.0, tick_size=1.0) for p in panel.products
    }
    bt = BacktestConfig(q_hi=0.7, q_lo=0.3, lookback=270, min_periods=135, max_hold_bars=135)
    cut = days[int(len(days) * 0.7)]
    ledger = TrialLedger(paths, config_hash="e2e")
    ws = run_mining_round(
        panel,
        ctx,
        ctx_shifted,
        costs,
        [seed_from_edge(leaders[0])],
        paths=paths,
        ledger=ledger,
        engine_cfg=EngineConfig(pop_size=8, generations=3, seed=42),
        bt_cfg=bt,
        space=TreeSpace(max_depth=5),
        train=(days[0], cut),
        validation=(cut + dt.timedelta(days=1), days[-1]),
        round_id="e2e",
    )
    best_train = max(c.train.fitness for c in ws.candidates)
    assert best_train > 0.5, f"edge seed failed to profit on planted panel: {best_train}"
    # +1-bar staleness must destroy most of the planted edge
    assert ws.killswitch_shifted_fitness < 0.5 * best_train
    assert ledger.count() > 0
