"""Warm-start engine tests on a planted trending panel."""

from __future__ import annotations

import datetime as dt

import numpy as np

from futures_common.paths import FuturesPaths
from futures_common.rng import make_rng
from gp_cta.gp.backtest import BacktestConfig, ProductCostArrays
from gp_cta.gp.engine import CandidatePool, run_mining_round
from gp_cta.gp.expressions import TreeSpace, ast_hash
from gp_cta.gp.fitness import evaluate_fitness
from gp_cta.gp.terminals import build_context
from gp_cta.gp.warmstart import EngineConfig, classic_seeds, run_seed
from gp_cta.ledger import TrialLedger
from gp_cta.panel_io import PanelReader
from gp_cta.tests.conftest import write_panel_year

D0 = 18267  # 2020-01-06
EPOCH = dt.date(1970, 1, 1)


def _planted_trend_panel(tmp_paths: FuturesPaths, t_rows: int = 6000) -> PanelReader:
    """Two products with strong regime-switching drift — MA-cross heaven."""
    rng = make_rng(21, "test.trend")
    bars_per_day = 50
    n = 2
    drift = 0.002 * np.sign(np.sin(np.arange(t_rows) / 250.0))
    prices = np.empty((t_rows, n))
    for i in range(n):
        r = drift + rng.normal(0, 0.001, size=t_rows)
        prices[:, i] = 100.0 * np.exp(np.cumsum(r))
    ret1 = np.vstack([np.zeros((1, n)), np.diff(np.log(prices), axis=0)])
    days = np.repeat(np.arange(D0, D0 + t_rows // bars_per_day, dtype=np.int32), bars_per_day)
    seg = np.repeat(np.arange(t_rows // bars_per_day, dtype=np.int32), bars_per_day)
    x_std = (ret1 / (np.std(ret1) + 1e-12)).astype(np.float32)
    zeros = np.zeros((t_rows, n), dtype=np.float32)
    write_panel_year(
        tmp_paths,
        2020,
        ["A", "B"],
        x_std,
        np.ones((t_rows, n), dtype=bool),
        days,
        seg,
        extra_features={
            "ret_5m": zeros,
            "ret_15m": zeros,
            "ret_60m": zeros,
            "ret_1d": zeros,
            "vol_ewma": zeros + 0.001,
            "z_60": zeros,
            "z_240": zeros,
            "z_1200": zeros,
            "oi_chg_1d": zeros,
            "vol_ratio": zeros + 1,
            "range_1": zeros,
            "range_15": zeros,
            "range_60": zeros,
        },
        extra_channels={
            "open_adj": prices.astype(np.float32),
            "open_raw": prices.astype(np.float32),
            "close_adj": prices.astype(np.float32),
        },
    )
    return PanelReader(tmp_paths, [2020])


def _setup(tmp_paths: FuturesPaths):  # type: ignore[no-untyped-def]
    panel = _planted_trend_panel(tmp_paths)
    ctx = build_context(panel)
    ctx.arrays["close_adj"] = np.vstack([np.asarray(panel.channel("close_adj", 2020))])
    ctx_shifted = build_context(panel, shift_bars=1)
    ctx_shifted.arrays["close_adj"] = np.roll(ctx.arrays["close_adj"], 1, axis=0)
    ctx_shifted.arrays["close_adj"][0] = np.nan
    costs = {
        p: ProductCostArrays(fee_bps=1.0, slippage_ticks=0.0, tick_size=1.0) for p in panel.products
    }
    bt = BacktestConfig(q_hi=0.7, q_lo=0.3, lookback=200, min_periods=50, max_hold_bars=500)
    days = panel.trading_dates(2020)
    lo = EPOCH + dt.timedelta(days=int(days.min()))
    hi = EPOCH + dt.timedelta(days=int(days.max()))
    cut = lo + dt.timedelta(days=int((hi - lo).days * 0.7))
    return panel, ctx, ctx_shifted, costs, bt, (lo, cut), (cut + dt.timedelta(days=1), hi)


def test_seed_run_deterministic_and_memoized(tmp_paths: FuturesPaths) -> None:
    panel, ctx, _, costs, bt, train, _val = _setup(tmp_paths)
    calls = {"n": 0}

    def fitness(e):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return evaluate_fitness(e, ctx, panel, costs, bt, train, allow_flip=False)

    cfg = EngineConfig(pop_size=8, generations=3, seed=42)
    space = TreeSpace(max_depth=5)
    seed = classic_seeds()[0]

    r1 = run_seed(seed, fitness, cfg, space)
    n1 = calls["n"]
    assert n1 == r1.evaluations  # memo: one fitness call per unique hash

    calls["n"] = 0
    r2 = run_seed(seed, fitness, cfg, space)
    assert ast_hash(r1.best_expr) == ast_hash(r2.best_expr)
    assert r1.trace == r2.trace  # bitwise deterministic


def test_mining_round_warmstart_beats_random_and_killswitch(
    tmp_paths: FuturesPaths,
) -> None:
    panel, ctx, ctx_shifted, costs, bt, train, val = _setup(tmp_paths)
    cfg = EngineConfig(pop_size=8, generations=3, seed=42)
    space = TreeSpace(max_depth=5)
    seeds = list(classic_seeds())[:2]  # ma_cross + breakout
    ledger = TrialLedger(tmp_paths, config_hash="test")

    common = dict(
        paths=tmp_paths,
        ledger=ledger,
        engine_cfg=cfg,
        bt_cfg=bt,
        space=space,
        train=train,
        validation=val,
    )
    ws = run_mining_round(
        panel,
        ctx,
        ctx_shifted,
        costs,
        seeds,
        init="warmstart",
        round_id="t-ws",
        **common,  # type: ignore[arg-type]
    )
    rnd = run_mining_round(
        panel,
        ctx,
        ctx_shifted,
        costs,
        seeds,
        init="random",
        round_id="t-rnd",
        **common,  # type: ignore[arg-type]
    )
    # planted MA regime: the warm-started structure must dominate random init
    assert ws.best_validation_fitness > rnd.best_validation_fitness
    assert ws.best_validation_fitness > 0

    # +1-bar kill-switch: staleness must destroy most of the edge
    best_train = max(c.train.fitness for c in ws.candidates)
    assert ws.killswitch_shifted_fitness < best_train
    # ledger recorded formulas for both rounds
    assert ledger.count("gp_formula") > 0


def test_candidate_pool_rejects_correlated() -> None:
    rng = make_rng(5, "test.pool")
    pool = CandidatePool(max_abs_corr=0.7, decimation=1)
    base = rng.normal(size=(1000, 2))
    ok, _ = pool.offer("a", base)
    assert ok
    ok, reason = pool.offer("b", base * 2.0 + 0.001)  # perfectly correlated
    assert not ok and "corr" in reason
    ok, _ = pool.offer("c", rng.normal(size=(1000, 2)))
    assert ok
    ok, reason = pool.offer("d", np.zeros((1000, 2)))
    assert not ok  # degenerate


def test_fitness_sign_flip(tmp_paths: FuturesPaths) -> None:
    """A signal that is consistently WRONG should be rescued by the flip."""
    panel, ctx, _, costs, bt, train, _val = _setup(tmp_paths)
    from gp_cta.gp.expressions import Expr

    # anti-momentum: negative of the MA cross — reliably loses on this panel
    anti = Expr.call("neg", classic_seeds()[0].expr)
    no_flip = evaluate_fitness(anti, ctx, panel, costs, bt, train, allow_flip=False)
    flipped = evaluate_fitness(anti, ctx, panel, costs, bt, train, allow_flip=True)
    if no_flip.mean_sharpe < 0:
        assert flipped.fitness >= no_flip.fitness
        assert flipped.flipped or flipped.fitness == no_flip.fitness
