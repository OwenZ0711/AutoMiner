"""`gp_cta mine` — one warm-start mining round over the requested slice."""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Mapping
from typing import Any

import numpy as np
from loguru import logger

from futures_common.config import DEFAULT_CONFIG_DIR, load_costs, load_yaml
from futures_common.stage import StageContext, run_stage


def _resolve_splits(
    ctx: StageContext,
) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    """Configured splits when the slice covers them; else 70/30 within slice."""
    train = ctx.defaults.splits.train
    valid = ctx.defaults.splits.validation
    t0, t1 = max(train[0], ctx.start), min(train[1], ctx.end)
    v0, v1 = max(valid[0], ctx.start), min(valid[1], ctx.end)
    if t0 < t1 and v0 < v1:
        return (t0, t1), (v0, v1)
    span = (ctx.end - ctx.start).days
    cut = ctx.start + dt.timedelta(days=int(span * 0.7))
    logger.info(
        "slice [{}, {}] outside configured splits — using 70/30 cut at {}",
        ctx.start,
        ctx.end,
        cut,
    )
    return (ctx.start, cut), (cut + dt.timedelta(days=1), ctx.end)


def cmd_mine(args: argparse.Namespace) -> int:
    from gp_cta.gp.backtest import BacktestConfig, product_costs
    from gp_cta.gp.engine import run_mining_round
    from gp_cta.gp.expressions import DEFAULT_FEATURES, TreeSpace
    from gp_cta.gp.terminals import build_context
    from gp_cta.gp.warmstart import EngineConfig, classic_seeds, seed_from_edge
    from gp_cta.leadlag.edges import EdgeStore
    from gp_cta.ledger import TrialLedger
    from gp_cta.panel_io import PanelReader

    def work(ctx: StageContext) -> Mapping[str, Any]:
        gp_raw = load_yaml(DEFAULT_CONFIG_DIR / "gp.yaml")
        bt_raw = gp_raw.get("backtest", {})
        gates = gp_raw.get("edges_gate", {})
        pool_raw = gp_raw.get("pool", {})

        years = list(range(ctx.start.year, ctx.end.year + 1))
        panel = PanelReader(ctx.paths, years)
        edges = EdgeStore(ctx.paths)
        costs_cfg = load_costs()
        costs = {p: product_costs(costs_cfg, p) for p in panel.products}

        bt_cfg = BacktestConfig(
            q_hi=float(bt_raw.get("q_hi", 0.70)),
            q_lo=float(bt_raw.get("q_lo", 0.30)),
            lookback=int(bt_raw.get("lookback", 1200)),
            min_periods=int(bt_raw.get("min_periods", 300)),
            max_hold_bars=int(bt_raw.get("max_hold_bars", 345)),
        )
        engine_cfg = EngineConfig(
            pop_size=int(gp_raw.get("population", 64)),
            generations=int(gp_raw.get("generations", 10)),
            stagnation=int(gp_raw.get("stagnation", 3)),
            tournament_k=int(gp_raw.get("tournament_k", 4)),
            p_crossover=float(gp_raw.get("probs", {}).get("crossover", 0.5)),
            p_point=float(gp_raw.get("probs", {}).get("point", 0.4)),
            seed=ctx.seed,
        )
        space = TreeSpace(
            features=DEFAULT_FEATURES,
            windows=tuple(int(w) for w in gp_raw.get("windows", [])) or TreeSpace().windows,
            constants=tuple(float(c) for c in gp_raw.get("constants", [])) or TreeSpace().constants,
            max_depth=int(gp_raw.get("max_depth", 6)),
        )
        top_k = int(gp_raw.get("max_leader_rank", 3))
        q_max = float(gates.get("q_max", 0.10))
        min_stability = float(gates.get("min_stability", 0.5))
        staleness = int(gates.get("max_staleness_days", 45))

        ctx_eval = build_context(
            panel,
            edges=edges,
            top_k=top_k,
            q_max=q_max,
            min_stability=min_stability,
            max_staleness_days=staleness,
        )
        ctx_shifted = build_context(
            panel,
            edges=edges,
            top_k=top_k,
            q_max=q_max,
            min_stability=min_stability,
            max_staleness_days=staleness,
            shift_bars=1,
        )

        train, validation = _resolve_splits(ctx)
        logger.info("splits: train [{}, {}] validation [{}, {}]", *train, *validation)

        seeds = list(classic_seeds())
        seen_edge_pairs: set[tuple[str, str]] = set()
        for follower in panel.products:
            for edge in edges.leaders_for(
                follower,
                validation[0],
                top_k=top_k,
                q_max=q_max,
                min_stability=min_stability,
                max_staleness_days=None,
            ):
                pair = (edge.leader, edge.follower)
                if pair not in seen_edge_pairs and edge.leader in panel.products:
                    seen_edge_pairs.add(pair)
                    seeds.append(seed_from_edge(edge))
        logger.info("seeds: {} classic + {} edge", len(classic_seeds()), len(seen_edge_pairs))

        ledger = TrialLedger(ctx.paths, config_hash="mine")
        round_id = args.round_id or f"{ctx.start:%Y%m%d}-{ctx.end:%Y%m%d}-s{ctx.seed}"
        report = run_mining_round(
            panel,
            ctx_eval,
            ctx_shifted,
            costs,
            seeds,
            paths=ctx.paths,
            ledger=ledger,
            engine_cfg=engine_cfg,
            bt_cfg=bt_cfg,
            space=space,
            train=train,
            validation=validation,
            pool_max_corr=float(pool_raw.get("max_abs_corr", 0.7)),
            pool_decimation=int(pool_raw.get("signal_decimation_bars", 15)),
            init=args.init,
            round_id=round_id,
        )
        out = dict(report.as_dict())
        best = out["best_validation_fitness"]
        out["best_validation_fitness"] = float(best) if np.isfinite(best) else -999.0
        return out

    return run_stage(f"mine-{args.init}", args, work)
