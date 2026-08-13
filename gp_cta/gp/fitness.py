"""Cost-adjusted fitness + validity gates + the one-time sign flip.

    fitness = mean_n(ann_sharpe_n) − 0.5 · mean_n(maxdd_n) − 0.05 · mean_n(turnover_n)
    valid   = mean trades/product-year ≥ 20 AND mean time-in-market ≥ 2%
    invalid → −10

A negative-mean-Sharpe formula is sign-flipped ONCE (the backtest re-runs on
−signal — not −positions, the quantile bands are asymmetric under flip),
BEFORE the validity gate; the flip is one extra ledger trial.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from gp_cta.gp.backtest import BacktestConfig, BacktestResult, ProductCostArrays, run_backtest
from gp_cta.gp.expressions import Expr, ast_hash
from gp_cta.gp.metrics import ProductMetrics, portfolio_metrics, product_metrics
from gp_cta.gp.terminals import EvalContext, evaluate
from gp_cta.ledger import TrialLedger
from gp_cta.panel_io import PanelReader

W_MAXDD = 0.5
W_TURNOVER = 0.05
INVALID_FITNESS = -10.0
MIN_TRADES_PER_PRODUCT_YEAR = 20.0
MIN_TIME_IN_MARKET = 0.02


@dataclass(slots=True)
class FitnessReport:
    fitness: float
    valid: bool
    flipped: bool
    mean_sharpe: float
    mean_maxdd: float
    mean_turnover: float
    per_product: list[ProductMetrics] = field(default_factory=list)
    portfolio: dict[str, Any] = field(default_factory=dict)

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "fitness": self.fitness,
            "valid": self.valid,
            "flipped": self.flipped,
            "mean_sharpe": self.mean_sharpe,
            "mean_maxdd": self.mean_maxdd,
            "mean_turnover": self.mean_turnover,
            "portfolio": self.portfolio,
            "per_product": [m.as_dict() for m in self.per_product],
        }


def _daily_slice(result: BacktestResult, split: tuple[dt.date, dt.date]) -> pl.DataFrame:
    return result.daily.filter(pl.col("trading_date").is_between(split[0], split[1], closed="both"))


def score_backtest(result: BacktestResult, split: tuple[dt.date, dt.date]) -> FitnessReport:
    daily = _daily_slice(result, split)
    years = max(daily.height / 252.0, 1e-9)
    pms: list[ProductMetrics] = []
    for row in result.per_product.iter_rows(named=True):
        product = str(row["product"])
        pms.append(
            product_metrics(
                daily[product].to_numpy(),
                product,
                trades=int(row["trades"]),
                turnover_per_day=float(row["turnover_per_day"]),
                time_in_market=float(row["time_in_market"]),
            )
        )
    active = [m for m in pms if m.n_days > 0]
    if not active:
        return FitnessReport(
            fitness=INVALID_FITNESS,
            valid=False,
            flipped=False,
            mean_sharpe=0.0,
            mean_maxdd=0.0,
            mean_turnover=0.0,
        )
    mean_sharpe = float(np.mean([m.ann_sharpe for m in active]))
    mean_maxdd = float(np.mean([m.max_drawdown for m in active]))
    mean_turnover = float(np.mean([m.turnover_per_day for m in active]))
    mean_trades_py = float(np.mean([m.trades / years for m in active]))
    mean_tim = float(np.mean([m.time_in_market for m in active]))

    fitness = mean_sharpe - W_MAXDD * mean_maxdd - W_TURNOVER * mean_turnover
    valid = mean_trades_py >= MIN_TRADES_PER_PRODUCT_YEAR and mean_tim >= MIN_TIME_IN_MARKET
    if not valid:
        fitness = INVALID_FITNESS
    return FitnessReport(
        fitness=float(fitness),
        valid=valid,
        flipped=False,
        mean_sharpe=mean_sharpe,
        mean_maxdd=mean_maxdd,
        mean_turnover=mean_turnover,
        per_product=pms,
        portfolio=portfolio_metrics(daily, result.products),
    )


def evaluate_fitness(
    expr: Expr,
    ctx: EvalContext,
    panel: PanelReader,
    costs: Mapping[str, ProductCostArrays],
    bt_cfg: BacktestConfig,
    split: tuple[dt.date, dt.date],
    *,
    ledger: TrialLedger | None = None,
    seed_id: str | None = None,
    split_name: str = "train",
    allow_flip: bool = True,
) -> FitnessReport:
    """Signal → backtest → metrics → fitness (with the one-time sign flip)."""
    signal = evaluate(expr, ctx)
    result = run_backtest(signal, panel, costs, bt_cfg)
    report = score_backtest(result, split)

    h = ast_hash(expr)
    if ledger is not None:
        ledger.log(
            "gp_formula",
            h,
            seed_id=seed_id,
            split=split_name,
            stat=report.fitness,
            metrics={"formula": str(expr), **report.metrics_dict()},
        )

    if allow_flip and report.mean_sharpe < 0:
        flipped_result = run_backtest(-signal, panel, costs, bt_cfg)
        flipped = score_backtest(flipped_result, split)
        flipped.flipped = True
        if ledger is not None:
            ledger.log(
                "gp_sign_flip",
                h,
                seed_id=seed_id,
                split=split_name,
                stat=flipped.fitness,
                metrics=flipped.metrics_dict(),
            )
        if flipped.fitness > report.fitness:
            return flipped
    return report
