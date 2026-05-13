"""Evaluate GP formulas as long/flat CTA strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from gp_cta.backtest import BacktestConfig, BacktestReport, run_long_flat_backtest
from gp_cta.expressions import Expr
from gp_cta.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class EvaluationConfig:
    cost_bps: float = 30.0
    threshold_lookback: int = 60
    threshold_quantile: float = 0.5
    min_periods: int = 20
    execution_lag: int = 1
    max_drawdown_weight: float = 0.5
    turnover_weight: float = 0.1
    min_trades: int = 2
    min_time_in_market: float = 0.01


@dataclass(frozen=True)
class EvaluationResult:
    formula: str
    fitness: float
    mean_sharpe: float
    mean_cagr: float
    mean_max_drawdown: float
    mean_daily_turnover: float
    mean_annual_turnover: float
    total_trades: int
    mean_time_in_market: float
    valid: bool

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "formula": self.formula,
            "fitness": self.fitness,
            "mean_sharpe": self.mean_sharpe,
            "mean_cagr": self.mean_cagr,
            "mean_max_drawdown": self.mean_max_drawdown,
            "mean_daily_turnover": self.mean_daily_turnover,
            "mean_annual_turnover": self.mean_annual_turnover,
            "total_trades": self.total_trades,
            "mean_time_in_market": self.mean_time_in_market,
            "valid": self.valid,
        }


class FormulaEvaluator:
    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self.backtest_config = BacktestConfig(
            cost_bps=self.config.cost_bps,
            threshold_lookback=self.config.threshold_lookback,
            threshold_quantile=self.config.threshold_quantile,
            min_periods=self.config.min_periods,
            execution_lag=self.config.execution_lag,
        )

    def evaluate(self, expr: Expr, feature_panel: pl.DataFrame) -> EvaluationResult:
        reports = self._reports(expr, feature_panel)
        if not reports:
            return _invalid_result(expr)

        sharpes = np.array([r.sharpe for r in reports], dtype=float)
        cagrs = np.array([r.cagr for r in reports], dtype=float)
        drawdowns = np.array([r.max_drawdown for r in reports], dtype=float)
        daily_turnovers = np.array([r.daily_turnover for r in reports], dtype=float)
        annual_turnovers = np.array([r.annual_turnover for r in reports], dtype=float)
        trades = int(sum(r.trade_count for r in reports))
        time_in_market = np.array([r.time_in_market for r in reports], dtype=float)

        mean_sharpe = float(np.mean(sharpes))
        mean_drawdown = float(np.mean(drawdowns))
        mean_daily_turnover = float(np.mean(daily_turnovers))
        mean_time = float(np.mean(time_in_market))
        valid = trades >= self.config.min_trades and mean_time >= self.config.min_time_in_market
        fitness = (
            mean_sharpe
            - self.config.max_drawdown_weight * mean_drawdown
            - self.config.turnover_weight * mean_daily_turnover
        )
        if not valid:
            fitness -= 10.0

        return EvaluationResult(
            formula=str(expr),
            fitness=float(fitness),
            mean_sharpe=mean_sharpe,
            mean_cagr=float(np.mean(cagrs)),
            mean_max_drawdown=mean_drawdown,
            mean_daily_turnover=mean_daily_turnover,
            mean_annual_turnover=float(np.mean(annual_turnovers)),
            total_trades=trades,
            mean_time_in_market=mean_time,
            valid=valid,
        )

    def portfolio_equity_curve(self, expr: Expr, feature_panel: pl.DataFrame) -> pd.DataFrame:
        """Equal-weight per-symbol strategy returns for the selected formula."""
        series_by_symbol: list[pd.Series] = []
        for report in self._reports(expr, feature_panel):
            series_by_symbol.append(report.daily_returns)
        if not series_by_symbol:
            return pd.DataFrame(columns=["date", "portfolio_return", "equity"])
        returns = pd.concat(series_by_symbol, axis=1).fillna(0.0).mean(axis=1)
        equity = (1.0 + returns).cumprod()
        return pd.DataFrame(
            {
                "date": returns.index.date,
                "portfolio_return": returns.to_numpy(),
                "equity": equity.to_numpy(),
            }
        )

    def _reports(self, expr: Expr, feature_panel: pl.DataFrame) -> list[BacktestReport]:
        reports: list[BacktestReport] = []
        for symbol in feature_panel["symbol"].unique().sort():
            frame = feature_panel.filter(pl.col("symbol") == symbol).sort("date")
            if frame.height < max(self.config.threshold_lookback, 5):
                continue
            missing = set(FEATURE_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"feature panel missing columns: {sorted(missing)}")
            signal = expr.evaluate(frame)
            report = run_long_flat_backtest(
                dates=frame["date"].to_list(),
                close=frame["close"].to_numpy(),
                signal=signal,
                config=self.backtest_config,
            )
            reports.append(report)
        return reports


def split_by_date(
    frame: pl.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2022-12-31",
) -> dict[str, pl.DataFrame]:
    train_end_date = pd.Timestamp(train_end).date()
    val_end_date = pd.Timestamp(val_end).date()
    return {
        "train": frame.filter(pl.col("date") <= train_end_date),
        "validation": frame.filter(
            (pl.col("date") > train_end_date) & (pl.col("date") <= val_end_date)
        ),
        "test": frame.filter(pl.col("date") > val_end_date),
    }


def _invalid_result(expr: Expr) -> EvaluationResult:
    return EvaluationResult(
        formula=str(expr),
        fitness=-10.0,
        mean_sharpe=0.0,
        mean_cagr=0.0,
        mean_max_drawdown=0.0,
        mean_daily_turnover=0.0,
        mean_annual_turnover=0.0,
        total_trades=0,
        mean_time_in_market=0.0,
        valid=False,
    )
