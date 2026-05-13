"""Long/flat CTA backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 30.0
    threshold_lookback: int = 60
    threshold_quantile: float = 0.5
    min_periods: int = 20
    execution_lag: int = 1


@dataclass(frozen=True)
class BacktestReport:
    sharpe: float
    cagr: float
    max_drawdown: float
    daily_turnover: float
    annual_turnover: float
    trade_count: int
    time_in_market: float
    total_return: float
    daily_returns: pd.Series
    equity_curve: pd.Series
    positions: pd.Series


def positions_from_signal(signal: pd.Series, config: BacktestConfig) -> pd.Series:
    """Convert a continuous signal to long/flat positions without future data."""
    if config.execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    threshold = (
        signal.shift(1)
        .rolling(
            window=config.threshold_lookback,
            min_periods=min(config.min_periods, config.threshold_lookback),
        )
        .quantile(config.threshold_quantile)
    )
    raw_position = (signal > threshold).astype(float)
    raw_position = raw_position.where(threshold.notna(), 0.0)
    position = raw_position.shift(config.execution_lag) if config.execution_lag else raw_position
    return position.fillna(0.0)


def run_long_flat_backtest(
    dates: list,
    close: np.ndarray,
    signal: np.ndarray,
    config: BacktestConfig | None = None,
) -> BacktestReport:
    """Backtest position_t held over close_t -> close_{t+1}."""
    cfg = config or BacktestConfig()
    idx = pd.to_datetime(dates)
    close_s = pd.Series(close, index=idx, dtype="float64")
    signal_s = pd.Series(signal, index=idx, dtype="float64").replace(
        [np.inf, -np.inf], np.nan
    )
    signal_s = signal_s.fillna(0.0)

    positions = positions_from_signal(signal_s, cfg)
    future_ret = close_s.shift(-1) / close_s - 1.0
    future_ret = future_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    turnover = positions.diff().abs().fillna(positions.abs())
    cost = turnover * (cfg.cost_bps / 10_000.0)
    returns = positions * future_ret - cost
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + returns).cumprod()

    std = float(returns.std(ddof=0))
    sharpe = 0.0 if std <= 1e-12 else float(returns.mean() / std * np.sqrt(TRADING_DAYS))
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    years = max(len(returns) / TRADING_DAYS, 1e-12)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(abs(drawdown.min())) if len(drawdown) else 0.0
    daily_turnover = float(turnover.mean()) if len(turnover) else 0.0
    trade_count = int((positions.diff().fillna(positions) > 0).sum())
    time_in_market = float(positions.mean()) if len(positions) else 0.0

    return BacktestReport(
        sharpe=sharpe,
        cagr=cagr,
        max_drawdown=max_drawdown,
        daily_turnover=daily_turnover,
        annual_turnover=daily_turnover * TRADING_DAYS,
        trade_count=trade_count,
        time_in_market=time_in_market,
        total_return=total_return,
        daily_returns=returns,
        equity_curve=equity,
        positions=positions,
    )
