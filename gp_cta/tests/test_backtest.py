from __future__ import annotations

import numpy as np
import pandas as pd

from gp_cta.backtest import BacktestConfig, positions_from_signal, run_long_flat_backtest


def test_positions_from_signal_uses_lagged_threshold() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0])
    config = BacktestConfig(
        threshold_lookback=2,
        threshold_quantile=0.5,
        min_periods=2,
        execution_lag=0,
    )

    positions = positions_from_signal(signal, config)

    assert positions.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]


def test_positions_default_to_next_bar_execution() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0])
    config = BacktestConfig(threshold_lookback=2, threshold_quantile=0.5, min_periods=2)

    positions = positions_from_signal(signal, config)

    assert positions.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_backtest_never_long_has_zero_return() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    signal = np.zeros(5)

    report = run_long_flat_backtest(
        dates=list(dates),
        close=close,
        signal=signal,
        config=BacktestConfig(threshold_lookback=2, min_periods=2),
    )

    assert report.trade_count == 0
    assert report.total_return == 0.0
    assert report.sharpe == 0.0
