from __future__ import annotations

import polars as pl

from gp_cta.evaluator import EvaluationConfig, FormulaEvaluator, split_by_date
from gp_cta.features import build_features
from gp_cta.gp import GPConfig, GPEngine


def _synthetic_panel() -> pl.DataFrame:
    dates = pl.date_range(
        start=pl.date(2020, 1, 1),
        end=pl.date(2020, 7, 31),
        interval="1d",
        eager=True,
    )
    rows = []
    for sym, drift in (("SH600519", 0.10), ("SZ000001", -0.03)):
        price = 100.0
        for i, date in enumerate(dates):
            price *= 1.0 + drift / 252.0 + (0.01 if i % 17 == 0 else 0.0)
            rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000_000.0 + i,
                    "turnover": price * (1_000_000.0 + i),
                    "vwap": price,
                    "return_pct": 0.0,
                    "turnover_rate": 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_gp_engine_is_reproducible_with_fixed_seed() -> None:
    panel = build_features(_synthetic_panel())
    train = split_by_date(panel, train_end="2020-05-31", val_end="2020-06-30")["train"]
    evaluator = FormulaEvaluator(
        EvaluationConfig(threshold_lookback=10, min_periods=5, min_trades=0)
    )
    config = GPConfig(
        population_size=8,
        generations=2,
        max_depth=3,
        elite_size=2,
        seed=7,
    )

    first = GPEngine(evaluator, config=config).evolve(train)
    second = GPEngine(evaluator, config=config).evolve(train)

    assert first[0].result.formula == second[0].result.formula
    assert first[0].result.fitness == second[0].result.fitness
    assert len(first) == 8
