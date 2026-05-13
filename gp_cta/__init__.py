"""Genetic-programming CTA baseline.

This package is intentionally small: it is a runnable baseline/demo that
reuses the repo's data layer without changing Pipeline A's STL+MCTS plan.
"""

from gp_cta.backtest import BacktestConfig, BacktestReport, run_long_flat_backtest
from gp_cta.evaluator import EvaluationConfig, FormulaEvaluator
from gp_cta.expressions import Expr
from gp_cta.gp import GPConfig, GPEngine

__all__ = [
    "BacktestConfig",
    "BacktestReport",
    "EvaluationConfig",
    "Expr",
    "FormulaEvaluator",
    "GPConfig",
    "GPEngine",
    "run_long_flat_backtest",
]
