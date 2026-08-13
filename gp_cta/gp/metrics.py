"""Full evaluation-metrics suite over daily net returns.

Everything the later admission gates (band 3) need is computed HERE and
stored with every trial: annualized Sharpe, Sortino, CAGR-style annual
return, max drawdown (return units, positions sized at unit gross notional),
Calmar, hit rate, profit factor, turnover, trades/year, time-in-market,
return skewness/kurtosis, PSR (probability the true Sharpe exceeds 0 given
non-normality, Bailey & López de Prado 2012), and the DSR inputs (moments +
sample size + the trial count lives in the ledger).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

TRADING_DAYS = 252


def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class ProductMetrics:
    product: str
    ann_return: float
    ann_sharpe: float
    ann_sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    profit_factor: float
    skewness: float
    kurtosis: float  # excess
    psr: float  # P(true Sharpe > 0)
    n_days: int
    trades: int
    trades_per_year: float
    turnover_per_day: float
    time_in_market: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def product_metrics(
    daily: np.ndarray,
    product: str,
    *,
    trades: int,
    turnover_per_day: float,
    time_in_market: float,
) -> ProductMetrics:
    r = daily[np.isfinite(daily)]
    n = r.size
    if n < 5 or np.all(r == 0):
        return ProductMetrics(
            product=product,
            ann_return=0.0,
            ann_sharpe=0.0,
            ann_sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            hit_rate=0.0,
            profit_factor=0.0,
            skewness=0.0,
            kurtosis=0.0,
            psr=0.5,
            n_days=n,
            trades=trades,
            trades_per_year=0.0,
            turnover_per_day=turnover_per_day,
            time_in_market=time_in_market,
        )
    mean, std = float(np.mean(r)), float(np.std(r, ddof=0))
    sharpe = 0.0 if std <= 1e-12 else mean / std * np.sqrt(TRADING_DAYS)
    downside = float(np.std(np.minimum(r, 0.0), ddof=0))
    sortino = 0.0 if downside <= 1e-12 else mean / downside * np.sqrt(TRADING_DAYS)
    equity = np.cumsum(r)  # unit-gross-notional return units
    dd = float(np.max(np.maximum.accumulate(equity) - equity))
    ann_ret = mean * TRADING_DAYS
    calmar = 0.0 if dd <= 1e-12 else ann_ret / dd
    active = r[r != 0]
    hit = float(np.mean(active > 0)) if active.size else 0.0
    gains = float(np.sum(r[r > 0]))
    losses = float(-np.sum(r[r < 0]))
    pf = gains / losses if losses > 1e-12 else (float("inf") if gains > 0 else 0.0)
    if std > 1e-12:
        z = (r - mean) / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)
    else:
        skew = kurt = 0.0
    psr = probabilistic_sharpe(sharpe / np.sqrt(TRADING_DAYS), 0.0, n, skew, kurt)
    return ProductMetrics(
        product=product,
        ann_return=ann_ret,
        ann_sharpe=sharpe,
        ann_sortino=sortino,
        max_drawdown=dd,
        calmar=calmar,
        hit_rate=hit,
        profit_factor=pf if np.isfinite(pf) else 999.0,
        skewness=skew,
        kurtosis=kurt,
        psr=psr,
        n_days=n,
        trades=trades,
        trades_per_year=trades / max(n / TRADING_DAYS, 1e-9),
        turnover_per_day=turnover_per_day,
        time_in_market=time_in_market,
    )


def probabilistic_sharpe(
    sr_daily: float, benchmark_daily: float, n: int, skew: float, kurt_excess: float
) -> float:
    """PSR (Bailey & López de Prado): P(true SR > benchmark), non-normal aware."""
    if n <= 1:
        return 0.5
    denom = np.sqrt(
        max(1.0 - skew * sr_daily + (kurt_excess + 2.0) / 4.0 * sr_daily**2, 1e-12) / (n - 1)
    )
    return float(norm_cdf((sr_daily - benchmark_daily) / denom))


def portfolio_metrics(daily_df: pl.DataFrame, products: tuple[str, ...]) -> dict[str, Any]:
    """Equal-weight portfolio metrics across products (informational)."""
    mat = daily_df.select(list(products)).to_numpy()
    port = np.nanmean(mat, axis=1)
    pm = product_metrics(port, "PORTFOLIO", trades=0, turnover_per_day=0.0, time_in_market=0.0)
    return {
        "ann_return": pm.ann_return,
        "ann_sharpe": pm.ann_sharpe,
        "ann_sortino": pm.ann_sortino,
        "max_drawdown": pm.max_drawdown,
        "psr": pm.psr,
        "n_days": pm.n_days,
    }
