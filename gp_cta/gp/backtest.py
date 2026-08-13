"""Long/short futures backtest — THE execution-timing chokepoint.

LOOK-AHEAD SAFETY §2: every trade in the system is timed HERE and only here.

Semantics per product, on its own COMPRESSED traded-bar sequence (positions
can only change when the product actually prints a bar):

    signal s[t]      known at bar t CLOSE
    thresholds       hi[t], lo[t] = rolling quantiles of s over `lookback`
                     bars ending at t−1 (shifted one bar — never sees s[t])
    desired d[t]     +1 if s[t] > hi[t]; −1 if s[t] < lo[t]; else hold;
                     forced flat when the position age exceeds max_hold bars
    fill             pos[t] = d[t−1] executed at bar t OPEN, unless bar t is
                     limit-locked (fill blocked → keep previous position);
                     forced flat at hard-segment starts (the 2025 hole)
    PnL              gross[t] = pos[t] · (open_adj[t+1] / open_adj[t] − 1)
    costs            Δ[t] = |pos[t] − pos[t−1]| sides at bar t's open:
                     Δ · (fee_bps·1e−4 + slippage_ticks · tick / open_RAW[t]);
                     holding through a roll bar adds one full round-trip
    daily            net pnl summed into trading-day buckets

Costs normalize slippage by the RAW open (tick sizes are raw-scale) while
returns use adjusted opens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from futures_common.config import CostsConfig
from futures_common.numba_compat import HAVE_NUMBA, njit_kernel
from gp_cta.panel_io import PanelReader


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    q_hi: float = 0.70
    q_lo: float = 0.30
    lookback: int = 1200
    min_periods: int = 300
    max_hold_bars: int = 345  # ~1 trading day of a night→23:00 product


@dataclass(frozen=True, slots=True)
class ProductCostArrays:
    fee_bps: float
    slippage_ticks: float
    tick_size: float


@dataclass(slots=True)
class BacktestResult:
    daily: pl.DataFrame  # trading_date + one net-return column per product
    per_product: pl.DataFrame  # product, trades, turnover_per_day, time_in_market, bars
    products: tuple[str, ...]


class UndeclaredCostsError(RuntimeError):
    """Backtests refuse products without declared-or-verified cost specs."""


def product_costs(costs: CostsConfig, product: str) -> ProductCostArrays:
    spec = costs.for_product(product)
    if not spec.declared or spec.tick_size is None:
        raise UndeclaredCostsError(
            f"{product}: no declared costs/tick in configs/costs.yaml — refuse to backtest"
        )
    return ProductCostArrays(
        fee_bps=spec.fee_bps_per_side,
        slippage_ticks=float(spec.slippage_ticks),
        tick_size=spec.tick_size,
    )


# ----------------------------------------------------------------- thresholds


def rolling_quantiles(
    s: np.ndarray, lookback: int, min_periods: int, q_hi: float, q_lo: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling quantiles of the window ENDING AT t−1 (shifted one bar)."""
    ser = pd.Series(s).shift(1).rolling(lookback, min_periods=min_periods)
    return (
        ser.quantile(q_hi).to_numpy(),
        ser.quantile(q_lo).to_numpy(),
    )


# ------------------------------------------------------------- position kernel


@njit_kernel(cache=True)
def _position_kernel(
    s: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    limit: np.ndarray,
    seg: np.ndarray,
    max_hold: int,
) -> np.ndarray:  # pragma: no cover - exercised via wrapper
    n = s.size
    pos = np.zeros(n, dtype=np.float64)
    d_prev = 0.0
    pos_prev = 0.0
    hold_age = 0
    for t in range(n):
        if t == 0 or seg[t] != seg[t - 1]:
            p = 0.0
            d_prev = 0.0
            hold_age = 0
        elif limit[t] > 0:
            p = pos_prev  # fill blocked; re-evaluated next bar
        else:
            p = d_prev
        if p != pos_prev or p == 0.0:
            hold_age = 0
        else:
            hold_age += 1
        pos[t] = p

        # decide desired for the NEXT bar from information at bar t close
        if np.isnan(hi[t]) or np.isnan(lo[t]):
            d = p
        elif s[t] > hi[t]:
            d = 1.0
        elif s[t] < lo[t]:
            d = -1.0
        else:
            d = p
        if hold_age >= max_hold and d == p:
            d = 0.0
        d_prev = d
        pos_prev = p
    return pos


def _position_python(
    s: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    limit: np.ndarray,
    seg: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    """Reference twin of the numba kernel (identical semantics, used in tests)."""
    n = s.size
    pos = np.zeros(n)
    d_prev = pos_prev = 0.0
    hold_age = 0
    for t in range(n):
        if t == 0 or seg[t] != seg[t - 1]:
            p, d_prev, hold_age = 0.0, 0.0, 0
        elif limit[t] > 0:
            p = pos_prev
        else:
            p = d_prev
        hold_age = 0 if (p != pos_prev or p == 0.0) else hold_age + 1
        pos[t] = p
        if np.isnan(hi[t]) or np.isnan(lo[t]):
            d = p
        elif s[t] > hi[t]:
            d = 1.0
        elif s[t] < lo[t]:
            d = -1.0
        else:
            d = p
        if hold_age >= max_hold and d == p:
            d = 0.0
        d_prev, pos_prev = d, p
    return pos


def positions_from_signal(
    s: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    limit: np.ndarray,
    seg: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    fn = _position_kernel if HAVE_NUMBA else _position_python
    return fn(
        np.ascontiguousarray(s, dtype=np.float64),
        np.ascontiguousarray(hi, dtype=np.float64),
        np.ascontiguousarray(lo, dtype=np.float64),
        np.ascontiguousarray(limit, dtype=np.float64),
        np.ascontiguousarray(seg, dtype=np.float64),
        max_hold,
    )


# ------------------------------------------------------------------- backtest


def run_backtest(
    signal: np.ndarray,  # (T, N) on the GRID
    panel: PanelReader,
    costs: Mapping[str, ProductCostArrays],
    cfg: BacktestConfig,
) -> BacktestResult:
    products = panel.products
    open_adj = np.vstack([np.asarray(panel.channel("open_adj", y)) for y in panel.years])
    open_raw = np.vstack([np.asarray(panel.channel("open_raw", y)) for y in panel.years])
    m_traded = np.vstack([np.asarray(panel.mask("m_traded", y)) for y in panel.years]) > 0
    m_limit = np.vstack([np.asarray(panel.mask("m_limit", y)) for y in panel.years]) > 0
    roll = np.vstack([np.asarray(panel.mask("roll_flag", y)) for y in panel.years]) > 0
    td = np.concatenate([panel.trading_dates(y) for y in panel.years])
    seg_macro = np.concatenate([panel.index(y)["seg_macro"].to_numpy() for y in panel.years])

    day_values = np.unique(td)
    day_of_row = np.searchsorted(day_values, td)

    daily = np.zeros((day_values.size, len(products)))
    pp_rows: list[dict[str, object]] = []

    for i, product in enumerate(products):
        rows = np.flatnonzero(m_traded[:, i] & np.isfinite(open_adj[:, i]))
        if rows.size < cfg.min_periods + 2:
            pp_rows.append(
                {
                    "product": product,
                    "trades": 0,
                    "turnover_per_day": 0.0,
                    "time_in_market": 0.0,
                    "bars": int(rows.size),
                }
            )
            continue
        s = signal[rows, i].astype(np.float64)
        oa = open_adj[rows, i].astype(np.float64)
        orw = open_raw[rows, i].astype(np.float64)
        lim = m_limit[rows, i].astype(np.float64)
        rl = roll[rows, i]
        sg = seg_macro[rows].astype(np.float64)
        dor = day_of_row[rows]

        hi, lo = rolling_quantiles(s, cfg.lookback, cfg.min_periods, cfg.q_hi, cfg.q_lo)
        pos = positions_from_signal(s, hi, lo, lim, sg, cfg.max_hold_bars)

        r_next = np.zeros(rows.size)
        same_seg = sg[1:] == sg[:-1]
        r_next[:-1] = np.where(same_seg, oa[1:] / oa[:-1] - 1.0, 0.0)
        gross = pos * r_next

        delta = np.abs(np.diff(np.concatenate([[0.0], pos])))
        per_side = costs[product].fee_bps * 1e-4 + (
            costs[product].slippage_ticks * costs[product].tick_size / orw
        )
        cost = delta * per_side
        cost += np.where(rl & (pos != 0), 2.0 * per_side, 0.0)  # roll round-trip
        net = gross - cost

        np.add.at(daily[:, i], dor, net)
        pp_rows.append(
            {
                "product": product,
                "trades": int((delta > 0).sum()),
                "turnover_per_day": float(delta.sum() / max(day_values.size, 1)),
                "time_in_market": float(np.mean(pos != 0)),
                "bars": int(rows.size),
            }
        )

    epoch = np.datetime64("1970-01-01")
    daily_df = pl.DataFrame(
        {
            "trading_date": pl.Series(
                (epoch + day_values.astype("timedelta64[D]")).astype("datetime64[D]")
            ).cast(pl.Date),
            **{p: daily[:, i] for i, p in enumerate(products)},
        }
    )
    return BacktestResult(
        daily=daily_df,
        per_product=pl.DataFrame(pp_rows),
        products=products,
    )
