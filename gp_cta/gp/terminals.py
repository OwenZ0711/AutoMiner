"""Evaluation context + cross-asset terminal resolution.

``EvalContext`` holds the (T, N) terminal arrays for the requested years
(concatenated) plus the hard-segment boundaries (the 2025 hole). ``evaluate``
walks the AST over it; every node output passes ``_clean`` (NaN/±inf → 0,
clip ±1e6) — the baseline's tested safety contract. Trade gating happens in
the backtest via masks, never via NaN propagation.

CROSS-ASSET TERMINALS (look-ahead safety §4): resolved month by month through
the EdgeStore — for month m only edges with ``asof_date < first trading day
of m`` are visible, leaders ranked by |a_stat| among validated rows, lag
clamped to the validated ℓ*. GP never searches lags.

    lead_ret_r[t, j]    rolling sum of leader_r(j)'s x_std over its ℓ* bars
                        ending at t (leader data through bar t close only)
    lead_oi_chg_r[t, j] leader's oi_chg_1d at t
    lead_stale_r[t, j]  grid rows since the leader last printed a traded bar
    ll_strength_r[t, j] the edge's strength, constant within the month

Followers with no validated rank-r edge in month m get zeros (signal dead →
no trade there; validity gates handle degenerate formulas).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from loguru import logger

from gp_cta.gp import ops
from gp_cta.gp.expressions import CROSS_TERMINALS, Expr
from gp_cta.leadlag.edges import EdgeStore
from gp_cta.panel_io import PanelReader

EPS = 1e-12
MAX_ABS_VALUE = 1e6
_EPOCH = dt.date(1970, 1, 1)

PANEL_FEATURES = (
    "x_std",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "ret_1d",
    "vol_ewma",
    "z_60",
    "z_240",
    "z_1200",
    "oi_chg_1d",
    "vol_ratio",
    "range_1",
    "range_15",
    "range_60",
)


def _clean(values: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -MAX_ABS_VALUE, MAX_ABS_VALUE)


@dataclass(slots=True)
class EvalContext:
    arrays: dict[str, np.ndarray]  # terminal name → (T, N) f32 (raw, NaN allowed)
    boundaries: list[int]  # hard-segment start rows (seg_macro changes)
    n_rows: int
    n_products: int

    def terminal(self, name: str) -> np.ndarray:
        arr = self.arrays.get(name)
        if arr is None:
            return np.zeros((self.n_rows, self.n_products), dtype=np.float32)
        return arr


def build_context(
    panel: PanelReader,
    *,
    edges: EdgeStore | None = None,
    top_k: int = 3,
    q_max: float = 0.10,
    min_stability: float = 0.0,
    max_staleness_days: int | None = 45,
    shift_bars: int = 0,
) -> EvalContext:
    """Concatenate panel years into terminal arrays (+ cross terminals).

    ``shift_bars`` > 0 is the +1-bar KILL-SWITCH (look-ahead gate (b)): every
    terminal is delayed by that many rows before evaluation.
    """
    feats: dict[str, list[np.ndarray]] = {name: [] for name in PANEL_FEATURES}
    closes: list[np.ndarray] = []
    segs: list[np.ndarray] = []
    for year in panel.years:
        for name in PANEL_FEATURES:
            feats[name].append(np.asarray(panel.feature(name, year), dtype=np.float32))
        closes.append(np.asarray(panel.channel("close_adj", year), dtype=np.float32))
        segs.append(panel.index(year)["seg_macro"].to_numpy())

    arrays = {name: np.vstack(chunks) for name, chunks in feats.items()}
    arrays["close_adj"] = np.vstack(closes)
    seg = np.concatenate(segs)
    boundaries = [0, *(np.flatnonzero(np.diff(seg)) + 1).tolist()]
    n_rows, n_products = arrays["close_adj"].shape

    if edges is not None:
        arrays.update(
            resolve_cross_terminals(
                panel,
                edges,
                x_std=arrays["x_std"],
                oi_chg=arrays["oi_chg_1d"],
                top_k=top_k,
                q_max=q_max,
                min_stability=min_stability,
                max_staleness_days=max_staleness_days,
            )
        )

    if shift_bars > 0:
        for name, arr in arrays.items():
            shifted = np.full_like(arr, np.nan)
            shifted[shift_bars:] = arr[:-shift_bars]
            arrays[name] = shifted

    return EvalContext(arrays=arrays, boundaries=boundaries, n_rows=n_rows, n_products=n_products)


def resolve_cross_terminals(
    panel: PanelReader,
    edges: EdgeStore,
    *,
    x_std: np.ndarray,
    oi_chg: np.ndarray,
    top_k: int,
    q_max: float,
    min_stability: float,
    max_staleness_days: int | None,
) -> dict[str, np.ndarray]:
    products = panel.products
    n = len(products)
    td = np.concatenate([panel.trading_dates(y) for y in panel.years])
    traded = np.vstack([np.asarray(panel.mask("m_traded", y)) for y in panel.years]).astype(bool)
    t_rows = td.size

    out = {
        f"{name}_{r}": np.zeros((t_rows, n), dtype=np.float32)
        for name in CROSS_TERMINALS
        for r in (1, 2, 3)
        if r <= top_k
    }

    # staleness per product: rows since last traded bar
    stale = np.zeros((t_rows, n), dtype=np.float32)
    row_idx = np.arange(t_rows)
    for i in range(n):
        last = np.where(traded[:, i], row_idx, -1)
        np.maximum.accumulate(last, out=last)
        stale[:, i] = np.where(last >= 0, row_idx - last, row_idx).astype(np.float32)

    # month blocks over the concatenated grid (months since 1970-01)
    dates = td.astype(np.int64)
    months = dates.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    month_changes = np.flatnonzero(np.diff(months)) + 1
    starts = np.concatenate([[0], month_changes])
    ends = np.concatenate([month_changes, [t_rows]])

    lead_ret_cache: dict[tuple[int, int], np.ndarray] = {}

    def lead_ret_col(i: int, lag: int) -> np.ndarray:
        key = (i, lag)
        if key not in lead_ret_cache:
            x = np.nan_to_num(x_std[:, i].astype(np.float64), nan=0.0)
            cs = np.concatenate([[0.0], np.cumsum(x)])
            col = np.zeros(t_rows, dtype=np.float32)
            col[lag - 1 :] = (cs[lag:] - cs[:-lag]).astype(np.float32)
            lead_ret_cache[key] = col
        return lead_ret_cache[key]

    resolved_months = 0
    for s, e in zip(starts, ends, strict=True):
        month_first = _EPOCH + dt.timedelta(days=int(dates[s]))
        any_edge = False
        for j, follower in enumerate(products):
            leaders = edges.leaders_for(
                follower,
                month_first,
                top_k=top_k,
                q_max=q_max,
                min_stability=min_stability,
                max_staleness_days=max_staleness_days,
            )
            for r, edge in enumerate(leaders, start=1):
                if edge.leader not in products:
                    continue
                i = products.index(edge.leader)
                any_edge = True
                out[f"lead_ret_{r}"][s:e, j] = lead_ret_col(i, max(edge.lag_bars, 1))[s:e]
                out[f"lead_oi_chg_{r}"][s:e, j] = np.nan_to_num(oi_chg[s:e, i], nan=0.0)
                out[f"lead_stale_{r}"][s:e, j] = stale[s:e, i]
                out[f"ll_strength_{r}"][s:e, j] = edge.strength
        resolved_months += int(any_edge)
    logger.info(
        "cross terminals: {}/{} months have at least one validated edge",
        resolved_months,
        starts.size,
    )
    return out


# ------------------------------------------------------------------ evaluation


def evaluate(e: Expr, ctx: EvalContext) -> np.ndarray:
    """Vectorized whole-panel evaluation; every node output is _clean-ed."""
    op = e.op
    if op == "feature":
        return _clean(ctx.terminal(str(e.value)))
    if op == "const":
        assert e.value is not None
        return np.full((ctx.n_rows, ctx.n_products), float(e.value), dtype=np.float32)
    if op in CROSS_TERMINALS:
        assert e.value is not None
        return _clean(ctx.terminal(f"{op}_{int(e.value)}"))

    args = [evaluate(c, ctx) for c in e.children]
    if op == "neg":
        return _clean(-args[0])
    if op == "abs":
        return _clean(np.abs(args[0]))
    if op == "log1p":
        return _clean(np.log1p(np.abs(args[0])))
    if op == "sign":
        return _clean(np.sign(args[0]))
    if op == "ref":
        return _clean(ops.apply_segmented(ops.ref, args[0], ctx.boundaries, int(e.value)))  # type: ignore[arg-type]
    if op == "add":
        return _clean(args[0] + args[1])
    if op == "sub":
        return _clean(args[0] - args[1])
    if op == "mul":
        return _clean(args[0] * args[1])
    if op == "div":
        denom = np.where(np.abs(args[1]) < EPS, np.nan, args[1])
        return _clean(args[0] / denom)
    if op == "min2":
        return _clean(np.minimum(args[0], args[1]))
    if op == "max2":
        return _clean(np.maximum(args[0], args[1]))
    if op == "gt":
        return _clean((args[0] > args[1]).astype(np.float32))
    if op == "lt":
        return _clean((args[0] < args[1]).astype(np.float32))
    if op == "if_then_else":
        return _clean(np.where(args[0] != 0, args[1], args[2]))

    window = int(e.value)  # type: ignore[arg-type]
    ts_fn = {
        "ts_mean": ops.ts_mean,
        "ts_std": ops.ts_std,
        "ts_min": ops.ts_min,
        "ts_max": ops.ts_max,
        "ts_delta": ops.ts_delta,
        "ts_sum": ops.ts_sum,
        "ts_rank": ops.ts_rank,
        "ts_zscore": ops.ts_zscore,
        "ema": ops.ema,
        "decay_linear": ops.decay_linear,
    }.get(op)
    if ts_fn is not None:
        return _clean(ops.apply_segmented(ts_fn, args[0], ctx.boundaries, window))
    if op == "ts_corr":
        return _clean(_corr_segmented(args[0], args[1], ctx.boundaries, window))
    raise ValueError(f"unhandled op {op!r}")


def _corr_segmented(x: np.ndarray, y: np.ndarray, boundaries: list[int], w: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    bounds = [*boundaries, x.shape[0]]
    for s, e in pairwise(bounds):
        if e > s:
            out[s:e] = ops.ts_corr(x[s:e], y[s:e], w)
    return out
