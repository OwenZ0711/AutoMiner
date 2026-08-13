"""Vectorized rolling kernels for GP evaluation on (T, N) panels.

All windows are TRAILING and inclusive, min_periods = window (NaN before) —
the same semantics as the data band's feature helpers, pinned by tests
against pandas references. Cumsum algebra runs in float64 internally
(windowed differences of f32 cumsums would cancel catastrophically); outputs
are float32.

Hard boundaries (the 2025 data hole, via ``seg_macro``) reset every window:
``apply_segmented`` slices the panel at boundary rows and applies the kernel
per segment. Intra-day session breaks do NOT reset GP windows (features are
bar-clock trailing by spec §L2b).

numba accelerates the sequential kernels (ema, rank); pandas provides the
reference/fallback implementations (see futures_common.numba_compat).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from futures_common.numba_compat import HAVE_NUMBA, njit_kernel


def apply_segmented(
    fn: Callable[..., np.ndarray],
    x: np.ndarray,
    boundaries: Sequence[int],
    *args: object,
) -> np.ndarray:
    """Apply `fn(x_segment, *args)` per hard segment; boundaries = start rows."""
    if len(boundaries) <= 1:
        return fn(x, *args)
    out = np.empty_like(x, dtype=np.float32)
    bounds = [*boundaries, x.shape[0]]
    for s, e in pairwise(bounds):
        if e > s:
            out[s:e] = fn(x[s:e], *args)
    return out


def _cs(x64: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, x64.shape[1])), np.cumsum(x64, axis=0)])


def ts_sum(x: np.ndarray, w: int) -> np.ndarray:
    x64 = x.astype(np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if x.shape[0] >= w:
        cs = _cs(x64)
        out[w - 1 :] = (cs[w:] - cs[:-w]).astype(np.float32)
    return out


def ts_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = ts_sum(x, w)
    np.divide(out, w, out=out)
    return out


def ts_std(x: np.ndarray, w: int) -> np.ndarray:
    x64 = x.astype(np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if x.shape[0] >= w:
        cs = _cs(x64)
        cs2 = _cs(x64 * x64)
        s = cs[w:] - cs[:-w]
        s2 = cs2[w:] - cs2[:-w]
        var = np.maximum(s2 / w - (s / w) ** 2, 0.0)
        out[w - 1 :] = np.sqrt(var).astype(np.float32)
    return out


def ts_zscore(x: np.ndarray, w: int) -> np.ndarray:
    mean = ts_mean(x, w).astype(np.float64)
    std = ts_std(x, w).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x.astype(np.float64) - mean) / np.where(std > 0, std, np.nan)
    return z.astype(np.float32)


def ts_delta(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if x.shape[0] > w:
        out[w:] = x[w:] - x[:-w]
    return out


def ts_min(x: np.ndarray, w: int) -> np.ndarray:
    return _pd_roll(x, w, "min")


def ts_max(x: np.ndarray, w: int) -> np.ndarray:
    return _pd_roll(x, w, "max")


def _pd_roll(x: np.ndarray, w: int, how: str) -> np.ndarray:
    r = pd.DataFrame(x).rolling(w, min_periods=w)
    out: np.ndarray = getattr(r, how)().to_numpy().astype(np.float32)
    return out


def ts_rank(x: np.ndarray, w: int) -> np.ndarray:
    """Percentile rank (0..1] of the current value within the window."""
    if HAVE_NUMBA:
        return _rank_kernel(np.ascontiguousarray(x, dtype=np.float64), w).astype(np.float32)
    out: np.ndarray = (
        pd.DataFrame(x).rolling(w, min_periods=w).rank(pct=True).to_numpy().astype(np.float32)
    )
    return out


@njit_kernel(cache=True, parallel=False)
def _rank_kernel(x: np.ndarray, w: int) -> np.ndarray:  # pragma: no cover
    t_rows, n = x.shape
    out = np.full((t_rows, n), np.nan)
    for j in range(n):
        for t in range(w - 1, t_rows):
            v = x[t, j]
            le = 0
            for k in range(t - w + 1, t + 1):
                if x[k, j] <= v:
                    le += 1
            out[t, j] = le / w
    return out


def ema(x: np.ndarray, w: int) -> np.ndarray:
    """EWMA with halflife = w bars (adjust=False semantics)."""
    alpha = 1.0 - float(np.exp(-np.log(2.0) / max(w, 1)))
    if HAVE_NUMBA:
        return _ema_kernel(np.ascontiguousarray(x, dtype=np.float64), alpha).astype(np.float32)
    out: np.ndarray = (
        pd.DataFrame(x).ewm(alpha=alpha, adjust=False).mean().to_numpy().astype(np.float32)
    )
    return out


@njit_kernel(cache=True)
def _ema_kernel(x: np.ndarray, alpha: float) -> np.ndarray:  # pragma: no cover
    t_rows, n = x.shape
    out = np.empty((t_rows, n))
    for j in range(n):
        state = x[0, j]
        out[0, j] = state
        for t in range(1, t_rows):
            state = alpha * x[t, j] + (1.0 - alpha) * state
            out[t, j] = state
    return out


def decay_linear(x: np.ndarray, w: int) -> np.ndarray:
    """Linearly-decayed weighted mean: newest weight w, oldest 1 (normalized).

    Pure cumsum algebra: with absolute index k, the window numerator is
    Σ (k − t + w)·x_k = Σ k·x_k − (t − w)·Σ x_k.
    """
    t_rows = x.shape[0]
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if t_rows < w:
        return out
    x64 = x.astype(np.float64)
    k = np.arange(t_rows, dtype=np.float64)[:, None]
    cs = _cs(x64)
    ks = _cs(x64 * k)
    a = cs[w:] - cs[:-w]
    b = ks[w:] - ks[:-w]
    t_idx = np.arange(w - 1, t_rows, dtype=np.float64)[:, None]
    numer = b - (t_idx - w) * a
    out[w - 1 :] = (numer / (w * (w + 1) / 2.0)).astype(np.float32)
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if x.shape[0] < w:
        return out
    csx, csy = _cs(x64), _cs(y64)
    csxy = _cs(x64 * y64)
    csx2, csy2 = _cs(x64 * x64), _cs(y64 * y64)
    sx = csx[w:] - csx[:-w]
    sy = csy[w:] - csy[:-w]
    sxy = csxy[w:] - csxy[:-w]
    sx2 = csx2[w:] - csx2[:-w]
    sy2 = csy2[w:] - csy2[:-w]
    cov = sxy / w - (sx / w) * (sy / w)
    vx = np.maximum(sx2 / w - (sx / w) ** 2, 0.0)
    vy = np.maximum(sy2 / w - (sy / w) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(vx * vy)
    out[w - 1 :] = corr.astype(np.float32)
    return out


def ref(x: np.ndarray, k: int) -> np.ndarray:
    """Value k bars ago."""
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if x.shape[0] > k:
        out[k:] = x[:-k]
    return out
