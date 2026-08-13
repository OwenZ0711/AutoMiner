"""Segment-blocked masked lagged-correlation scan (LEADLAG_DESIGN §2.4).

Estimator, for ordered pair (i, j), lag ℓ ∈ [1, L], X = x_std·M (NaN→0):

    S_xy[ℓ]  = Σ_seg  X[:-ℓ]ᵀ X[ℓ:]        (uncentered — x_std is standardized)
    S_x2m[ℓ] = Σ_seg  (X²)[:-ℓ]ᵀ M[ℓ:]     (leader variance on joint support)
    S_mx2[ℓ] = Σ_seg  M[:-ℓ]ᵀ (X²)[ℓ:]     (follower variance on joint support)
    n[ℓ]     = Σ_seg  M[:-ℓ]ᵀ M[ℓ:]        (joint-valid count)

    ρ[i,j,ℓ] = S_xy / sqrt(S_x2m · S_mx2)   (NaN where n < min_joint_bars)
    ρ[i,j,−ℓ] = ρ[j,i,ℓ]                     (negative lags by transpose)

Implementation — zero-pad instead of per-segment GEMM loops: insert ``L``
all-zero rows (X = X² = M = 0) at every session-segment boundary, then plain
shifted GEMMs over the whole padded block are EXACT: any (t, t+ℓ) pair that
would straddle a boundary has at least one factor inside the padding and
contributes zero to all four Gram products, with ``n`` staying exact.

GEMMs run in f32 (Accelerate); accumulation across year chunks in f64.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from futures_common.manifest import config_hash
from gp_cta.panel_io import PanelReader

_EPOCH = dt.date(1970, 1, 1)


@dataclass(frozen=True, slots=True)
class ScanConfig:
    max_lag: int = 60
    min_joint_bars: int = 20_000

    @property
    def hash(self) -> str:
        return config_hash({"max_lag": self.max_lag, "min_joint": self.min_joint_bars, "v": 1})


@dataclass(slots=True)
class GramSums:
    """f64 accumulators of the four Gram products, shape (L, N, N)."""

    s_xy: np.ndarray
    s_x2m: np.ndarray
    s_mx2: np.ndarray
    n: np.ndarray

    @classmethod
    def zeros(cls, max_lag: int, n_products: int) -> GramSums:
        shape = (max_lag, n_products, n_products)
        return cls(
            s_xy=np.zeros(shape),
            s_x2m=np.zeros(shape),
            s_mx2=np.zeros(shape),
            n=np.zeros(shape),
        )

    def add(self, other: GramSums) -> None:
        self.s_xy += other.s_xy
        self.s_x2m += other.s_x2m
        self.s_mx2 += other.s_mx2
        self.n += other.n

    def subtract(self, other: GramSums) -> None:
        self.s_xy -= other.s_xy
        self.s_x2m -= other.s_x2m
        self.s_mx2 -= other.s_mx2
        self.n -= other.n


@dataclass(frozen=True, slots=True)
class ScanResult:
    rho: np.ndarray  # (L, N, N) f64, NaN where n < min_joint_bars
    n: np.ndarray  # (L, N, N) int64
    products: tuple[str, ...]
    window_start: dt.date
    window_end: dt.date
    config_hash: str


def pad_panel(
    x: np.ndarray, mask: np.ndarray, spans: Sequence[tuple[int, int]], pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Insert `pad` zero rows between session segments. x: (T,N) f32, mask u8."""
    if not spans:
        return x.astype(np.float32), mask.astype(np.float32)
    n = x.shape[1]
    blocks_x: list[np.ndarray] = []
    blocks_m: list[np.ndarray] = []
    zeros = np.zeros((pad, n), dtype=np.float32)
    for k, (s, e) in enumerate(spans):
        if k > 0:
            blocks_x.append(zeros)
            blocks_m.append(zeros)
        blocks_x.append(x[s:e])
        blocks_m.append(mask[s:e])
    return np.vstack(blocks_x), np.vstack(blocks_m)


def gram_sums_for_block(x_masked: np.ndarray, mask: np.ndarray, lags: Sequence[int]) -> GramSums:
    """Four Gram products per lag over one padded block (f32 GEMM → f64)."""
    n_products = x_masked.shape[1]
    max_lag = max(lags)
    out = GramSums.zeros(max_lag, n_products)
    x2 = x_masked * x_masked
    for lag in lags:
        a, b = x_masked[:-lag], x_masked[lag:]
        m_a, m_b = mask[:-lag], mask[lag:]
        out.s_xy[lag - 1] = (a.T @ b).astype(np.float64)
        out.s_x2m[lag - 1] = (x2[:-lag].T @ m_b).astype(np.float64)
        out.s_mx2[lag - 1] = (m_a.T @ x2[lag:]).astype(np.float64)
        out.n[lag - 1] = (m_a.T @ m_b).astype(np.float64)
    return out


def prep_year_block(
    panel: PanelReader,
    year: int,
    window: tuple[dt.date, dt.date] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    """UNPADDED (X·M, M, trading_days, spans) f32 arrays for one year,
    optionally restricted to a trading-date window. Shared by the scan and
    the permutation null (which must shift within the same window)."""
    if window and (dt.date(year, 12, 31) < window[0] or dt.date(year, 1, 1) > window[1]):
        return None
    x = np.asarray(panel.feature("x_std", year), dtype=np.float32)
    m = (np.asarray(panel.mask("m_traded", year)) > 0) & np.isfinite(x)
    lim = np.asarray(panel.mask("m_limit", year)) > 0
    m &= ~lim  # limit-locked bars are not valid observations
    td = panel.trading_dates(year)
    spans = panel.segment_spans(year)
    if window:
        lo = (window[0] - _EPOCH).days
        hi = (window[1] - _EPOCH).days
        row_filter = (td >= lo) & (td <= hi)
        if not row_filter.any():
            return None
        if not row_filter.all():
            keep = np.flatnonzero(row_filter)
            spans = _filter_spans(spans, keep)
            x = x[keep]
            m = m[keep]
            td = td[keep]
    xm = np.where(m, x, 0.0).astype(np.float32)
    return xm, m.astype(np.float32), td, spans


def _filter_spans(spans: Sequence[tuple[int, int]], keep: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in spans:
        lo = int(np.searchsorted(keep, s))
        hi = int(np.searchsorted(keep, e))
        if hi > lo:
            out.append((lo, hi))
    return out


def scan_lagged_corr(
    panel: PanelReader,
    cfg: ScanConfig,
    *,
    window: tuple[dt.date, dt.date] | None = None,
    lags: Sequence[int] | None = None,
) -> ScanResult:
    """Full scan over the reader's years (optionally restricted to a window)."""
    lags = list(lags) if lags is not None else list(range(1, cfg.max_lag + 1))
    total = GramSums.zeros(cfg.max_lag, panel.n_products)
    w_start, w_end = window if window else (dt.date.min, dt.date.max)

    for year in panel.years:
        block = prep_year_block(panel, year, window)
        if block is None:
            continue
        xm_raw, m_raw, _td, spans = block
        xm, m = pad_panel(xm_raw, m_raw, spans, cfg.max_lag)
        total.add(gram_sums_for_block(xm, m, lags))

    return finalize(total, panel.products, cfg, w_start, w_end, lags)


def finalize(
    sums: GramSums,
    products: Sequence[str],
    cfg: ScanConfig,
    window_start: dt.date,
    window_end: dt.date,
    lags: Sequence[int],
) -> ScanResult:
    denom = np.sqrt(sums.s_x2m * sums.s_mx2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = sums.s_xy / denom
    n = sums.n.astype(np.int64)
    rho = np.where(n >= cfg.min_joint_bars, rho, np.nan)
    unused = np.setdiff1d(np.arange(1, cfg.max_lag + 1), np.asarray(lags))
    rho[unused - 1] = np.nan
    return ScanResult(
        rho=rho,
        n=n,
        products=tuple(products),
        window_start=window_start,
        window_end=window_end,
        config_hash=cfg.hash,
    )


def save_scan(result: ScanResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rho=result.rho, n=result.n)
    sidecar = {
        "products": list(result.products),
        "window_start": result.window_start.isoformat(),
        "window_end": result.window_end.isoformat(),
        "config_hash": result.config_hash,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


def monthly_windows(
    panel: PanelReader, *, window_months: int, step_months: int = 1
) -> Iterator[tuple[dt.date, dt.date]]:
    """Trailing windows [start, end] stepped monthly over the panel's span."""
    years = panel.years
    first = dt.date(years[0], 1, 1)
    last = dt.date(years[-1], 12, 31)
    anchor = dt.date(first.year + window_months // 12, first.month + window_months % 12, 1)
    while anchor <= last + dt.timedelta(days=31):
        end = min(anchor - dt.timedelta(days=1), last)
        start_month = (end.year * 12 + end.month - 1) - window_months
        start = dt.date(start_month // 12, start_month % 12 + 1, 1)
        yield (start, end)
        nxt = anchor.month + step_months
        anchor = dt.date(anchor.year + (nxt - 1) // 12, (nxt - 1) % 12 + 1, 1)


def rolling_scan(
    panel: PanelReader,
    cfg: ScanConfig,
    *,
    window_months: int = 24,
    step_months: int = 1,
    lags: Sequence[int] | None = None,
) -> Iterator[ScanResult]:
    """Monthly re-scans on a trailing window (for stability metrics)."""
    for start, end in monthly_windows(panel, window_months=window_months, step_months=step_months):
        result = scan_lagged_corr(panel, cfg, window=(start, end), lags=lags)
        if int(np.nanmax(result.n)) > 0:
            yield result
        else:
            logger.debug("rolling scan window [{}, {}] empty — skipped", start, end)
