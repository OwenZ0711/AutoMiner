"""Statistical validation: asymmetry stats, permutation nulls, FDR, stability.

THE RIGHT NULL (LEADLAG_DESIGN §3.1): H0 is "no lead-lag GIVEN contemporaneous
dependence", not independence. Bid-ask bounce leaks contemporaneous ρ₀ into
small lags mechanically, so naive peak-|ρ| tests "discover" every correlated
sector-mate. Two defenses, both implemented here:

1. Inference runs on the ASYMMETRY statistic
       A_ij = Σ_{ℓ∈grid} (|ρ_ij(ℓ)| − |ρ_ji(ℓ)|)
   (absolute values — signed sums confuse direction-of-lead with sign-of-
   relationship), which is null-centered under symmetric bounce leakage.
2. The permutation null shifts the LEADER side only — value and mask jointly,
   by whole trading days, within strata of identical grid-day structure — so
   the follower panel's contemporaneous dependence is preserved. Observed and
   null statistics use the SAME decimated lag grid.

Per draw, one uniform day-offset is applied to the whole leader panel (each
ordered pair sees leader-shifted vs follower-original); draws are exchangeable
across pairs.

FDR families: A = pre-registered economic priors (configs/leadlag_priors.yaml)
→ Benjamini–Hochberg; B = everything else → Benjamini–Yekutieli.

Stability: share of trailing monthly re-scans agreeing in sign with the
full-window A_ij; half_life = ln2 / (−ln φ̂) months from an AR(1) fit.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

from futures_common.rng import make_rng
from gp_cta.leadlag.scan import (
    GramSums,
    ScanConfig,
    ScanResult,
    finalize,
    pad_panel,
)
from gp_cta.panel_io import PanelReader

DEFAULT_LAG_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60)


# ------------------------------------------------------------- pair statistics


def asymmetry_matrix(rho: np.ndarray, lag_grid: Sequence[int]) -> np.ndarray:
    """A[i, j] = Σ_grid |ρ_ij| − |ρ_ji| (NaN lags contribute 0)."""
    grid_idx = np.asarray(lag_grid) - 1
    abs_rho = np.abs(rho[grid_idx])
    abs_rho = np.where(np.isfinite(abs_rho), abs_rho, 0.0)
    summed = abs_rho.sum(axis=0)
    out: np.ndarray = summed - summed.T
    return out


def pair_statistics(
    scan: ScanResult, *, lag_grid: Sequence[int] = DEFAULT_LAG_GRID
) -> pl.DataFrame:
    """Per ordered pair: a_stat, detection sup|ρ|, best lag, sign, n_eff."""
    a = asymmetry_matrix(scan.rho, lag_grid)
    grid_idx = np.asarray(lag_grid) - 1
    rho_g = scan.rho[grid_idx]
    with np.errstate(invalid="ignore"):
        abs_rho = np.abs(rho_g)
    det = np.nanmax(abs_rho, axis=0)
    best_k = np.nanargmax(np.where(np.isfinite(abs_rho), abs_rho, -1.0), axis=0)
    n_products = len(scan.products)
    rows: list[dict[str, object]] = []
    for i in range(n_products):
        for j in range(n_products):
            if i == j:
                continue
            k = int(best_k[i, j])
            lag = int(lag_grid[k])
            rho_best = float(rho_g[k, i, j])
            n_eff = int(np.min(scan.n[grid_idx, i, j]))
            rows.append(
                {
                    "leader": scan.products[i],
                    "follower": scan.products[j],
                    "a_stat": float(a[i, j]),
                    "det_stat": float(det[i, j]) if np.isfinite(det[i, j]) else None,
                    "lag_bars": lag,
                    "strength": rho_best if np.isfinite(rho_best) else None,
                    "sign": int(np.sign(rho_best)) if np.isfinite(rho_best) else 0,
                    "n_eff": n_eff,
                }
            )
    return pl.DataFrame(rows)


# ------------------------------------------------------------ permutation null


def _day_strata(td: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group the year's rows into strata of consecutive days with IDENTICAL
    per-day row counts. Returns [(day_start_rows, rows_per_day)] per stratum —
    within a stratum a whole-day circular shift is a plain np.roll."""
    if td.size == 0:
        return []
    day_changes = np.flatnonzero(np.diff(td)) + 1
    day_starts = np.concatenate([[0], day_changes])
    day_lens = np.diff(np.concatenate([day_starts, [td.size]]))
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    s = 0
    for k in range(1, day_lens.size + 1):
        if k == day_lens.size or day_lens[k] != day_lens[s]:
            strata.append((day_starts[s:k], day_lens[s:k]))
            s = k
    return strata


def permutation_null(
    panel: PanelReader,
    observed: pl.DataFrame,
    cfg: ScanConfig,
    *,
    lag_grid: Sequence[int] = DEFAULT_LAG_GRID,
    n_draws: int = 200,
    min_shift_days: int = 5,
    seed: int = 42,
    window: tuple[dt.date, dt.date] | None = None,
    stream: str = "leadlag.null",
) -> pl.DataFrame:
    """Adds `p_value` per ordered pair: P(|A_null| ≥ |A_obs|), leader-shifted null."""
    from gp_cta.leadlag.scan import prep_year_block

    rng = make_rng(seed, stream)
    lags = list(lag_grid)
    n_products = panel.n_products

    # cache per-year raw blocks once (masked values, mask, trading dates, spans)
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]] = []
    for year in panel.years:
        block = prep_year_block(panel, year, window)
        if block is not None:
            blocks.append(block)

    a_obs = {(r["leader"], r["follower"]): abs(r["a_stat"]) for r in observed.iter_rows(named=True)}
    exceed = dict.fromkeys(a_obs, 0)

    for draw in range(n_draws):
        total = GramSums.zeros(cfg.max_lag, n_products)
        for xm, m, td, spans in blocks:
            # leader side: values AND mask shifted jointly by the same offsets
            xs, ms = _shift_pair(xm, m, td, rng, min_shift_days)
            xp_s, mp_s = pad_panel(xs, ms, spans, cfg.max_lag)
            xp_f, mp_f = pad_panel(xm, m, spans, cfg.max_lag)
            total.add(_cross_gram(xp_s, mp_s, xp_f, mp_f, lags))
        null_scan = finalize(total, panel.products, cfg, dt.date.min, dt.date.max, lags)
        a_null = asymmetry_matrix(null_scan.rho, lags)
        for (leader, follower), obs_val in a_obs.items():
            i = panel.products.index(leader)
            j = panel.products.index(follower)
            if abs(a_null[i, j]) >= obs_val:
                exceed[(leader, follower)] += 1
        if (draw + 1) % 50 == 0:
            logger.info("permutation null: {}/{} draws", draw + 1, n_draws)

    p = {pair: (1 + count) / (n_draws + 1) for pair, count in exceed.items()}
    return observed.with_columns(
        pl.struct(["leader", "follower"])
        .map_elements(lambda s: p[(s["leader"], s["follower"])], return_dtype=pl.Float64)
        .alias("p_value")
    )


def _shift_pair(
    x: np.ndarray,
    m: np.ndarray,
    td: np.ndarray,
    rng: np.random.Generator,
    min_shift_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift value and mask JOINTLY (one offset per stratum for both)."""
    xs = x.copy()
    ms = m.copy()
    for day_starts, day_lens in _day_strata(td):
        n_days = day_starts.size
        if n_days < 2 * min_shift_days:
            continue
        shift_days = int(rng.integers(min_shift_days, n_days - min_shift_days + 1))
        lo = day_starts[0]
        hi = day_starts[-1] + day_lens[-1]
        rows = shift_days * int(day_lens[0])
        xs[lo:hi] = np.roll(x[lo:hi], rows, axis=0)
        ms[lo:hi] = np.roll(m[lo:hi], rows, axis=0)
    return xs, ms


def _cross_gram(
    x_lead: np.ndarray,
    m_lead: np.ndarray,
    x_fol: np.ndarray,
    m_fol: np.ndarray,
    lags: Sequence[int],
) -> GramSums:
    """Gram products with DIFFERENT leader (rows t) and follower (rows t+ℓ) panels."""
    n_products = x_lead.shape[1]
    out = GramSums.zeros(max(lags), n_products)
    x2_lead = x_lead * x_lead
    x2_fol = x_fol * x_fol
    for lag in lags:
        out.s_xy[lag - 1] = (x_lead[:-lag].T @ x_fol[lag:]).astype(np.float64)
        out.s_x2m[lag - 1] = (x2_lead[:-lag].T @ m_fol[lag:]).astype(np.float64)
        out.s_mx2[lag - 1] = (m_lead[:-lag].T @ x2_fol[lag:]).astype(np.float64)
        out.n[lag - 1] = (m_lead[:-lag].T @ m_fol[lag:]).astype(np.float64)
    return out


# ----------------------------------------------------------------------- FDR


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values."""
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def by_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Yekutieli q-values (harsh c(M) correction)."""
    m = p.size
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return np.minimum(bh_adjust(p) * c_m, 1.0)


@dataclass(frozen=True, slots=True)
class PriorSet:
    """Family-A directed pairs (both directions of each configured pair)."""

    pairs: frozenset[tuple[str, str]]

    @classmethod
    def from_config(cls, raw_pairs: Sequence[Sequence[str]]) -> PriorSet:
        out: set[tuple[str, str]] = set()
        for a, b in raw_pairs:
            out.add((str(a).upper(), str(b).upper()))
            out.add((str(b).upper(), str(a).upper()))
        return cls(pairs=frozenset(out))

    def contains(self, leader: str, follower: str) -> bool:
        return (leader, follower) in self.pairs


def apply_fdr(
    stats: pl.DataFrame, priors: PriorSet, *, q_a: float = 0.05, q_b: float = 0.10
) -> pl.DataFrame:
    """Adds family + q_value; `validated` requires q ≤ the family's cut AND
    a_stat > 0 — the asymmetry test is antisymmetric (A_ij = −A_ji, identical
    p both ways), so only the direction that actually leads becomes an edge."""
    fam = [
        "A" if priors.contains(r["leader"], r["follower"]) else "B"
        for r in stats.iter_rows(named=True)
    ]
    stats = stats.with_columns(pl.Series("family", fam))
    frames = []
    for family, adjust, cut in (("A", bh_adjust, q_a), ("B", by_adjust, q_b)):
        sub = stats.filter(pl.col("family") == family)
        if sub.is_empty():
            continue
        q = adjust(sub["p_value"].to_numpy())
        frames.append(
            sub.with_columns(
                pl.Series("q_value", q),
                ((pl.Series("q_value", q) <= cut) & (pl.col("a_stat") > 0)).alias("validated"),
            )
        )
    return pl.concat(frames, how="vertical") if frames else stats


# ------------------------------------------------------------------ stability


def stability_metrics(
    monthly: pl.DataFrame,
    full_stats: pl.DataFrame,
    *,
    lookback: int = 12,
    q_cut: float = 0.10,
) -> pl.DataFrame:
    """monthly: (leader, follower, window_end, a_stat[, q_value]) rolling rows.

    stability = share of the last `lookback` monthly windows with the SAME
    sign as the full-window A_ij AND (when a per-window q is available)
    q ≤ q_cut. half_life from an AR(1) fit on the full monthly A series.
    """
    if monthly.is_empty():
        return full_stats.with_columns(
            pl.lit(0.0).alias("stability"), pl.lit(float("nan")).alias("half_life")
        )
    has_q = "q_value" in monthly.columns
    rows: list[dict[str, object]] = []
    for r in full_stats.iter_rows(named=True):
        sub = monthly.filter(
            (pl.col("leader") == r["leader"]) & (pl.col("follower") == r["follower"])
        ).sort("window_end")
        series = sub["a_stat"].to_numpy()
        tail = sub.tail(lookback)
        if tail.is_empty():
            stability, half_life = 0.0, float("nan")
        else:
            ok = np.sign(tail["a_stat"].to_numpy()) == np.sign(r["a_stat"])
            if has_q:
                ok &= tail["q_value"].to_numpy() <= q_cut
            stability = float(np.mean(ok))
            half_life = _ar1_half_life(series)
        rows.append(
            {
                "leader": r["leader"],
                "follower": r["follower"],
                "stability": stability,
                "half_life": half_life,
            }
        )
    return full_stats.join(pl.DataFrame(rows), on=["leader", "follower"], how="left")


def _ar1_half_life(series: np.ndarray) -> float:
    if series.size < 6:
        return float("nan")
    x, y = series[:-1], series[1:]
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    phi = float(np.dot(x, y)) / denom
    if not (0 < phi < 1):
        return float("nan")
    return float(np.log(2) / -np.log(phi))
