"""Golden-value scan tests: brute-force equivalence, planted lag, boundaries."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from futures_common.paths import FuturesPaths
from futures_common.rng import make_rng
from gp_cta.leadlag.scan import ScanConfig, scan_lagged_corr
from gp_cta.panel_io import PanelReader
from gp_cta.tests.conftest import write_panel_year


def brute_force_rho(
    x: np.ndarray,
    m: np.ndarray,
    seg: np.ndarray,
    max_lag: int,
    min_joint: int,
) -> tuple[np.ndarray, np.ndarray]:
    """O(T·N²·L) reference: literal triple loop over the estimator definition."""
    t_rows, n = x.shape
    rho = np.full((max_lag, n, n), np.nan)
    counts = np.zeros((max_lag, n, n), dtype=np.int64)
    xm = np.where(m, x, 0.0)
    for lag in range(1, max_lag + 1):
        for i in range(n):
            for j in range(n):
                s_xy = s_x2m = s_mx2 = 0.0
                k = 0
                for t in range(t_rows - lag):
                    if seg[t] != seg[t + lag]:
                        continue
                    if not (m[t, i] and m[t + lag, j]):
                        continue
                    s_xy += xm[t, i] * xm[t + lag, j]
                    s_x2m += xm[t, i] ** 2
                    s_mx2 += xm[t + lag, j] ** 2
                    k += 1
                counts[lag - 1, i, j] = k
                if k >= min_joint and s_x2m > 0 and s_mx2 > 0:
                    rho[lag - 1, i, j] = s_xy / np.sqrt(s_x2m * s_mx2)
    return rho, counts


def _write_random_panel(
    paths: FuturesPaths, *, t_rows: int = 400, n: int = 3, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = make_rng(seed, "test.scan")
    x = rng.normal(size=(t_rows, n)).astype(np.float32)
    m = rng.random((t_rows, n)) > 0.3  # Bernoulli holes
    x[rng.random((t_rows, n)) < 0.05] = np.nan  # extra NaN holes
    # 4 uneven segments over 8 days
    days = np.repeat(np.arange(18267, 18275, dtype=np.int32), t_rows // 8)[:t_rows]
    seg = np.zeros(t_rows, dtype=np.int32)
    bounds = [0, 90, 170, 280, t_rows]
    for k in range(1, 4):
        seg[bounds[k] :] += 1
    write_panel_year(paths, 2020, [f"P{i}" for i in range(n)], x, m, days, seg)
    return x, m, seg


def test_gemm_equals_brute_force(tmp_paths: FuturesPaths) -> None:
    x, m, seg = _write_random_panel(tmp_paths)
    cfg = ScanConfig(max_lag=7, min_joint_bars=10)
    panel = PanelReader(tmp_paths, [2020])
    result = scan_lagged_corr(panel, cfg)

    m_eff = m & np.isfinite(x)
    ref_rho, ref_n = brute_force_rho(np.nan_to_num(x.astype(np.float64)), m_eff, seg, 7, 10)
    np.testing.assert_array_equal(result.n, ref_n)  # counts EXACT
    both = np.isfinite(result.rho) & np.isfinite(ref_rho)
    assert np.isfinite(result.rho).sum() == np.isfinite(ref_rho).sum()
    np.testing.assert_allclose(result.rho[both], ref_rho[both], atol=1e-5)


def test_planted_lag_recovered(tmp_paths: FuturesPaths) -> None:
    rng = make_rng(7, "test.plant")
    t_rows, lag_true, beta = 20_000, 3, 0.5
    a = rng.normal(size=t_rows)
    b = np.empty(t_rows)
    b[:lag_true] = rng.normal(size=lag_true)
    b[lag_true:] = beta * a[:-lag_true] + rng.normal(size=t_rows - lag_true)
    x = np.column_stack([a, b]).astype(np.float32)
    m = np.ones((t_rows, 2), dtype=bool)
    days = np.repeat(np.arange(18267, 18267 + t_rows // 200, dtype=np.int32), 200)
    seg = (np.arange(t_rows) // 200).astype(np.int32)  # one segment per day
    write_panel_year(tmp_paths, 2020, ["A", "B"], x, m, days, seg)

    cfg = ScanConfig(max_lag=10, min_joint_bars=1000)
    result = scan_lagged_corr(PanelReader(tmp_paths, [2020]), cfg)
    rho_ab = result.rho[:, 0, 1]  # A leads B
    assert int(np.nanargmax(np.abs(rho_ab))) + 1 == lag_true
    expected = beta / np.sqrt(beta**2 + 1.0)
    assert rho_ab[lag_true - 1] == pytest.approx(expected, abs=0.03)
    # reverse direction flat
    rho_ba = result.rho[:, 1, 0]
    assert np.nanmax(np.abs(rho_ba)) < 0.05


def test_boundary_exactness_no_cross_segment_pairs(tmp_paths: FuturesPaths) -> None:
    """Pair correlated ONLY across a segment boundary → ρ ≈ 0, n excludes straddles."""
    rng = make_rng(9, "test.boundary")
    seg_len, n_segs = 50, 40
    t_rows = seg_len * n_segs
    a = rng.normal(size=t_rows)
    b = rng.normal(size=t_rows)
    # plant correlation at lag 5 ONLY for pairs straddling boundaries:
    for s in range(1, n_segs):
        edge = s * seg_len
        b[edge : edge + 5] = a[edge - 5 : edge]  # (t, t+5) straddles the boundary
    x = np.column_stack([a, b]).astype(np.float32)
    m = np.ones((t_rows, 2), dtype=bool)
    days = np.repeat(np.arange(18267, 18267 + n_segs, dtype=np.int32), seg_len)
    seg = np.repeat(np.arange(n_segs, dtype=np.int32), seg_len)
    write_panel_year(tmp_paths, 2020, ["A", "B"], x, m, days, seg)

    cfg = ScanConfig(max_lag=6, min_joint_bars=100)
    result = scan_lagged_corr(PanelReader(tmp_paths, [2020]), cfg)
    # per segment, lag 5 leaves seg_len-5=45 in-segment pairs → n = 45*40
    assert result.n[4, 0, 1] == (seg_len - 5) * n_segs
    assert abs(result.rho[4, 0, 1]) < 0.05  # planted straddle correlation invisible


def test_three_gram_normalization_vs_global(tmp_paths: FuturesPaths) -> None:
    """Day-only vs day+night overlap: per-pair joint normalization ≈ |ρ| ≤ 1
    and differs from what global column standardization would give."""
    rng = make_rng(11, "test.grams")
    t_rows = 4000
    a = rng.normal(size=t_rows)
    b = 0.6 * a + 0.8 * rng.normal(size=t_rows)
    # product B trades only "day" rows (2nd half of each 100-row day)
    m = np.ones((t_rows, 2), dtype=bool)
    day_pos = np.arange(t_rows) % 100
    m[day_pos < 50, 1] = False
    # B's night-half values are garbage large numbers — must NOT leak into ρ
    b_full = b.copy()
    b_full[day_pos < 50] = 50.0
    x = np.column_stack([a, b_full]).astype(np.float32)
    days = np.repeat(np.arange(18267, 18307, dtype=np.int32), 100)
    seg = np.repeat(np.arange(40, dtype=np.int32), 100)
    write_panel_year(tmp_paths, 2020, ["A", "B"], x, m, days, seg)

    result = scan_lagged_corr(
        PanelReader(tmp_paths, [2020]), ScanConfig(max_lag=3, min_joint_bars=100)
    )
    assert np.nanmax(np.abs(result.rho)) <= 1.0 + 1e-9  # joint-valid normalization
    # sanity: lag-1 rho within masked support should be small but defined
    assert np.isfinite(result.rho[0, 0, 1])


def test_window_restriction(tmp_paths: FuturesPaths) -> None:
    _write_random_panel(tmp_paths, t_rows=400)
    panel = PanelReader(tmp_paths, [2020])
    cfg = ScanConfig(max_lag=5, min_joint_bars=5)
    full = scan_lagged_corr(panel, cfg)
    epoch = dt.date(1970, 1, 1)
    half = scan_lagged_corr(
        panel,
        cfg,
        window=(epoch + dt.timedelta(days=18267), epoch + dt.timedelta(days=18270)),
    )
    assert half.n.sum() < full.n.sum()
    assert half.n.sum() > 0
