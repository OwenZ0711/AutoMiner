"""Validation-layer tests: null calibration, bounce robustness, FDR, edges."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from futures_common.paths import FuturesPaths
from futures_common.rng import make_rng
from gp_cta.leadlag.edges import EdgeStore
from gp_cta.leadlag.scan import ScanConfig, scan_lagged_corr
from gp_cta.leadlag.validate import (
    PriorSet,
    apply_fdr,
    asymmetry_matrix,
    bh_adjust,
    by_adjust,
    pair_statistics,
    permutation_null,
)
from gp_cta.panel_io import PanelReader
from gp_cta.tests.conftest import write_panel_year

LAG_GRID = (1, 2, 3, 4, 5, 6)
D0 = 18267  # 2020-01-06 epoch days


def _panel(
    tmp_paths: FuturesPaths,
    x: np.ndarray,
    *,
    names: list[str] | None = None,
    bars_per_day: int = 100,
) -> PanelReader:
    t_rows, n = x.shape
    names = names or [f"P{i}" for i in range(n)]
    m = np.ones((t_rows, n), dtype=bool)
    days = np.repeat(np.arange(D0, D0 + t_rows // bars_per_day, dtype=np.int32), bars_per_day)
    seg = np.repeat(np.arange(t_rows // bars_per_day, dtype=np.int32), bars_per_day)
    write_panel_year(tmp_paths, 2020, names, x.astype(np.float32), m, days, seg)
    return PanelReader(tmp_paths, [2020])


def test_null_pvalues_uniform_on_iid_panel(tmp_paths: FuturesPaths) -> None:
    """On an iid panel the null p-values must be ~U(0,1) (loose KS check)."""
    rng = make_rng(3, "test.nulluniform")
    x = rng.normal(size=(8000, 4))
    panel = _panel(tmp_paths, x)
    cfg = ScanConfig(max_lag=6, min_joint_bars=500)
    scan = scan_lagged_corr(panel, cfg, lags=list(LAG_GRID))
    stats = pair_statistics(scan, lag_grid=LAG_GRID)
    stats = permutation_null(
        panel, stats, cfg, lag_grid=LAG_GRID, n_draws=60, min_shift_days=3, seed=42
    )
    p = np.sort(stats["p_value"].to_numpy())
    # Kolmogorov-Smirnov against U(0,1), very loose (12 pairs, 60 draws)
    grid = (np.arange(p.size) + 1) / p.size
    ks = np.max(np.abs(p - grid))
    assert ks < 0.45, f"null p-values not uniform: KS={ks:.2f}, p={p}"
    assert p.min() > 1.0 / 61  # discrete floor respected


def test_planted_lead_detected(tmp_paths: FuturesPaths) -> None:
    rng = make_rng(5, "test.plantvalid")
    t_rows, lag_true, beta = 30_000, 3, 0.35
    a = rng.normal(size=t_rows)
    b = np.empty(t_rows)
    b[:lag_true] = rng.normal(size=lag_true)
    b[lag_true:] = beta * a[:-lag_true] + rng.normal(size=t_rows - lag_true)
    panel = _panel(tmp_paths, np.column_stack([a, b]), names=["LEAD", "FOLL"])
    cfg = ScanConfig(max_lag=6, min_joint_bars=1000)
    scan = scan_lagged_corr(panel, cfg, lags=list(LAG_GRID))
    stats = pair_statistics(scan, lag_grid=LAG_GRID)
    row = stats.filter((pl.col("leader") == "LEAD") & (pl.col("follower") == "FOLL"))
    assert row["a_stat"][0] > 0.2
    assert row["lag_bars"][0] == lag_true
    assert row["sign"][0] == 1

    stats = permutation_null(
        panel, stats, cfg, lag_grid=LAG_GRID, n_draws=99, min_shift_days=3, seed=42
    )
    p_lead = stats.filter((pl.col("leader") == "LEAD") & (pl.col("follower") == "FOLL"))["p_value"][
        0
    ]
    assert p_lead == pytest.approx(1 / 100, abs=1e-9)  # never exceeded


def test_bounce_leakage_does_not_fool_asymmetry(tmp_paths: FuturesPaths) -> None:
    """ρ₀=0.6 contemporaneous + MA(1) bounce, ZERO true lead: sup|ρ| fires at
    small lags mechanically, but the asymmetry statistic stays null-centered."""
    rng = make_rng(13, "test.bounce")
    t_rows = 40_000
    common = rng.normal(size=t_rows)
    e_a = rng.normal(size=t_rows)
    e_b = rng.normal(size=t_rows)
    # efficient prices share a factor
    pa = 0.8 * common + 0.6 * e_a
    pb = 0.8 * common + 0.6 * e_b
    # symmetric MA(1) microstructure noise (bid-ask bounce)
    ma_a = rng.normal(size=t_rows)
    ma_b = rng.normal(size=t_rows)
    a = pa + 0.5 * ma_a - 0.5 * np.concatenate([[0], ma_a[:-1]])
    b = pb + 0.5 * ma_b - 0.5 * np.concatenate([[0], ma_b[:-1]])
    panel = _panel(tmp_paths, np.column_stack([a, b]), names=["A", "B"])
    cfg = ScanConfig(max_lag=6, min_joint_bars=1000)
    scan = scan_lagged_corr(panel, cfg, lags=list(LAG_GRID))
    stats = pair_statistics(scan, lag_grid=LAG_GRID)

    # the asymmetry stat is tiny relative to the detection stat
    a_ab = abs(stats.filter(pl.col("leader") == "A")["a_stat"][0])
    stats = permutation_null(
        panel, stats, cfg, lag_grid=LAG_GRID, n_draws=99, min_shift_days=3, seed=42
    )
    p_ab = stats.filter(pl.col("leader") == "A")["p_value"][0]
    assert p_ab > 0.05, f"bounce false positive: A={a_ab:.4f}, p={p_ab}"


def test_asymmetry_matrix_antisymmetric() -> None:
    rng = make_rng(1, "t")
    rho = rng.normal(size=(6, 3, 3))
    a = asymmetry_matrix(rho, (1, 2, 3))
    np.testing.assert_allclose(a, -a.T, atol=1e-12)


def test_bh_by_hand_computed() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    q = bh_adjust(p)
    np.testing.assert_allclose(q[0], 0.008, atol=1e-9)  # 0.001*8/1 = 0.008
    np.testing.assert_allclose(q[1], 0.032, atol=1e-9)  # 0.008*8/2 = 0.032
    np.testing.assert_allclose(q[4], 0.0672, atol=1e-4)  # min accumul from right
    assert (by_adjust(p) >= q).all()  # BY is uniformly harsher


def test_apply_fdr_families() -> None:
    stats = pl.DataFrame(
        {
            "leader": ["I", "CU", "TA", "AL"],
            "follower": ["RB", "AL", "MA", "CU"],
            "p_value": [0.001, 0.001, 0.5, 0.001],
            "a_stat": [0.3, 0.2, 0.1, -0.2],
        }
    )
    priors = PriorSet.from_config([["I", "RB"]])
    out = apply_fdr(stats, priors, q_a=0.05, q_b=0.10)
    fam = {(r["leader"], r["follower"]): r["family"] for r in out.iter_rows(named=True)}
    assert fam[("I", "RB")] == "A"
    assert fam[("CU", "AL")] == "B"
    validated = {(r["leader"], r["follower"]): r["validated"] for r in out.iter_rows(named=True)}
    assert validated[("I", "RB")]
    assert not validated[("TA", "MA")]
    assert not validated[("AL", "CU")]  # significant q but NEGATIVE asymmetry → not a lead


# ------------------------------------------------------------------- EdgeStore


def _edge_frame(leader: str = "I", follower: str = "RB", a_stat: float = 0.5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "leader": [leader],
            "follower": [follower],
            "lag_bars": [3],
            "strength": [0.04],
            "sign": [1],
            "a_stat": [a_stat],
            "p_value": [0.005],
            "q_value": [0.02],
            "family": ["A"],
            "n_eff": [50_000],
            "stability": [0.9],
            "half_life": [6.0],
            "regime_flags": [""],
            "window_start": [dt.date(2019, 1, 1)],
            "window_end": [dt.date(2021, 12, 31)],
        }
    )


def test_edge_store_as_of_never_leaks_future(tmp_paths: FuturesPaths) -> None:
    tmp_paths.edges_dir.mkdir(parents=True, exist_ok=True)
    store = EdgeStore(tmp_paths)
    store.append(_edge_frame(), asof=dt.date(2022, 1, 10), config_hash="c1" * 6)

    assert store.as_of(dt.date(2022, 1, 10)).is_empty()  # same-day NOT visible
    assert not store.as_of(dt.date(2022, 1, 11)).is_empty()
    assert store.as_of(dt.date(2021, 6, 1)).is_empty()  # before asof


def test_edge_store_latest_asof_wins_and_append_only(tmp_paths: FuturesPaths) -> None:
    tmp_paths.edges_dir.mkdir(parents=True, exist_ok=True)
    store = EdgeStore(tmp_paths)
    store.append(_edge_frame(a_stat=0.5), asof=dt.date(2022, 1, 10), config_hash="a" * 12)
    first_file = next(tmp_paths.edges_dir.glob("*.parquet"))
    first_bytes = first_file.read_bytes()

    store.append(_edge_frame(a_stat=0.9), asof=dt.date(2022, 2, 10), config_hash="a" * 12)
    got = store.as_of(dt.date(2022, 3, 1))
    assert got.height == 1 and got["a_stat"][0] == pytest.approx(0.9)
    assert first_file.read_bytes() == first_bytes  # old file untouched

    with pytest.raises(FileExistsError):
        store.append(_edge_frame(), asof=dt.date(2022, 2, 10), config_hash="a" * 12)


def test_leaders_ranked_by_abs_a_stat(tmp_paths: FuturesPaths) -> None:
    tmp_paths.edges_dir.mkdir(parents=True, exist_ok=True)
    store = EdgeStore(tmp_paths)
    frame = pl.concat(
        [
            _edge_frame(leader="I", a_stat=0.3),
            _edge_frame(leader="J", a_stat=-0.8),
            _edge_frame(leader="HC", a_stat=0.5),
        ]
    )
    store.append(frame, asof=dt.date(2022, 1, 1), config_hash="b" * 12)
    leaders = store.leaders_for("RB", dt.date(2022, 6, 1), top_k=2, max_staleness_days=365)
    assert [e.leader for e in leaders] == ["J", "HC"]
