"""GP rolling kernels vs pandas references + segmentation resets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from futures_common.rng import make_rng
from gp_cta.gp import ops

RNG = make_rng(42, "test.ops")
X = RNG.normal(size=(500, 3)).astype(np.float32)
Y = RNG.normal(size=(500, 3)).astype(np.float32)
W = 20


def _pd(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(x.astype(np.float64))


def test_ts_mean_std_sum() -> None:
    np.testing.assert_allclose(
        ops.ts_mean(X, W), _pd(X).rolling(W).mean().to_numpy(), atol=1e-5, equal_nan=True
    )
    np.testing.assert_allclose(
        ops.ts_std(X, W), _pd(X).rolling(W).std(ddof=0).to_numpy(), atol=1e-5, equal_nan=True
    )
    np.testing.assert_allclose(
        ops.ts_sum(X, W), _pd(X).rolling(W).sum().to_numpy(), atol=1e-4, equal_nan=True
    )


def test_ts_min_max_delta_ref() -> None:
    np.testing.assert_allclose(
        ops.ts_min(X, W), _pd(X).rolling(W).min().to_numpy(), atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        ops.ts_max(X, W), _pd(X).rolling(W).max().to_numpy(), atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        ops.ts_delta(X, W), (_pd(X) - _pd(X).shift(W)).to_numpy(), atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(ops.ref(X, 5), _pd(X).shift(5).to_numpy(), atol=1e-6, equal_nan=True)


def test_ts_rank_vs_pandas() -> None:
    ours = ops.ts_rank(X, W)
    ref = _pd(X).rolling(W).rank(pct=True).to_numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-6, equal_nan=True)


def test_ts_zscore() -> None:
    mean = _pd(X).rolling(W).mean()
    std = _pd(X).rolling(W).std(ddof=0)
    ref = ((_pd(X) - mean) / std).to_numpy()
    np.testing.assert_allclose(ops.ts_zscore(X, W), ref, atol=1e-4, equal_nan=True)


def test_ema_vs_pandas() -> None:
    alpha = 1.0 - np.exp(-np.log(2.0) / W)
    ref = _pd(X).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(ops.ema(X, W), ref, atol=1e-5)


def test_decay_linear_manual() -> None:
    ours = ops.decay_linear(X, 5)
    weights = np.arange(1, 6, dtype=np.float64)
    for t in (10, 100, 499):
        window = X[t - 4 : t + 1, 0].astype(np.float64)
        ref = float(np.dot(window, weights) / weights.sum())
        assert abs(float(ours[t, 0]) - ref) < 1e-5


def test_ts_corr_vs_pandas() -> None:
    ours = ops.ts_corr(X, Y, W)
    ref = np.column_stack(
        [_pd(X)[c].rolling(W).corr(_pd(Y)[c]).to_numpy() for c in range(X.shape[1])]
    )
    both = np.isfinite(ours) & np.isfinite(ref)
    np.testing.assert_allclose(ours[both], ref[both], atol=1e-4)


def test_apply_segmented_resets_windows() -> None:
    boundaries = [0, 250]  # hard boundary mid-panel (the 2025 hole)
    out = ops.apply_segmented(ops.ts_mean, X, boundaries, W)
    # rows just after the boundary must be NaN (window restarted)
    assert np.isnan(out[250 : 250 + W - 1]).all()
    assert np.isfinite(out[250 + W - 1]).all()
    # and equal a fresh computation on the second segment alone
    fresh = ops.ts_mean(X[250:], W)
    np.testing.assert_allclose(out[250:], fresh, equal_nan=True)
