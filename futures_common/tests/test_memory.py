"""PeakTracker must catch a planted allocation spike and enforce hard budgets."""

from __future__ import annotations

import numpy as np
import pytest

from futures_common.memory import (
    MemoryBudgetError,
    PeakTracker,
    assert_rss_under,
    current_rss_bytes,
)


def test_current_rss_positive() -> None:
    assert current_rss_bytes() > 10 * 1024 * 1024  # a python process is > 10 MB


def test_peak_tracker_catches_spike() -> None:
    baseline = current_rss_bytes()
    with PeakTracker("test-spike", budget_gb=64.0, interval_s=0.01, hard=False) as tracker:
        blob = np.ones(int(500e6 // 8))  # ~500 MB of float64
        blob[::4096] += 1.0  # touch pages so they are resident
        del blob
    assert tracker.peak_rss_bytes > baseline + 300 * 1024 * 1024


def test_peak_tracker_hard_budget_raises() -> None:
    with pytest.raises(MemoryBudgetError):
        with PeakTracker("test-hard", budget_gb=0.001, interval_s=0.01, hard=True):
            pass  # any python process already exceeds 1 MB


def test_peak_tracker_never_swallows_exceptions() -> None:
    with pytest.raises(ValueError, match="inner"):
        with PeakTracker("test-exc", budget_gb=0.001, interval_s=0.01, hard=True):
            raise ValueError("inner")  # budget also exceeded, original error must win


def test_assert_rss_under() -> None:
    assert_rss_under(64.0, "plenty")  # fine
    with pytest.raises(MemoryBudgetError, match="tight"):
        assert_rss_under(0.001, "tight")
