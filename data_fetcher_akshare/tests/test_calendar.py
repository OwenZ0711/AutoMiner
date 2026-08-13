"""Tests for calendar.py — pure-function helpers over a cached pd.DatetimeIndex.

The calendar's job is small: persist the upstream calendar to parquet, decide
when it's stale, and answer "is X a trading day / what's the next one / how
many between A and B." All the rest belongs in Source.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from data_fetcher import trade_calendar as tc


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    cal = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    path = tmp_path / "trade_dates.parquet"
    tc.save_cache(cal, path)
    loaded = tc.load_cached(path)
    assert loaded is not None
    assert list(loaded) == list(cal.normalize())


def test_load_cached_missing_returns_none(tmp_path: Path) -> None:
    assert tc.load_cached(tmp_path / "does_not_exist.parquet") is None


def test_load_cached_stale_returns_none(tmp_path: Path) -> None:
    cal = pd.DatetimeIndex(["2024-01-02"])
    path = tmp_path / "stale.parquet"
    tc.save_cache(cal, path)
    # Set mtime to 30 days ago.
    old = (dt.datetime.now() - dt.timedelta(days=30)).timestamp()
    import os

    os.utime(path, (old, old))
    assert tc.load_cached(path, max_age_days=7) is None
    # But with a long enough max_age_days, it loads.
    assert tc.load_cached(path, max_age_days=60) is not None


def test_is_trading_day(trade_calendar_fixture: pd.DatetimeIndex) -> None:
    cal = trade_calendar_fixture
    assert tc.is_trading_day(cal, pd.Timestamp("2023-01-03"))   # Tuesday, normal day
    assert not tc.is_trading_day(cal, pd.Timestamp("2023-01-01"))  # New Year
    assert not tc.is_trading_day(cal, pd.Timestamp("2023-01-07"))  # Saturday
    assert not tc.is_trading_day(cal, pd.Timestamp("2023-01-23"))  # CNY


def test_next_trading_day(trade_calendar_fixture: pd.DatetimeIndex) -> None:
    cal = trade_calendar_fixture
    # Across the CNY holiday: Jan 20 (Fri) → Jan 30 (Mon)
    assert tc.next_trading_day(cal, pd.Timestamp("2023-01-20")) == pd.Timestamp(
        "2023-01-30"
    )
    # Across a weekend: Jan 6 (Fri) → Jan 9 (Mon)
    assert tc.next_trading_day(cal, pd.Timestamp("2023-01-06")) == pd.Timestamp(
        "2023-01-09"
    )


def test_previous_trading_day(trade_calendar_fixture: pd.DatetimeIndex) -> None:
    cal = trade_calendar_fixture
    # Across CNY: Jan 30 (Mon) ← Jan 20 (Fri)
    assert tc.previous_trading_day(cal, pd.Timestamp("2023-01-30")) == pd.Timestamp(
        "2023-01-20"
    )
    assert tc.previous_trading_day(cal, pd.Timestamp("2023-01-09")) == pd.Timestamp(
        "2023-01-06"
    )


def test_trading_days_between_inclusive(
    trade_calendar_fixture: pd.DatetimeIndex,
) -> None:
    cal = trade_calendar_fixture
    days = tc.trading_days_between(
        cal, pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-09")
    )
    # Jan 3, 4, 5, 6 are trading days (Tue-Fri), then Jan 9 (Mon). 5 days inclusive.
    assert len(days) == 5
    assert days[0] == pd.Timestamp("2023-01-03")
    assert days[-1] == pd.Timestamp("2023-01-09")


def test_trading_days_between_empty_range(
    trade_calendar_fixture: pd.DatetimeIndex,
) -> None:
    cal = trade_calendar_fixture
    days = tc.trading_days_between(
        cal, pd.Timestamp("2023-01-22"), pd.Timestamp("2023-01-22")
    )
    # Sunday — 0 trading days.
    assert len(days) == 0


def test_trading_days_between_swapped_args_is_empty(
    trade_calendar_fixture: pd.DatetimeIndex,
) -> None:
    cal = trade_calendar_fixture
    days = tc.trading_days_between(
        cal, pd.Timestamp("2023-01-09"), pd.Timestamp("2023-01-03")
    )
    assert len(days) == 0


def test_save_cache_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "trade_dates.parquet"
    cal = pd.DatetimeIndex(["2024-01-02"])
    tc.save_cache(cal, nested)
    assert nested.exists()
    loaded = tc.load_cached(nested)
    assert loaded is not None and list(loaded) == list(cal.normalize())


def test_save_cache_writes_one_column(tmp_path: Path) -> None:
    cal = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    path = tmp_path / "trade_dates.parquet"
    tc.save_cache(cal, path)
    df = pl.read_parquet(path)
    assert df.columns == ["trade_date"]
    assert df.height == 2
