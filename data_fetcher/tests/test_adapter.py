"""Tests for adapter.py — AKshareParquetAdapter implements the DataAdapter
protocol from proposal §8.3.

The adapter is the type every downstream pipeline depends on. Verifying the
protocol contract here means the rest of the AutoLLM build can program
against `DataAdapter` without runtime surprises.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from data_fetcher.adapter import AKshareParquetAdapter, DataAdapter
from data_fetcher.fetcher import Fetcher
from data_fetcher.storage import OHLCV_SCHEMA


# ---------- Fakes (re-used from test_fetcher) ----------


def _v1_event_log() -> pl.DataFrame:
    rows = [
        ("600519", dt.date(2018, 1, 1), None),
        ("000001", dt.date(2018, 1, 1), None),
    ]
    return pl.DataFrame(
        rows,
        schema={"stock_code": pl.Utf8, "in_date": pl.Date, "out_date": pl.Date},
        orient="row",
    )


def _make_panel(symbol: str, n_days: int = 30) -> pl.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days).date.tolist()
    rows = []
    for i, d in enumerate(dates):
        p = 100.0 + i * 0.5
        rows.append(
            {
                "date": d,
                "symbol": symbol,
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p + 0.3,
                "volume": 1e6 + i * 100,
                "turnover": 1e8,
                "vwap": p + 0.2,
                "return_pct": 0.3,
                "turnover_rate": 0.5,
            }
        )
    return pl.DataFrame(rows, schema=OHLCV_SCHEMA)


class _FakeSource:
    name = "fake"

    def pull_ohlcv(self, symbol, start, end, adjust="qfq"):  # type: ignore[no-untyped-def]
        return _make_panel(symbol)

    def list_universe_event_log(self, name):  # type: ignore[no-untyped-def]
        return _v1_event_log()

    def trade_calendar(self):  # type: ignore[no-untyped-def]
        return pd.bdate_range("2018-01-01", "2024-12-31").normalize()


@pytest.fixture
def adapter(tmp_data_root: Path) -> AKshareParquetAdapter:
    f = Fetcher(source=_FakeSource(), root=tmp_data_root, sleep_s=0.0)
    f.pull(symbols=["SH600519", "SZ000001"], start="2024-01-02", end="2024-02-15")
    return AKshareParquetAdapter(f)


# ---------- Protocol conformance ----------


def test_adapter_satisfies_protocol(adapter: AKshareParquetAdapter) -> None:
    """Runtime-checkable Protocol; assert isinstance."""
    assert isinstance(adapter, DataAdapter)


# ---------- panel ----------


def test_panel_returns_lazyframe(adapter: AKshareParquetAdapter) -> None:
    lf = adapter.panel(
        universe=["SH600519"],
        fields=["close", "vwap"],
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-31"),
    )
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert set(df.columns) == {"date", "symbol", "close", "vwap"}
    assert df.height > 0


def test_panel_with_universe_name(adapter: AKshareParquetAdapter) -> None:
    lf = adapter.panel(
        universe="csi300",
        fields=["close"],
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-31"),
    )
    df = lf.collect()
    assert "SH600519" in set(df["symbol"].unique().to_list())


# ---------- calendar / universe ----------


def test_calendar_returns_datetimeindex(adapter: AKshareParquetAdapter) -> None:
    cal = adapter.calendar()
    assert isinstance(cal, pd.DatetimeIndex)
    assert len(cal) > 100


def test_universe_returns_normalised_symbols(adapter: AKshareParquetAdapter) -> None:
    members = adapter.universe("csi300", pd.Timestamp("2024-06-01"))
    assert {"SH600519", "SZ000001"} <= set(members)
    assert all(m.startswith(("SH", "SZ")) for m in members)


# ---------- windows ----------


def test_windows_yields_correct_shape(adapter: AKshareParquetAdapter) -> None:
    """Sliding 5-day windows over 30 trading days should yield 26 windows."""
    wins = list(
        adapter.windows(
            asset="SH600519",
            start=pd.Timestamp("2024-01-02"),
            end=pd.Timestamp("2024-02-15"),
            length=5,
        )
    )
    assert len(wins) >= 20  # depending on calendar, ~30-5+1 windows
    for w in wins:
        assert isinstance(w, np.ndarray)
        # default fields = open, high, low, close, volume, vwap → 6 columns
        assert w.shape == (5, 6)


def test_windows_chronological(adapter: AKshareParquetAdapter) -> None:
    wins = list(
        adapter.windows(
            asset="SH600519",
            start=pd.Timestamp("2024-01-02"),
            end=pd.Timestamp("2024-01-31"),
            length=3,
        )
    )
    # Close column (index 3) of consecutive windows shifts by one bar.
    closes = [w[:, 3] for w in wins]
    for prev, nxt in zip(closes, closes[1:]):
        # The last 2 elements of prev should match the first 2 elements of nxt.
        np.testing.assert_array_equal(prev[1:], nxt[:-1])


def test_windows_empty_when_insufficient_data(adapter: AKshareParquetAdapter) -> None:
    """If we ask for length > # available days, yield nothing."""
    wins = list(
        adapter.windows(
            asset="SH600519",
            start=pd.Timestamp("2024-01-02"),
            end=pd.Timestamp("2024-01-08"),
            length=1000,
        )
    )
    assert wins == []


def test_windows_unknown_asset_yields_empty(adapter: AKshareParquetAdapter) -> None:
    wins = list(
        adapter.windows(
            asset="SH999999",
            start=pd.Timestamp("2024-01-02"),
            end=pd.Timestamp("2024-01-31"),
            length=5,
        )
    )
    assert wins == []


# ---------- context_bundle (stub) ----------


def test_context_bundle_returns_dict(adapter: AKshareParquetAdapter) -> None:
    bundle = adapter.context_bundle(pd.Timestamp("2024-06-01"))
    assert isinstance(bundle, dict)
    # v1 stub: at minimum has 'as_of'.
    assert "as_of" in bundle
