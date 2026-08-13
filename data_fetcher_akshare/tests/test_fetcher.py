"""Tests for fetcher.py — orchestrator over Source + ParquetStore + UniverseCache."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from data_fetcher.exceptions import EmptyResponseError
from data_fetcher.fetcher import Fetcher
from data_fetcher.storage import OHLCV_SCHEMA


# ---------- Fakes ----------


def _v1_event_log() -> pl.DataFrame:
    rows = [
        ("600519", dt.date(2018, 1, 1), None),
        ("000001", dt.date(2018, 1, 1), None),
        ("300750", dt.date(2018, 12, 14), None),
    ]
    return pl.DataFrame(
        rows,
        schema={"stock_code": pl.Utf8, "in_date": pl.Date, "out_date": pl.Date},
        orient="row",
    )


def _make_panel(symbol: str, dates: list[dt.date], price_base: float = 100.0) -> pl.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        p = price_base + i * 0.5
        rows.append(
            {
                "date": d,
                "symbol": symbol,
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p + 0.3,
                "volume": 1e6,
                "turnover": 1e8,
                "vwap": p,
                "return_pct": 0.3,
                "turnover_rate": 0.5,
            }
        )
    return pl.DataFrame(rows, schema=OHLCV_SCHEMA)


class _FakeSource:
    name = "fake"

    def __init__(self) -> None:
        self.ohlcv_calls: list[dict] = []
        self.calendar_calls = 0
        self.fail_for: set[str] = set()  # symbols that should raise EmptyResponseError

    def pull_ohlcv(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjust: str = "qfq",
    ) -> pl.DataFrame:
        self.ohlcv_calls.append(
            {"symbol": symbol, "start": start, "end": end, "adjust": adjust}
        )
        if symbol in self.fail_for:
            raise EmptyResponseError(f"fake: nothing for {symbol}")
        # Synthesise 5 rows starting at start.
        dates = pd.bdate_range(pd.Timestamp(start), periods=5).date.tolist()
        return _make_panel(symbol, dates, price_base=100.0 if symbol == "SH600519" else 50.0)

    def list_universe_event_log(self, name: str) -> pl.DataFrame:
        if name == "csi300":
            return _v1_event_log()
        from data_fetcher.exceptions import UniverseUnavailableError

        raise UniverseUnavailableError(name)

    def trade_calendar(self) -> pd.DatetimeIndex:
        self.calendar_calls += 1
        return pd.bdate_range("2018-01-01", "2024-12-31").normalize()


# ---------- Fixtures ----------


@pytest.fixture
def fake_fetcher(tmp_data_root: Path) -> Fetcher:
    return Fetcher(source=_FakeSource(), root=tmp_data_root)


# ---------- pull() ----------


def test_pull_explicit_symbols(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    result = f.pull(
        symbols=["SH600519", "SZ000001"],
        start="2024-01-02",
        end="2024-01-08",
    )
    assert result["pulled"] > 0
    assert result["failed"] == 0
    assert len(src.ohlcv_calls) == 2
    # Symbol ordering preserved.
    assert src.ohlcv_calls[0]["symbol"] == "SH600519"
    assert src.ohlcv_calls[0]["adjust"] == "qfq"


def test_pull_universe_uses_ever_members(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    f.pull(universe="csi300", start="2024-01-02", end="2024-01-08")
    pulled_symbols = {c["symbol"] for c in src.ohlcv_calls}
    # 3 symbols in our v1 event log → 3 pull calls (regardless of in_date).
    assert pulled_symbols == {"SH600519", "SZ000001", "SZ300750"}


def test_pull_idempotent(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    first = f.pull(
        symbols=["SH600519"], start="2024-01-02", end="2024-01-08"
    )
    second = f.pull(
        symbols=["SH600519"], start="2024-01-02", end="2024-01-08"
    )
    assert first["pulled"] > 0
    assert second["pulled"] == 0
    assert second["skipped"] == first["pulled"]


def test_pull_handles_per_symbol_failure(tmp_data_root: Path) -> None:
    src = _FakeSource()
    src.fail_for = {"SZ000001"}
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    result = f.pull(
        symbols=["SH600519", "SZ000001", "SZ300750"],
        start="2024-01-02",
        end="2024-01-08",
    )
    assert result["failed"] == 1
    assert result["pulled"] > 0
    # The failure didn't stop the other two from being pulled.
    assert {c["symbol"] for c in src.ohlcv_calls} == {
        "SH600519",
        "SZ000001",
        "SZ300750",
    }


# ---------- panel() ----------


def test_panel_returns_lazyframe_with_requested_fields(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    f.pull(symbols=["SH600519", "SZ000001"], start="2024-01-02", end="2024-01-08")
    lf = f.panel(
        universe=["SH600519", "SZ000001"],
        fields=["close", "vwap"],
        start="2024-01-02",
        end="2024-01-08",
    )
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert df.height == 10  # 2 symbols × 5 days
    assert set(df.columns) == {"date", "symbol", "close", "vwap"}


def test_panel_with_universe_name_uses_pit_filter(tmp_data_root: Path) -> None:
    """When universe is a name, on_date selects the PIT roster."""
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    f.pull(universe="csi300", start="2018-06-01", end="2018-06-08")

    # As of 2018-06-01: only 600519, 000001 (300750 joined 2018-12-14).
    lf = f.panel(
        universe="csi300",
        fields=["close"],
        start="2018-06-01",
        end="2018-06-08",
        on_date=dt.date(2018, 6, 1),
    )
    df = lf.collect()
    assert set(df["symbol"].unique().to_list()) == {"SH600519", "SZ000001"}


def test_panel_empty_when_no_data(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root, sleep_s=0.0)
    lf = f.panel(
        universe=["SH999999"],
        fields=["close"],
        start="2024-01-02",
        end="2024-01-08",
    )
    df = lf.collect()
    assert df.height == 0


# ---------- universe() / calendar() ----------


def test_universe_forwards_to_cache(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root)
    members = f.universe("csi300", dt.date(2024, 1, 1))
    assert "SH600519" in members
    assert "SZ300750" in members  # in_date 2018-12-14 < 2024-01-01


def test_calendar_caches_to_disk(tmp_data_root: Path) -> None:
    src = _FakeSource()
    f = Fetcher(source=src, root=tmp_data_root)
    cal = f.calendar()
    assert isinstance(cal, pd.DatetimeIndex)
    # Cached file exists.
    cache_path = (
        tmp_data_root / "calendar" / "source=fake" / "trade_dates.parquet"
    )
    assert cache_path.exists()
    # Second call reads from cache (no second source call).
    f.calendar()
    assert src.calendar_calls == 1
