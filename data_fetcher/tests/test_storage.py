"""Tests for storage.py — partitioned parquet OHLCV store.

Idempotency is non-negotiable: re-running a pull with the same args must
write zero new rows on the second invocation. Tested explicitly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from data_fetcher.storage import OHLCV_SCHEMA, ParquetStore


@pytest.fixture
def store(tmp_data_root: Path) -> ParquetStore:
    return ParquetStore(root=tmp_data_root, source="akshare", adjust="qfq")


@pytest.fixture
def two_year_panel() -> pl.DataFrame:
    """A 4-row panel spanning 2023 and 2024 with 2 symbols."""
    rows = [
        # Within 2023
        {"date": dt.date(2023, 12, 28), "symbol": "SH600519", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1e6, "turnover": 1e8, "vwap": 100.2, "return_pct": 0.5, "turnover_rate": 0.4},
        {"date": dt.date(2023, 12, 28), "symbol": "SZ000001", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 5e6, "turnover": 5.1e7, "vwap": 10.1, "return_pct": 2.0, "turnover_rate": 1.0},
        # Within 2024
        {"date": dt.date(2024, 1, 2), "symbol": "SH600519", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1.1e6, "turnover": 1.1e8, "vwap": 101.0, "return_pct": 1.0, "turnover_rate": 0.5},
        {"date": dt.date(2024, 1, 2), "symbol": "SZ000001", "open": 10.2, "high": 10.6, "low": 10.1, "close": 10.4, "volume": 5.2e6, "turnover": 5.4e7, "vwap": 10.3, "return_pct": 1.5, "turnover_rate": 1.1},
    ]
    return pl.DataFrame(rows, schema=OHLCV_SCHEMA)


def test_write_creates_year_symbol_partitions(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    result = store.write_ohlcv(two_year_panel)
    assert result["written"] == 4
    assert result["skipped"] == 0
    # Expect 4 files: 2 symbols × 2 years.
    expected = {
        store.partition_path("SH600519", 2023),
        store.partition_path("SZ000001", 2023),
        store.partition_path("SH600519", 2024),
        store.partition_path("SZ000001", 2024),
    }
    for p in expected:
        assert p.exists(), f"missing partition {p}"


def test_partition_path_format(store: ParquetStore) -> None:
    p = store.partition_path("SH600519", 2024)
    assert p.parts[-3:] == ("adjust=qfq", "year=2024", "symbol=SH600519.parquet")


def test_write_round_trip(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    store.write_ohlcv(two_year_panel)
    loaded = store.read_ohlcv(
        symbols=["SH600519", "SZ000001"],
        start=dt.date(2023, 1, 1),
        end=dt.date(2024, 12, 31),
    ).collect()
    # Same row count.
    assert loaded.height == two_year_panel.height
    # Set of (symbol, date) matches.
    src = {(r["symbol"], r["date"]) for r in two_year_panel.iter_rows(named=True)}
    dst = {(r["symbol"], r["date"]) for r in loaded.iter_rows(named=True)}
    assert src == dst


def test_write_is_idempotent(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    first = store.write_ohlcv(two_year_panel)
    second = store.write_ohlcv(two_year_panel)
    assert first["written"] == 4
    assert first["skipped"] == 0
    assert second["written"] == 0
    assert second["skipped"] == 4


def test_write_partial_overlap_dedupes(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    # First write a 2024-1-2 row for SH600519.
    first = two_year_panel.filter(
        (pl.col("date") == dt.date(2024, 1, 2)) & (pl.col("symbol") == "SH600519")
    )
    store.write_ohlcv(first)
    # Now write a panel that includes that same row PLUS new rows.
    new_rows = pl.DataFrame(
        [
            {"date": dt.date(2024, 1, 2), "symbol": "SH600519", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1.1e6, "turnover": 1.1e8, "vwap": 101.0, "return_pct": 1.0, "turnover_rate": 0.5},
            {"date": dt.date(2024, 1, 3), "symbol": "SH600519", "open": 101.5, "high": 102.5, "low": 100.5, "close": 102.0, "volume": 1.2e6, "turnover": 1.2e8, "vwap": 101.5, "return_pct": 0.5, "turnover_rate": 0.5},
        ],
        schema=OHLCV_SCHEMA,
    )
    result = store.write_ohlcv(new_rows)
    assert result["written"] == 1   # only Jan 3
    assert result["skipped"] == 1   # Jan 2 already there

    # Read back: 2 distinct dates.
    loaded = store.read_ohlcv(
        symbols=["SH600519"], start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31)
    ).collect()
    assert loaded.height == 2
    assert sorted(loaded["date"].to_list()) == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]


def test_read_missing_partition_returns_empty(store: ParquetStore) -> None:
    loaded = store.read_ohlcv(
        symbols=["SH999999"], start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31)
    ).collect()
    assert loaded.height == 0
    # Schema preserved.
    assert "date" in loaded.columns and "symbol" in loaded.columns


def test_read_filters_by_date(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    store.write_ohlcv(two_year_panel)
    loaded = store.read_ohlcv(
        symbols=["SH600519"], start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31)
    ).collect()
    assert loaded.height == 1
    assert loaded["date"][0] == dt.date(2024, 1, 2)


def test_read_filters_by_symbol(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    store.write_ohlcv(two_year_panel)
    loaded = store.read_ohlcv(
        symbols=["SZ000001"], start=dt.date(2023, 1, 1), end=dt.date(2024, 12, 31)
    ).collect()
    assert loaded.height == 2
    assert set(loaded["symbol"].unique().to_list()) == {"SZ000001"}


def test_has_partition(store: ParquetStore, two_year_panel: pl.DataFrame) -> None:
    assert not store.has_partition("SH600519", 2024)
    store.write_ohlcv(two_year_panel)
    assert store.has_partition("SH600519", 2024)
    assert store.has_partition("SZ000001", 2023)
    assert not store.has_partition("SZ000001", 2025)


def test_list_symbols_with_data(
    store: ParquetStore, two_year_panel: pl.DataFrame
) -> None:
    assert store.list_symbols_with_data() == set()
    store.write_ohlcv(two_year_panel)
    assert store.list_symbols_with_data() == {"SH600519", "SZ000001"}


def test_two_stores_different_adjust_dont_collide(tmp_data_root: Path) -> None:
    qfq = ParquetStore(root=tmp_data_root, source="akshare", adjust="qfq")
    hfq = ParquetStore(root=tmp_data_root, source="akshare", adjust="hfq")
    assert qfq.partition_path("SH600519", 2024) != hfq.partition_path("SH600519", 2024)
    assert "adjust=qfq" in str(qfq.partition_path("SH600519", 2024))
    assert "adjust=hfq" in str(hfq.partition_path("SH600519", 2024))
