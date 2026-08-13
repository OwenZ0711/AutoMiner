"""Reader Strategy tests: layout parity, case handling, pseudo skip, schema drift."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from data_fetcher.exceptions import SchemaDriftError
from data_fetcher.readers import DailyTreeReader, HistoricalTreeReader
from data_fetcher.readers.base import is_pseudo_contract, split_contract
from data_fetcher.tests.conftest import (
    Bar,
    ContractSpec,
    day_session_bars,
    write_daily_tree,
    write_historical_tree,
)

D = dt.date(2020, 3, 2)  # a Monday
D26 = dt.date(2026, 1, 5)  # a Monday


def _collect(reader, product: str) -> dict[str, pl.DataFrame]:  # type: ignore[no-untyped-def]
    return {b.contract: b.lf.collect() for b in reader.iter_product(product)}


def test_layout_parity_same_bars_normalize_identically(tmp_path: Path) -> None:
    """The same synthetic bars written into both layouts yield identical frames."""
    bars = day_session_bars(D26, n=4, close=3100.0, open=3099.0, high=3101.0, low=3098.0)
    hist_root = tmp_path / "h"
    daily_root = tmp_path / "d"
    write_historical_tree(hist_root, [ContractSpec("SHFE", "RB2605", bars)])
    write_daily_tree(daily_root, [ContractSpec("SHFE", "rb2605", bars)])

    hist = _collect(HistoricalTreeReader(hist_root / "2005-202506"), "RB")["RB2605"]
    daily = _collect(DailyTreeReader(daily_root / "2026"), "RB")["RB2605"]
    assert hist.equals(daily)
    assert hist["contract"].unique().to_list() == ["RB2605"]
    assert str(hist.schema["bob"]) == "Datetime(time_unit='us', time_zone='Asia/Shanghai')"


def test_lowercase_2026_filenames_and_symbols_uppercased(tmp_path: Path) -> None:
    write_daily_tree(
        tmp_path, [ContractSpec("SHFE", "ag2606", day_session_bars(D26, n=2, close=5000.0))]
    )
    reader = DailyTreeReader(tmp_path / "2026")
    assert reader.discover(None) == ["AG"]
    frames = _collect(reader, "AG")
    assert list(frames) == ["AG2606"]
    assert (frames["AG2606"]["symbol"] == "AG2606").all()


def test_pseudo_contracts_never_read(tmp_path: Path) -> None:
    specs = [
        ContractSpec("SHFE", "rb2605", day_session_bars(D26, n=2)),
        ContractSpec("SHFE", "RB8888", day_session_bars(D26, n=2)),
        ContractSpec("SHFE", "rb9999", day_session_bars(D26, n=2)),
        ContractSpec("SHFE", "rb9998", day_session_bars(D26, n=2)),
    ]
    write_daily_tree(tmp_path, specs)
    reader = DailyTreeReader(tmp_path / "2026")
    assert list(_collect(reader, "RB")) == ["RB2605"]

    write_historical_tree(tmp_path, specs)
    hist = HistoricalTreeReader(tmp_path / "2005-202506")
    assert list(_collect(hist, "RB")) == ["RB2605"]


def test_prefix_collision_a_vs_ag(tmp_path: Path) -> None:
    """Product 'A' must not swallow AG/AL/AP files (full-stem regex, not startswith)."""
    write_daily_tree(
        tmp_path,
        [
            ContractSpec("DCE", "a2605", day_session_bars(D26, n=2)),
            ContractSpec("SHFE", "ag2606", day_session_bars(D26, n=2)),
        ],
    )
    reader = DailyTreeReader(tmp_path / "2026")
    assert list(_collect(reader, "A")) == ["A2605"]
    assert list(_collect(reader, "AG")) == ["AG2606"]


def test_sequence_column_dropped_and_12_13_schema_enforced(tmp_path: Path) -> None:
    write_daily_tree(tmp_path, [ContractSpec("SHFE", "rb2605", day_session_bars(D26, n=2))])
    frames = _collect(DailyTreeReader(tmp_path / "2026"), "RB")
    assert "sequence" not in frames["RB2605"].columns
    assert "eob" not in frames["RB2605"].columns
    assert "type" not in frames["RB2605"].columns


def test_schema_drift_on_bad_contract_code() -> None:
    with pytest.raises(SchemaDriftError, match="does not match"):
        split_contract("RB605", Path("x.csv"))  # 3-digit code
    with pytest.raises(SchemaDriftError, match="month"):
        split_contract("RB2613", Path("x.csv"))  # month 13


def test_split_contract_and_pseudo() -> None:
    assert split_contract("TA0702", Path("x")) == ("TA", 200702)
    assert split_contract("RB2605", Path("x")) == ("RB", 202605)
    assert is_pseudo_contract("rb9999") and is_pseudo_contract("RB8888")
    assert not is_pseudo_contract("RB2605")


def test_historical_range_pruning_skips_out_of_range_contracts(tmp_path: Path) -> None:
    write_historical_tree(
        tmp_path,
        [
            ContractSpec("SHFE", "RB1601", day_session_bars(dt.date(2015, 6, 1), n=2)),
            ContractSpec("SHFE", "RB2010", day_session_bars(D, n=2)),
        ],
    )
    reader = HistoricalTreeReader(
        tmp_path / "2005-202506", start=dt.date(2020, 1, 1), end=dt.date(2020, 12, 31)
    )
    assert list(_collect(reader, "RB")) == ["RB2010"]


def test_zero_volume_and_nan_rows_filtered(tmp_path: Path) -> None:
    bars = [
        Bar(bob=dt.datetime(2026, 1, 5, 9, 0), volume=10.0, close=3100.0),
        Bar(bob=dt.datetime(2026, 1, 5, 9, 1), volume=0.0, close=3100.0),  # padded
        Bar(bob=dt.datetime(2026, 1, 5, 9, 2), nan_row=True),  # literal nan
        Bar(bob=dt.datetime(2026, 1, 5, 9, 3), volume=5.0, close=3101.0),
    ]
    write_historical_tree(tmp_path, [ContractSpec("SHFE", "RB2605", bars)])
    frames = _collect(HistoricalTreeReader(tmp_path / "2005-202506"), "RB")
    df = frames["RB2605"]
    assert df.height == 2
    assert (df["volume"] > 0).all()
