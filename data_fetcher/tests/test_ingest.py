"""Ingest tests: trading-date rule, cross-tree merge, idempotency, byte-identity."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data_fetcher.ingest import ingest
from data_fetcher.tests.conftest import (
    Bar,
    ContractSpec,
    day_session_bars,
    write_daily_tree,
    write_historical_tree,
)
from data_fetcher.trading_days import TradingDays, assign_trading_date
from futures_common.paths import FuturesPaths

_EPOCH = dt.date(1970, 1, 1)


def _days(*dates: dt.date) -> TradingDays:
    return TradingDays(days=np.array(sorted((d - _EPOCH).days for d in dates), dtype=np.int32))


# ------------------------------------------------------------- trading-date rule


@pytest.mark.parametrize(
    ("bob", "expected"),
    [
        # Friday 21:30 night bar → Monday
        (dt.datetime(2020, 3, 6, 21, 30), dt.date(2020, 3, 9)),
        # Saturday 01:30 tail (sits in Saturday's calendar date) → Monday
        (dt.datetime(2020, 3, 7, 1, 30), dt.date(2020, 3, 9)),
        # Ordinary day bar keeps its date
        (dt.datetime(2020, 3, 6, 10, 0), dt.date(2020, 3, 6)),
        # Pre-holiday night (Thu 21:05, Friday is a holiday) → next trading day Monday
        (dt.datetime(2020, 3, 5, 21, 5), dt.date(2020, 3, 6)),
        # 00:30 on a trading day belongs to that same trading day
        (dt.datetime(2020, 3, 6, 0, 30), dt.date(2020, 3, 6)),
    ],
)
def test_trading_date_rule(bob: dt.datetime, expected: dt.date) -> None:
    td = _days(dt.date(2020, 3, 5), dt.date(2020, 3, 6), dt.date(2020, 3, 9))
    df = pl.DataFrame(
        {"bob": [bob]},
        schema={"bob": pl.Datetime("us")},
    ).with_columns(pl.col("bob").dt.replace_time_zone("Asia/Shanghai"))
    out, dropped = assign_trading_date(df, td)
    assert dropped == 0
    assert out["trading_date"][0] == expected


def test_day_bar_on_non_trading_day_dropped() -> None:
    td = _days(dt.date(2020, 3, 6))
    df = pl.DataFrame(
        {"bob": [dt.datetime(2020, 3, 7, 10, 0)]},  # Saturday day bar: garbage
        schema={"bob": pl.Datetime("us")},
    ).with_columns(pl.col("bob").dt.replace_time_zone("Asia/Shanghai"))
    out, dropped = assign_trading_date(df, td)
    assert dropped == 1 and out.height == 0


def test_night_bar_past_calendar_end_dropped() -> None:
    td = _days(dt.date(2020, 3, 6))
    df = pl.DataFrame(
        {"bob": [dt.datetime(2020, 3, 6, 21, 30)]},  # maps beyond last known td
        schema={"bob": pl.Datetime("us")},
    ).with_columns(pl.col("bob").dt.replace_time_zone("Asia/Shanghai"))
    out, dropped = assign_trading_date(df, td)
    assert dropped == 1 and out.height == 0


# ------------------------------------------------------------------- full ingest


def _paths(tmp_path: Path) -> FuturesPaths:
    return FuturesPaths(data_root=tmp_path / "data", results_root=tmp_path / "results")


def _sha_all(paths: FuturesPaths) -> dict[str, str]:
    """Content hashes of all CONTRACT parquets (the manifest carries a
    written_at timestamp by design, so it is excluded from byte-identity)."""
    return {
        str(f.relative_to(paths.raw_root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(paths.raw_root.rglob("*.parquet"))
        if f.name != "_manifest.parquet"
    }


def _make_trees(tmp_path: Path) -> tuple[Path, Path]:
    monday = dt.date(2026, 1, 5)
    tuesday = dt.date(2026, 1, 6)
    # Historical side: RB2605 day bars on 2025-05-15 won't be used (out of range
    # for the 2026 window) — instead give it bars mapping into the window is
    # impossible for historical tree (ends 2025-07); so cross-tree merge is
    # tested with an overlapping-window ingest below using only day bars.
    hist = write_historical_tree(
        tmp_path,
        [
            ContractSpec("SHFE", "RB2605", day_session_bars(monday, n=3, close=3100.0)),
            ContractSpec("SHFE", "AU2606", day_session_bars(monday, n=3, close=750.0)),
        ],
    )
    # 2026 side: same contract RB2605 — Monday duplicate bar + Tuesday fresh bars,
    # plus a Friday-night tail in the *Saturday* folder mapping to Monday 01-12.
    friday_night_tail = [
        Bar(bob=dt.datetime(2026, 1, 10, 0, 30), close=3120.0, volume=7.0),
    ]
    daily = write_daily_tree(
        tmp_path,
        [
            ContractSpec(
                "SHFE",
                "rb2605",
                [
                    *day_session_bars(monday, n=1, close=9999.0),  # duplicate of hist bar 1
                    *day_session_bars(tuesday, n=2, close=3105.0),
                    *friday_night_tail,
                ],
            ),
        ],
    )
    return hist, daily


def _trading_days_2026() -> TradingDays:
    return _days(
        dt.date(2026, 1, 5),
        dt.date(2026, 1, 6),
        dt.date(2026, 1, 7),
        dt.date(2026, 1, 8),
        dt.date(2026, 1, 9),
        dt.date(2026, 1, 12),
    )


def test_ingest_cross_tree_merge_dedupe_and_saturday_tail(tmp_path: Path) -> None:
    hist, daily = _make_trees(tmp_path)
    paths = _paths(tmp_path)
    report = ingest(
        paths,
        raw_historical=hist,
        raw_2026=daily,
        products=["RB"],
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 1, 31),
        trading_days=_trading_days_2026(),
    )
    assert report.contracts_written == 1
    df = pl.read_parquet(paths.raw_contract("SHFE", "RB", "RB2605"))
    # 3 hist Monday bars + 2 Tuesday bars + 1 Saturday tail; the 2026 duplicate
    # of Monday 09:00 lost to the historical row (keep-first)
    assert df.height == 6
    monday_first = df.filter(pl.col("bob").dt.date() == dt.date(2026, 1, 5)).sort("bob")
    assert monday_first["close"][0] == pytest.approx(3100.0)  # historical won
    tail = df.filter(pl.col("bob").dt.date() == dt.date(2026, 1, 10))
    assert tail["trading_date"].to_list() == [dt.date(2026, 1, 12)]  # Saturday → Monday
    assert df["bob"].is_sorted()


def test_ingest_idempotent_second_run_writes_nothing(tmp_path: Path) -> None:
    hist, daily = _make_trees(tmp_path)
    paths = _paths(tmp_path)
    kw = dict(
        raw_historical=hist,
        raw_2026=daily,
        products=None,
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 1, 31),
        trading_days=_trading_days_2026(),
    )
    r1 = ingest(paths, **kw)  # type: ignore[arg-type]
    hashes1 = _sha_all(paths)
    r2 = ingest(paths, **kw)  # type: ignore[arg-type]
    assert r1.contracts_written >= 1
    assert r2.contracts_written == 0
    assert r2.contracts_skipped == r1.contracts_written + r1.contracts_empty
    assert _sha_all(paths) == hashes1


def test_ingest_rebuild_on_source_touch_is_byte_identical(tmp_path: Path) -> None:
    hist, daily = _make_trees(tmp_path)
    paths = _paths(tmp_path)
    kw = dict(
        raw_historical=hist,
        raw_2026=daily,
        products=["RB"],
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 1, 31),
        trading_days=_trading_days_2026(),
    )
    ingest(paths, **kw)  # type: ignore[arg-type]
    hashes1 = _sha_all(paths)

    # touch a source file (same content, new mtime) → contract must rebuild...
    src = next((hist / "SHFE" / "RB").glob("*.csv"))
    src.touch()
    r2 = ingest(paths, **kw)  # type: ignore[arg-type]
    assert r2.contracts_written == 1
    # ...to a byte-identical parquet (determinism convention)
    assert _sha_all(paths) == hashes1


def test_ingest_force_rebuilds_everything(tmp_path: Path) -> None:
    hist, daily = _make_trees(tmp_path)
    paths = _paths(tmp_path)
    kw = dict(
        raw_historical=hist,
        raw_2026=daily,
        products=None,
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 1, 31),
        trading_days=_trading_days_2026(),
    )
    r1 = ingest(paths, **kw)  # type: ignore[arg-type]
    r2 = ingest(paths, force=True, **kw)  # type: ignore[arg-type]
    assert r2.contracts_written + r2.contracts_empty >= r1.contracts_written


def test_ingest_range_filter_drops_out_of_window_contract(tmp_path: Path) -> None:
    hist, daily = _make_trees(tmp_path)
    paths = _paths(tmp_path)
    report = ingest(
        paths,
        raw_historical=hist,
        raw_2026=daily,
        products=["AU"],
        start=dt.date(2027, 1, 1),  # far in the future: nothing survives
        end=dt.date(2027, 1, 31),
        trading_days=_days(dt.date(2027, 1, 4)),
    )
    assert report.contracts_written == 0
