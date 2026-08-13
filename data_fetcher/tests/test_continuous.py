"""Dominant-contract state machine + PIT adjustment tests."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

from data_fetcher.continuous import (
    build_continuous,
    dominance_spans,
    segment_of,
    select_dominant,
)
from futures_common.paths import FuturesPaths

D0 = dt.date(2020, 3, 2)  # Monday


def _daily(rows: list[tuple[str, int, str, int, int, float, int]]) -> pl.DataFrame:
    """rows: (contract, day_offset_from_D0, _, day_volume, last_position, day_close, delivery)."""
    return pl.DataFrame(
        {
            "contract": [r[0] for r in rows],
            "trading_date": [D0 + dt.timedelta(days=r[1]) for r in rows],
            "delivery_ym": pl.Series([r[6] for r in rows], dtype=pl.Int32),
            "day_volume": pl.Series([r[3] for r in rows], dtype=pl.Int64),
            "last_position": pl.Series([r[4] for r in rows], dtype=pl.Int64),
            "day_close": pl.Series([r[5] for r in rows], dtype=pl.Float32),
        }
    )


def test_two_day_confirmation_blocks_one_day_whipsaw() -> None:
    rows = []
    # RB05 dominant; RB10 spikes above it for ONE day (day 2), then falls back
    for d in range(6):
        rows.append(("RB05", d, "", 1000 if d != 2 else 800, 500, 100.0, 202005))
        rows.append(("RB10", d, "", 900 if d != 2 else 1200, 400, 105.0, 202010))
    events = select_dominant(_daily(rows), confirm_days=2)
    assert [e.reason for e in events] == ["bootstrap"]
    assert events[0].new_contract == "RB05"


def test_confirmed_switch_fires_next_day_with_correct_factor() -> None:
    rows = []
    for d in range(6):
        vol05 = 1000 if d < 3 else 500
        vol10 = 800 if d < 3 else 1500  # overtakes on days 3 and 4 → switch day 5
        rows.append(("RB05", d, "", vol05, 500, 100.0, 202005))
        rows.append(("RB10", d, "", vol10, 400, 110.0, 202010))
    events = select_dominant(_daily(rows), confirm_days=2)
    assert len(events) == 2
    roll = events[1]
    assert roll.reason == "volume"
    assert roll.old_contract == "RB05" and roll.new_contract == "RB10"
    assert roll.trading_date == D0 + dt.timedelta(days=5)  # effective NEXT day
    assert roll.factor == 110.0 / 100.0


def test_switch_only_forward_never_rolls_to_nearer_month() -> None:
    rows = []
    for d in range(6):
        rows.append(("RB10", d, "", 1000, 500, 100.0, 202010))
        # nearer delivery month out-trades — must be ignored
        rows.append(("RB05", d, "", 5000, 900, 99.0, 202005))
    events = select_dominant(_daily(rows), confirm_days=2)
    # bootstrap picks RB05 (max volume, no restriction on day one)...
    assert events[0].new_contract == "RB05"
    # ...after which RB10 (later delivery) may take over, but never backwards
    for ev in events[1:]:
        assert ev.new_contract != "RB05"


def test_tie_break_oi_then_nearer_month() -> None:
    rows = [
        ("RB05", 0, "", 1000, 900, 100.0, 202005),
        ("RB10", 0, "", 1000, 400, 100.0, 202010),
    ]
    events = select_dominant(_daily(rows), confirm_days=2)
    assert events[0].new_contract == "RB05"  # same volume, larger OI wins


def test_segment_restart_after_hole() -> None:
    rows = [
        ("RB2510", 0, "", 1000, 500, 100.0, 202510),
        ("RB2605", 0, "", 100, 50, 200.0, 202605),
    ]
    daily = _daily(rows)
    # move dates: one day before the hole, one after
    daily = pl.concat(
        [
            daily.with_columns(pl.lit(dt.date(2025, 7, 4)).alias("trading_date")),
            daily.with_columns(pl.lit(dt.date(2026, 1, 5)).alias("trading_date")),
        ]
    )
    events = select_dominant(daily, confirm_days=2)
    assert [e.reason for e in events] == ["bootstrap", "segment_restart"]
    assert math.isnan(events[1].factor)
    assert segment_of(events[0].trading_date) == 0
    assert segment_of(events[1].trading_date) == 1


def test_pit_adjustment_is_segment_start_anchored() -> None:
    """adj starts at 1.0 and is divided by each roll factor — never rescaled later."""
    events = select_dominant(
        _daily(
            [
                *[("A", d, "", 1000 if d < 3 else 100, 1, 100.0, 202001) for d in range(6)],
                *[("B", d, "", 900 if d < 3 else 2000, 1, 120.0, 202002) for d in range(6)],
            ]
        ),
        confirm_days=2,
    )
    spans = dominance_spans(events, D0 + dt.timedelta(days=10))
    assert spans[0].adj_factor == 1.0  # PIT anchor: segment START
    assert spans[1].adj_factor == 1.0 / (120.0 / 100.0)
    # continuity: old raw close * A_old == new raw close * A_new at the roll
    assert np.isclose(100.0 * spans[0].adj_factor, 120.0 * spans[1].adj_factor)


def _write_contract_bars(
    paths: FuturesPaths,
    product: str,
    contract: str,
    delivery: int,
    days: list[dt.date],
    price: float,
    volume: int,
) -> None:
    rows = []
    for day in days:
        for minute in range(3):
            rows.append(
                {
                    "exchange": "SHFE",
                    "product": product,
                    "contract": contract,
                    "delivery_ym": delivery,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "amount": price * volume * 10,
                    "volume": volume,
                    "position": 100,
                    "bob": dt.datetime.combine(day, dt.time(9, minute)),
                    "trading_date": day,
                }
            )
    df = pl.DataFrame(rows).with_columns(
        pl.col("bob").dt.replace_time_zone("Asia/Shanghai"),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Int64),
        pl.col("position").cast(pl.Int64),
        pl.col("delivery_ym").cast(pl.Int32),
    )
    out = paths.raw_contract("SHFE", product, contract)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort("bob").write_parquet(out)


def test_stitched_series_has_no_roll_return_spike(tmp_paths: FuturesPaths) -> None:
    """A 9.5%-style roll gap (JD 2026-02 case) must vanish in adjusted prices."""
    days = [D0 + dt.timedelta(days=d) for d in range(10)]
    # contract A trades days 0-5 at 100; B trades all days at 109.5 and
    # overtakes from day 4 → confirmation days 4,5 → switch effective day 6
    _write_contract_bars(tmp_paths, "JD", "JD2001", 202001, days[:7], 100.0, 1000)
    _write_contract_bars(tmp_paths, "JD", "JD2003", 202003, days, 109.5, 500)
    big = pl.read_parquet(tmp_paths.raw_contract("SHFE", "JD", "JD2003"))
    big = big.with_columns(
        pl.when(pl.col("trading_date") >= days[4])
        .then(pl.lit(5000, dtype=pl.Int64))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    big.write_parquet(tmp_paths.raw_contract("SHFE", "JD", "JD2003"))

    report = build_continuous(tmp_paths, products=["JD"])
    assert report.rolls == 1

    year = pl.read_parquet(tmp_paths.continuous_year("JD", 2020)).sort("bob")
    adj = year["close"].cast(pl.Float64) * year["adj_factor"]
    rets = np.diff(np.log(adj.to_numpy()))
    assert np.abs(rets).max() < 0.01  # no 9.5% splice jump
    assert year.filter(pl.col("roll_flag")).height == 1  # exactly one roll bar
    roll_bar = year.filter(pl.col("roll_flag"))
    assert roll_bar["contract"][0] == "JD2003"
    # raw recoverable exactly
    raw_back = (adj / year["adj_factor"]).cast(pl.Float32)
    assert (raw_back == year["close"]).all()


def test_roll_calendar_written(tmp_paths: FuturesPaths) -> None:
    days = [D0 + dt.timedelta(days=d) for d in range(8)]
    _write_contract_bars(tmp_paths, "RB", "RB2005", 202005, days, 100.0, 1000)
    _write_contract_bars(tmp_paths, "RB", "RB2010", 202010, days[3:], 105.0, 3000)
    build_continuous(tmp_paths, products=["RB"])
    rc = pl.read_parquet(tmp_paths.roll_calendar)
    assert rc.filter(pl.col("reason") == "volume").height == 1
    assert rc.filter(pl.col("reason") == "bootstrap").height == 1
