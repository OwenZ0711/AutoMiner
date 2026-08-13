"""Calendar-inference tests on synthetic ingested stores."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data_fetcher.calendar import (
    CalendarConfig,
    build_calendar,
    load_session_table,
    load_trading_days,
    rle_runs,
    run_template_row,
    smooth_majority,
)
from futures_common.paths import FuturesPaths

EPOCH = dt.date(1970, 1, 1)


# ------------------------------------------------------------------ pure units


def test_smooth_majority_kills_single_day_hole() -> None:
    m = np.ones((30, 4), dtype=bool)
    m[10, 2] = False  # one-day liquidity hole
    sm = smooth_majority(m, 15)
    assert sm.all()


def test_smooth_majority_tracks_regime_change() -> None:
    m = np.zeros((40, 1), dtype=bool)
    m[20:, 0] = True  # bucket turns on at day 20
    sm = smooth_majority(m, 15)
    assert not sm[20, 0]  # lag while the trailing majority catches up
    assert sm[30:, 0].all()


def test_rle_runs_merges_short_runs() -> None:
    rows = np.array([[0]] * 20 + [[1]] * 3 + [[0]] * 20, dtype=bool)
    runs = rle_runs(rows, min_run=10)
    assert runs == [(0, 42)]  # 3-day blip absorbed, halves re-merged


def test_rle_runs_keeps_real_change() -> None:
    rows = np.array([[0]] * 20 + [[1]] * 25, dtype=bool)
    runs = rle_runs(rows, min_run=10)
    assert runs == [(0, 19), (20, 44)]
    assert run_template_row(rows, 20, 44)[0]


def test_rle_short_first_run_absorbed_forward() -> None:
    rows = np.array([[1]] * 4 + [[0]] * 30, dtype=bool)
    runs = rle_runs(rows, min_run=10)
    assert runs == [(0, 33)]
    assert not run_template_row(rows, 0, 33)[0]


# ------------------------------------------------------- synthetic store helper


def _write_contract(
    paths: FuturesPaths,
    product: str,
    contract: str,
    days: list[dt.date],
    intervals_by_day: dict[dt.date, list[tuple[dt.time, dt.time]]],
) -> None:
    """Write a minimal ingested parquet with one bar per minute per interval."""
    rows: list[dict[str, object]] = []
    for day in days:
        for lo, hi in intervals_by_day[day]:
            # bob calendar date: night bars ≥21:00 belong to the PREVIOUS calendar
            # day relative to trading day; keep it simple — bars stamped on the
            # trading day itself except the 21:00+ block, stamped the day before.
            cal = day - dt.timedelta(days=1) if lo >= dt.time(21, 0) else day
            t = dt.datetime.combine(cal, lo)
            end = (
                dt.datetime.combine(cal, hi)
                if hi > lo
                else dt.datetime.combine(cal + dt.timedelta(days=1), hi)
            )
            while t < end:
                rows.append(
                    {
                        "exchange": "SHFE",
                        "product": product,
                        "contract": contract,
                        "delivery_ym": 202001,
                        "open": 100.0,
                        "high": 100.0,
                        "low": 100.0,
                        "close": 100.0,
                        "amount": 1000.0,
                        "volume": 10,
                        "position": 100,
                        "bob": t,
                        "trading_date": day,
                    }
                )
                t += dt.timedelta(minutes=1)
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


DAY_ONLY = [
    (dt.time(9, 0), dt.time(10, 15)),
    (dt.time(10, 30), dt.time(11, 30)),
    (dt.time(13, 30), dt.time(15, 0)),
]
NIGHT_23 = [(dt.time(21, 0), dt.time(23, 0)), *DAY_ONLY]
NIGHT_01 = [(dt.time(21, 0), dt.time(1, 0)), *DAY_ONLY]


def _weekdays(start: dt.date, n: int, skip: set[dt.date] | None = None) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5 and (skip is None or d not in skip):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def test_night_end_change_detected_with_correct_boundary(tmp_paths: FuturesPaths) -> None:
    """Synthetic RB-2016-style change: night to 01:00, then to 23:00."""
    days = _weekdays(dt.date(2016, 1, 4), 90)
    switch = 45
    intervals = {d: (NIGHT_01 if i < switch else NIGHT_23) for i, d in enumerate(days)}
    _write_contract(tmp_paths, "RB", "RB1610", days, intervals)

    report = build_calendar(tmp_paths, products=["RB"], cfg=CalendarConfig(anchors_required=1))
    assert report.n_templates == 2
    table = load_session_table(tmp_paths)
    before = table.intervals_for("RB", days[10])
    after = table.intervals_for("RB", days[80])
    assert before[0] == (dt.time(21, 0), dt.time(1, 0))
    assert after[0] == (dt.time(21, 0), dt.time(23, 0))
    # boundary within smoothing lag of the true switch day
    runs = pl.read_parquet(tmp_paths.product_sessions).sort("date_start")
    boundary = runs["date_start"][1]
    true_switch = days[switch]
    assert abs((boundary - true_switch).days) <= 15


def test_covid_style_night_gap_becomes_own_run(tmp_paths: FuturesPaths) -> None:
    """A multi-month night suspension must appear as a no-night template run."""
    days = _weekdays(dt.date(2019, 11, 4), 130)
    gap = range(50, 110)  # ~3 months without night sessions
    intervals = {d: (DAY_ONLY if i in gap else NIGHT_23) for i, d in enumerate(days)}
    _write_contract(tmp_paths, "RB", "RB2010", days, intervals)

    build_calendar(tmp_paths, products=["RB"], cfg=CalendarConfig(anchors_required=1))
    table = load_session_table(tmp_paths)
    assert len(table.intervals_for("RB", days[70])) == 3  # day-only during gap
    assert len(table.intervals_for("RB", days[20])) == 4
    assert len(table.intervals_for("RB", days[120])) == 4


def test_preholiday_night_absence_is_exception_not_template(
    tmp_paths: FuturesPaths,
) -> None:
    days = _weekdays(dt.date(2020, 6, 1), 60)
    no_night = {days[30], days[31]}  # two isolated pre-holiday suspensions
    intervals = {d: (DAY_ONLY if d in no_night else NIGHT_23) for d in days}
    _write_contract(tmp_paths, "RB", "RB2101", days, intervals)

    report = build_calendar(tmp_paths, products=["RB"], cfg=CalendarConfig(anchors_required=1))
    assert report.n_templates == 1  # template unchanged
    exc = pl.read_parquet(tmp_paths.night_exceptions)
    assert set(exc["trading_date"].to_list()) == no_night


def test_trading_days_exclude_golden_week(tmp_paths: FuturesPaths) -> None:
    golden_week = {dt.date(2020, 10, d) for d in range(1, 9)}
    days = _weekdays(dt.date(2020, 9, 1), 60, skip=golden_week)
    intervals = {d: DAY_ONLY for d in days}
    _write_contract(tmp_paths, "RB", "RB2101", days, intervals)
    _write_contract(tmp_paths, "I", "I2101", days, intervals)

    build_calendar(tmp_paths, products=["RB", "I"], cfg=CalendarConfig(anchors_required=2))
    td = load_trading_days(tmp_paths)
    dates = {EPOCH + dt.timedelta(days=int(d)) for d in td}
    assert dates == set(days)
    assert not (dates & golden_week)
    assert all(d.weekday() < 5 for d in dates)


def test_override_escape_hatch(tmp_paths: FuturesPaths) -> None:
    days = _weekdays(dt.date(2020, 6, 1), 40)
    _write_contract(tmp_paths, "RB", "RB2101", days, {d: NIGHT_23 for d in days})
    override = {
        "product": "RB",
        "date_start": str(days[10]),
        "date_end": str(days[20]),
        "intervals": [["09:00", "10:15"], ["10:30", "11:30"], ["13:30", "15:00"]],
    }
    build_calendar(
        tmp_paths,
        products=["RB"],
        cfg=CalendarConfig(anchors_required=1, overrides=(override,)),
    )
    table = load_session_table(tmp_paths)
    assert len(table.intervals_for("RB", days[15])) == 3  # overridden: day-only
    assert len(table.intervals_for("RB", days[5])) == 4
    assert len(table.intervals_for("RB", days[25])) == 4


@pytest.mark.realdata
def test_real_acceptance_slice_calendar(tmp_path: Path) -> None:
    """Real check: build calendar over the ingested acceptance slice."""
    from futures_common.paths import default_paths

    paths = default_paths()
    if not paths.raw_root.is_dir():
        pytest.skip("acceptance slice not ingested")
    report = build_calendar(paths, products=["RB", "I", "TA"])
    assert report.n_trading_days > 400  # ~3 years
