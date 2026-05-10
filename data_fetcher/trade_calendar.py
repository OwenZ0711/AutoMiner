"""Trade-calendar cache + lookup helpers.

Pure functions over a `pd.DatetimeIndex`. The Source layer fetches the actual
calendar (e.g. via `ak.tool_trade_date_hist_sina`); this module only handles
persistence + standard date queries.

Named `trade_calendar` (not `calendar`) so it doesn't shadow the stdlib
`calendar` module.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl


def save_cache(cal: pd.DatetimeIndex, path: Path) -> None:
    """Write the calendar to parquet at `path`. Creates parent dirs as needed.

    Schema: a single column `trade_date` of date32. We strip time-of-day on
    write so consumers can rely on `cal[i].normalize() == cal[i]`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = [d.date() for d in pd.DatetimeIndex(cal).normalize()]
    pl.DataFrame({"trade_date": dates}, schema={"trade_date": pl.Date}).write_parquet(
        path
    )


def load_cached(path: Path, max_age_days: int = 7) -> pd.DatetimeIndex | None:
    """Return cached calendar if `path` exists and is younger than `max_age_days`.

    Returns None if the file is missing or stale; the caller should refetch.
    """
    if not path.exists():
        return None
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    if age.days > max_age_days:
        return None
    df = pl.read_parquet(path)
    # Polars Date -> pandas Timestamp via to_pandas (yields datetime64[ns]).
    series = df["trade_date"].to_pandas()
    return pd.DatetimeIndex(series).normalize()


def is_trading_day(cal: pd.DatetimeIndex, d: pd.Timestamp) -> bool:
    return pd.Timestamp(d).normalize() in cal


def next_trading_day(cal: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    """First trading day strictly after `d`. Raises IndexError past calendar end."""
    target = pd.Timestamp(d).normalize()
    idx = cal.searchsorted(target, side="right")
    if idx >= len(cal):
        raise IndexError(
            f"no trading day after {target.date()} in calendar (last is {cal[-1].date()})"
        )
    return cal[idx]


def previous_trading_day(cal: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    """Last trading day strictly before `d`. Raises IndexError before calendar start."""
    target = pd.Timestamp(d).normalize()
    idx = cal.searchsorted(target, side="left")
    if idx == 0:
        raise IndexError(
            f"no trading day before {target.date()} in calendar (first is {cal[0].date()})"
        )
    return cal[idx - 1]


def trading_days_between(
    cal: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """All trading days in [start, end] inclusive.

    Returns an empty index if start > end or the range falls entirely on
    non-trading days.
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if s > e:
        return pd.DatetimeIndex([])
    mask = (cal >= s) & (cal <= e)
    return cal[mask]
