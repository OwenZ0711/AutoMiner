"""Bootstrap trading-day set + the trading-date rule (trap 3).

The trading-date rule needs the trading-day set, which the full calendar stage
(L1a) derives from ingested bars — a circular dependency. The bootstrap pass
breaks it: scan a small set of liquid anchor products, keep bars whose local
time is in the day-session window [08:30, 16:00), and call a date a trading
day when at least ``anchors_required`` anchors printed such bars. L1a later
recomputes the full set and asserts the bootstrap is a subset.

Trading-date rule (proposal L0, exact):
- bob time ∈ [21:00, 24:00) → the next trading day strictly AFTER bob's date;
- bob time ∈ [00:00, 03:00) → the first trading day ≥ bob's date;
- otherwise bob's own date (dropped + counted if that's not a trading day —
  catches exchange-halt garbage).
This absorbs the 2026 calendar-keyed day folders and Saturday night-tails.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

from data_fetcher.readers.base import RawReader
from futures_common.manifest import config_hash
from futures_common.paths import FuturesPaths

_EPOCH = dt.date(1970, 1, 1)

# Buffer around the requested range. Contract parquets hold the FULL contract
# lifetime (~14 months around delivery), so the calendar must cover every bar
# of every contract whose lifetime intersects [start, end] — not just the
# slice itself. 460 days ≈ 15 months on each side.
BUFFER_BEFORE_DAYS = 460
BUFFER_AFTER_DAYS = 460

_DAY_SESSION_START_MIN = 8 * 60 + 30
_DAY_SESSION_END_MIN = 16 * 60


def _to_epoch_day(d: dt.date) -> int:
    return (d - _EPOCH).days


@dataclass(frozen=True, slots=True)
class TradingDays:
    """Sorted trading days as epoch-day int32 array."""

    days: np.ndarray  # int32, sorted ascending

    def __len__(self) -> int:
        return int(self.days.size)

    def contains(self, d: dt.date) -> bool:
        day = _to_epoch_day(d)
        idx = int(np.searchsorted(self.days, day))
        return idx < self.days.size and int(self.days[idx]) == day


def build_bootstrap_trading_days(
    readers: Sequence[RawReader],
    *,
    anchors: Sequence[str],
    start: dt.date,
    end: dt.date,
    anchors_required: int = 3,
    paths: FuturesPaths | None = None,
) -> TradingDays:
    """Infer the trading-day set for [start - buffer, end + buffer] from anchor products.

    Cached under ``calendar/trading_days_bootstrap.parquet`` keyed by
    (anchors, buffered range, code version); recomputed when the key changes.
    """
    lo = start - dt.timedelta(days=BUFFER_BEFORE_DAYS)
    hi = end + dt.timedelta(days=BUFFER_AFTER_DAYS)

    available = sorted(
        {a.upper() for a in anchors} & {p for r in readers for p in r.discover(None)}
    )
    required = min(anchors_required, max(1, len(available)))
    if not available:
        raise ValueError("no anchor products present in the raw trees")

    cache_key = config_hash(
        {"anchors": available, "lo": lo, "hi": hi, "required": required, "v": 1}
    )
    if paths is not None:
        cached = _load_cache(paths, cache_key)
        if cached is not None:
            return cached

    logger.info(
        "bootstrap trading days: anchors={} range=[{}, {}] required={}",
        ",".join(available),
        lo,
        hi,
        required,
    )
    counts: dict[int, set[str]] = {}
    for anchor in available:
        anchor_dates: set[int] = set()
        for reader in readers:
            for batch in reader.iter_product(anchor):
                frame = (
                    batch.lf.select(
                        pl.col("bob").dt.date().alias("date"),
                        (
                            pl.col("bob").dt.hour().cast(pl.Int32) * 60
                            + pl.col("bob").dt.minute().cast(pl.Int32)
                        ).alias("min"),
                    )
                    .filter(
                        pl.col("min").is_between(
                            _DAY_SESSION_START_MIN, _DAY_SESSION_END_MIN, closed="left"
                        )
                        & pl.col("date").is_between(lo, hi, closed="both")
                    )
                    .select(pl.col("date").unique())
                    .collect(engine="streaming")
                )
                anchor_dates.update(frame["date"].cast(pl.Int32).to_list())
        for day in anchor_dates:
            counts.setdefault(day, set()).add(anchor)

    days = np.array(
        sorted(day for day, who in counts.items() if len(who) >= required), dtype=np.int32
    )
    if days.size == 0:
        raise ValueError(f"bootstrap found no trading days in [{lo}, {hi}]")
    out = TradingDays(days=days)
    if paths is not None:
        _save_cache(paths, cache_key, out)
    return out


def _load_cache(paths: FuturesPaths, key: str) -> TradingDays | None:
    meta_path = paths.trading_days_bootstrap.with_suffix(".json")
    if not (paths.trading_days_bootstrap.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    if meta.get("key") != key:
        return None
    frame = pl.read_parquet(paths.trading_days_bootstrap)
    return TradingDays(days=frame["trading_date"].cast(pl.Int32).to_numpy().astype(np.int32))


def _save_cache(paths: FuturesPaths, key: str, td: TradingDays) -> None:
    paths.trading_days_bootstrap.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"trading_date": pl.Series(td.days, dtype=pl.Int32).cast(pl.Date)}).write_parquet(
        paths.trading_days_bootstrap
    )
    paths.trading_days_bootstrap.with_suffix(".json").write_text(json.dumps({"key": key}))


def assign_trading_date(df: pl.DataFrame, td: TradingDays) -> tuple[pl.DataFrame, int]:
    """Apply the trading-date rule; returns (frame + trading_date, n dropped unmapped)."""
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Date).alias("trading_date")), 0

    date_days = df["bob"].dt.date().cast(pl.Int32).to_numpy()
    # dt.hour() is Int8 — cast before arithmetic or 21*60 overflows
    minutes = (
        df["bob"].dt.hour().cast(pl.Int32) * 60 + df["bob"].dt.minute().cast(pl.Int32)
    ).to_numpy()
    days = td.days

    idx_after = np.searchsorted(days, date_days, side="right")  # first td > date
    idx_geq = np.searchsorted(days, date_days, side="left")  # first td >= date

    is_late = minutes >= 21 * 60
    is_early = minutes < 3 * 60
    is_day = ~(is_late | is_early)

    idx = np.where(is_late, idx_after, idx_geq)
    in_range = idx < days.size
    mapped = np.where(in_range, days[np.minimum(idx, days.size - 1)], -1).astype(np.int64)

    # day bars keep their own date and must BE a trading day
    day_is_trading = in_range & (mapped == date_days)
    valid = np.where(is_day, day_is_trading, in_range)
    trading = np.where(is_day, date_days, mapped)

    out = df.with_columns(
        pl.Series("trading_date", trading.astype(np.int32)).cast(pl.Date),
        pl.Series("_td_valid", valid),
    )
    dropped = int((~valid).sum())
    if dropped:
        out = out.filter(pl.col("_td_valid"))
    return out.drop("_td_valid"), dropped
