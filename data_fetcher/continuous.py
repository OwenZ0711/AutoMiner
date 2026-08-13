"""L1b: dominant-contract selection + PIT forward-adjusted continuous series.

Dominant rules (proposal, exact):
- candidate(day) = argmax day-volume; ties → larger open interest, then nearer
  delivery month;
- **switch-only-forward**: candidates with an earlier delivery month than the
  current dominant are ignored;
- **2-consecutive-day confirmation** (anti-whipsaw): the same challenger must
  out-trade the incumbent on `confirm_days` consecutive trading days; the
  switch takes effect the NEXT trading day (at its open);
- bootstrap (first listed day) picks that day's max-volume contract outright;
- the 2025 data hole restarts the chain cold (`segment_restart`); nothing
  bridges segments.

Adjustment (LOOK-AHEAD SAFETY §1 — PIT, segment-START anchored):
roll factor f = day_close_new(t) / day_close_old(t) at confirmation day t;
the cumulative adjustment is divided at each roll:
``A_after = A_before / f`` with A = 1.0 at segment start, and
``adjusted = raw * A``. Every adjusted level therefore depends only on rolls
at-or-before the bar — no future roll can rescale the past — while the
adjusted series stays continuous across rolls. Prices stay raw on disk; raw is
always recoverable as ``adjusted / adj_factor``.
"""

from __future__ import annotations

import datetime as dt
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from loguru import logger

from futures_common.paths import FuturesPaths

HOLE_LAST_DAY = dt.date(2025, 7, 4)  # segment 0 ends here; 1 starts 2026-01-05

DAY_CLOSE_WINDOW = (8 * 60 + 30, 15 * 60 + 15)  # bob minutes for "day close"


@dataclass(frozen=True, slots=True)
class RollEvent:
    trading_date: dt.date  # first trading day the NEW contract is dominant
    old_contract: str | None
    new_contract: str
    factor: float  # close_new/close_old at confirmation day; NaN at chain starts
    reason: str  # "volume" | "bootstrap" | "segment_restart"
    day_volume_old: float
    day_volume_new: float


@dataclass(slots=True)
class ContinuousReport:
    products: list[str]
    rolls: int
    years_written: int

    def as_dict(self) -> dict[str, object]:
        return {
            "products": self.products,
            "rolls": self.rolls,
            "years_written": self.years_written,
        }


def segment_of(day: dt.date) -> int:
    return 0 if day <= HOLE_LAST_DAY else 1


# ------------------------------------------------------------------ daily stats


def build_daily_stats(paths: FuturesPaths, product: str) -> pl.DataFrame:
    """Per (contract, trading_date): day volume, last OI, 15:00-ish day close."""
    files = sorted(paths.raw_root.glob(f"exchange=*/product={product}/*.parquet"))
    if not files:
        return pl.DataFrame(
            schema={
                "contract": pl.Utf8,
                "trading_date": pl.Date,
                "delivery_ym": pl.Int32,
                "day_volume": pl.Int64,
                "last_position": pl.Int64,
                "day_close": pl.Float32,
            }
        )
    minute = pl.col("bob").dt.hour().cast(pl.Int32) * 60 + pl.col("bob").dt.minute().cast(pl.Int32)
    lf = (
        pl.scan_parquet([str(f) for f in files])
        .select(
            "contract",
            "trading_date",
            "delivery_ym",
            "volume",
            "position",
            "close",
            "bob",
            minute.alias("minute"),
        )
        .sort("bob")
        .group_by("contract", "trading_date", maintain_order=True)
        .agg(
            pl.col("delivery_ym").first(),
            pl.col("volume").sum().alias("day_volume"),
            pl.col("position").last().alias("last_position"),
            pl.col("close")
            .filter(pl.col("minute").is_between(DAY_CLOSE_WINDOW[0], DAY_CLOSE_WINDOW[1] - 1))
            .last()
            .alias("day_close"),
        )
    )
    df = lf.collect(engine="streaming").sort("trading_date", "contract")
    out = paths.daily_stats(product)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, out)
    return df


# --------------------------------------------------------------- state machine


def select_dominant(daily: pl.DataFrame, *, confirm_days: int = 2) -> list[RollEvent]:
    """Pure state machine over daily stats rows (a few thousand rows)."""
    if daily.is_empty():
        return []
    events: list[RollEvent] = []

    dates: list[dt.date] = sorted(daily["trading_date"].unique().to_list())
    by_date: dict[dt.date, dict[str, tuple[int, int, float | None, int]]] = {}
    for row in daily.iter_rows(named=True):
        by_date.setdefault(row["trading_date"], {})[row["contract"]] = (
            int(row["day_volume"]),
            int(row["last_position"]),
            row["day_close"],
            int(row["delivery_ym"]),
        )

    current: str | None = None
    current_delivery = -1
    challenger: str | None = None
    streak = 0
    segment = -1

    def day_close_walkback(contract: str, day_idx: int, max_back: int = 3) -> float | None:
        for k in range(day_idx, max(day_idx - max_back, -1), -1):
            info = by_date[dates[k]].get(contract)
            if info is not None and info[2] is not None:
                return float(info[2])
        return None

    for i, day in enumerate(dates):
        contracts = by_date[day]
        seg = segment_of(day)
        if seg != segment:
            # chain (re)start: bootstrap on this day's max volume
            segment = seg
            best = _best_candidate(contracts, min_delivery=-1)
            assert best is not None  # every date row set is non-empty
            current, current_delivery = best, contracts[best][3]
            challenger, streak = None, 0
            events.append(
                RollEvent(
                    trading_date=day,
                    old_contract=None,
                    new_contract=best,
                    factor=math.nan,
                    reason="bootstrap" if not events else "segment_restart",
                    day_volume_old=0.0,
                    day_volume_new=float(contracts[best][0]),
                )
            )
            continue

        assert current is not None
        cur_info = contracts.get(current)
        cur_volume = cur_info[0] if cur_info is not None else 0
        cand = _best_candidate(contracts, min_delivery=current_delivery, exclude=current)
        if cand is not None and contracts[cand][0] > cur_volume:
            streak = streak + 1 if cand == challenger else 1
            challenger = cand
        else:
            challenger, streak = None, 0

        if challenger is not None and streak >= confirm_days:
            old_close = day_close_walkback(current, i)
            new_close = day_close_walkback(challenger, i)
            if old_close is None or new_close is None or old_close <= 0:
                factor = math.nan
                logger.warning(
                    "roll {}→{} on {}: missing day close, factor=NaN (treated as 1.0)",
                    current,
                    challenger,
                    day,
                )
            else:
                factor = new_close / old_close
            effective = dates[i + 1] if i + 1 < len(dates) else None
            if effective is not None and segment_of(effective) == segment:
                events.append(
                    RollEvent(
                        trading_date=effective,
                        old_contract=current,
                        new_contract=challenger,
                        factor=factor,
                        reason="volume",
                        day_volume_old=float(cur_volume),
                        day_volume_new=float(contracts[challenger][0]),
                    )
                )
                current, current_delivery = challenger, contracts[challenger][3]
            challenger, streak = None, 0

    return events


def _best_candidate(
    contracts: dict[str, tuple[int, int, float | None, int]],
    *,
    min_delivery: int,
    exclude: str | None = None,
) -> str | None:
    """argmax volume; ties → larger OI, then nearer (smaller) delivery month."""
    best: str | None = None
    best_key: tuple[int, int, int] | None = None
    for name, (vol, oi, _close, delivery) in contracts.items():
        if name == exclude or delivery < min_delivery:
            continue
        key = (vol, oi, -delivery)
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best


# -------------------------------------------------------------------- stitching


@dataclass(frozen=True, slots=True)
class _Span:
    contract: str
    date_start: dt.date
    date_end: dt.date  # inclusive
    adj_factor: float
    roll_flag: bool  # True → first day of a volume-roll dominance
    segment_id: int


def dominance_spans(events: Sequence[RollEvent], last_date: dt.date) -> list[_Span]:
    spans: list[_Span] = []
    adj = 1.0
    for k, ev in enumerate(events):
        if ev.reason in ("bootstrap", "segment_restart"):
            adj = 1.0
        else:
            f = ev.factor if math.isfinite(ev.factor) and ev.factor > 0 else 1.0
            adj = adj / f
        end = (
            events[k + 1].trading_date - dt.timedelta(days=1) if k + 1 < len(events) else last_date
        )
        spans.append(
            _Span(
                contract=ev.new_contract,
                date_start=ev.trading_date,
                date_end=end,
                adj_factor=adj,
                roll_flag=ev.reason == "volume",
                segment_id=segment_of(ev.trading_date),
            )
        )
    return spans


def build_continuous(
    paths: FuturesPaths,
    *,
    products: Sequence[str],
    confirm_days: int = 2,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> ContinuousReport:
    """Dominance is selected over trading dates in [start, end] ONLY: the raw
    store is complete (all contracts present) exactly for lifetimes that
    intersect the ingested range — outside it, partial contract sets would
    elect false dominants (thin far months)."""
    all_events: list[tuple[str, RollEvent]] = []
    years_written = 0

    for product in [p.upper() for p in products]:
        daily = build_daily_stats(paths, product)
        if start is not None:
            daily = daily.filter(pl.col("trading_date") >= start)
        if end is not None:
            daily = daily.filter(pl.col("trading_date") <= end)
        if daily.is_empty():
            logger.warning("continuous: no ingested bars for {} — skipped", product)
            continue
        events = select_dominant(daily, confirm_days=confirm_days)
        all_events.extend((product, ev) for ev in events)
        last_date = daily["trading_date"].max()
        spans = dominance_spans(events, last_date)  # type: ignore[arg-type]

        years = sorted({d.year for s in spans for d in (s.date_start, s.date_end)})
        for year in years:
            y0, y1 = dt.date(year, 1, 1), dt.date(year, 12, 31)
            frames: list[pl.DataFrame] = []
            for span in spans:
                lo, hi = max(span.date_start, y0), min(span.date_end, y1)
                if lo > hi:
                    continue
                contract_file = _find_contract_file(paths, product, span.contract)
                if contract_file is None:
                    continue
                df = (
                    pl.scan_parquet(contract_file)
                    .filter(pl.col("trading_date").is_between(lo, hi))
                    .select(
                        "bob",
                        "trading_date",
                        "contract",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "position",
                    )
                    .collect(engine="streaming")
                )
                if df.is_empty():
                    continue
                first_flag_date = span.date_start if span.roll_flag else None
                df = df.with_columns(
                    pl.lit(span.adj_factor, dtype=pl.Float64).alias("adj_factor"),
                    (pl.col("trading_date") == first_flag_date).fill_null(False).alias("_roll_day")
                    if first_flag_date is not None
                    else pl.lit(False).alias("_roll_day"),
                    pl.lit(span.segment_id, dtype=pl.Int8).alias("segment_id"),
                )
                frames.append(df)
            if not frames:
                continue
            out = (
                pl.concat(frames, how="vertical")
                .sort("bob")
                .with_columns(
                    # roll_flag only on the FIRST bar of the switch-effective day
                    (
                        pl.col("_roll_day")
                        & (pl.col("bob") == pl.col("bob").min().over("trading_date"))
                    ).alias("roll_flag")
                )
                .drop("_roll_day")
            )
            path = paths.continuous_year(product, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            out.write_parquet(tmp, compression="zstd")
            os.replace(tmp, path)
            years_written += 1
        logger.info("continuous {}: {} rolls, {} spans", product, len(events), len(spans))

    roll_rows = [
        {
            "product": prod,
            "trading_date": ev.trading_date,
            "old_contract": ev.old_contract,
            "new_contract": ev.new_contract,
            "factor": ev.factor,
            "reason": ev.reason,
            "day_volume_old": ev.day_volume_old,
            "day_volume_new": ev.day_volume_new,
        }
        for prod, ev in all_events
    ]
    if roll_rows:
        new_rows = pl.DataFrame(roll_rows)
        if paths.roll_calendar.exists():
            # merge: keep other products' rows, replace the rebuilt products'
            existing = pl.read_parquet(paths.roll_calendar).filter(
                ~pl.col("product").is_in(new_rows["product"].implode())
            )
            new_rows = pl.concat([existing, new_rows], how="vertical")
        paths.roll_calendar.parent.mkdir(parents=True, exist_ok=True)
        tmp = paths.roll_calendar.with_suffix(".parquet.tmp")
        new_rows.sort("product", "trading_date").write_parquet(tmp)
        os.replace(tmp, paths.roll_calendar)

    return ContinuousReport(
        products=[p.upper() for p in products],
        rolls=sum(1 for _, ev in all_events if ev.reason == "volume"),
        years_written=years_written,
    )


def _find_contract_file(paths: FuturesPaths, product: str, contract: str) -> str | None:
    hits = list(paths.raw_root.glob(f"exchange=*/product={product}/{contract}.parquet"))
    return str(hits[0]) if hits else None
