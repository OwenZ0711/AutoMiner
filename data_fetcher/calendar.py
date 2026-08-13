"""L1a: per-product per-date session tables, INFERRED from the data (trap 1).

Session calendars are time-varying (night launches from 2013-07, COVID
suspension 2020-02→05, RB night end 01:00→23:00 in 2016-05, CFFEX hour change
2016), so nothing is hard-coded: templates are discovered per product from the
bars of its top-2 daily-volume contracts.

Algorithm (per product):
1. Project bars to (trading_date, rel_min) where rel_min is the minute-of-day
   shifted to a 20:30-anchored cycle — a night session and its post-midnight
   tail become one contiguous block.
2. Bucket into 5-minute slots → per-day boolean vector (288 slots).
3. Trailing 15-day majority vote per slot (kills single-day liquidity holes;
   pre-holiday night suspensions survive as exceptions, not new templates).
4. Run-length encode identical smoothed vectors; merge runs < 10 trading days
   into the preceding run (real regime changes persist).
5. Recover exact interval endpoints inside each run from raw minutes
   (mode of per-day min / max+1 per block) — yields (21:00, 23:00), not
   bucket-rounded values.
6. Days inside a night-template run whose night slots are empty while day
   slots are populated → ``night_exceptions`` (masks carry per-minute truth
   downstream; the template stays).
7. Templates are canonicalized globally: identical interval tuples share a
   ``template_id`` across products (ids assigned in sorted order → stable).

Outputs under ``~/AutoLLM_data/futures/calendar/``: session_templates,
product_sessions, night_exceptions, trading_days (full recompute, checked
against the ingest bootstrap where their ranges overlap).
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from data_fetcher.exceptions import CalendarInferenceError
from futures_common.paths import FuturesPaths

ANCHOR_MIN = 20 * 60 + 30  # 20:30 — the day-cycle anchor
N_BUCKETS = 288  # 1440 / 5
NIGHT_BUCKETS = slice(0, 78)  # rel [0, 390) = wall 20:30 → 03:00
DAY_BUCKETS = slice(78, N_BUCKETS)

_MINUTES_PER_DAY = 1440


@dataclass(frozen=True, slots=True)
class CalendarConfig:
    bucket_minutes: int = 5
    smoothing_days: int = 15
    min_run_days: int = 10
    anchors_required: int = 3
    overrides: tuple[Mapping[str, Any], ...] = ()


@dataclass(slots=True)
class CalendarReport:
    products: list[str]
    n_templates: int
    n_trading_days: int
    n_night_exceptions: int

    def as_dict(self) -> dict[str, object]:
        return {
            "products": self.products,
            "n_templates": self.n_templates,
            "n_trading_days": self.n_trading_days,
            "n_night_exceptions": self.n_night_exceptions,
        }


# ------------------------------------------------------------------ pure steps


def smooth_majority(m: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window majority vote per column. m: (D, B) bool → (D, B) bool."""
    csum = np.cumsum(m.astype(np.int32), axis=0)
    padded = np.vstack([np.zeros((1, m.shape[1]), dtype=np.int32), csum])
    d = np.arange(1, m.shape[0] + 1)
    lo = np.maximum(d - window, 0)
    counts = padded[d] - padded[lo]
    length = (d - lo)[:, None]
    out: np.ndarray = counts * 2 > length
    return out


def rle_runs(rows: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Run-length encode identical rows; merge runs shorter than min_run into
    the PRECEDING run (a short first run is absorbed by the one after it).
    Returns [start, end] index pairs (inclusive) covering all rows."""
    d = rows.shape[0]
    changes = [0]
    for i in range(1, d):
        if not np.array_equal(rows[i], rows[i - 1]):
            changes.append(i)
    runs = [
        (s, (changes[k + 1] - 1) if k + 1 < len(changes) else d - 1) for k, s in enumerate(changes)
    ]

    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and (e - s + 1) < min_run:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    if len(merged) > 1 and (merged[0][1] - merged[0][0] + 1) < min_run:
        first = merged.pop(0)
        merged[0] = (first[0], merged[0][1])

    # re-merge adjacent runs whose MAJORITY rows are identical
    out: list[tuple[int, int]] = []
    for s, e in merged:
        if out and np.array_equal(run_template_row(rows, *out[-1]), run_template_row(rows, s, e)):
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def run_template_row(rows: np.ndarray, s: int, e: int) -> np.ndarray:
    """The run's template = the most frequent row pattern inside it (merged
    runs contain transitional/absorbed rows; the majority wins)."""
    uniq, counts = np.unique(rows[s : e + 1], axis=0, return_counts=True)
    row: np.ndarray = uniq[int(np.argmax(counts))]
    return row


def _blocks(vec: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True bucket blocks of a template vector → [b0, b1] inclusive."""
    idx = np.flatnonzero(vec)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [idx.size - 1]])
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends, strict=True)]


def _mode(values: pl.Series) -> int:
    counts = values.value_counts().sort(["count", values.name], descending=[True, False])
    return int(counts[values.name][0])


def rel_to_wall(rel: int) -> dt.time:
    wall = (rel + ANCHOR_MIN) % _MINUTES_PER_DAY
    return dt.time(wall // 60, wall % 60)


# ------------------------------------------------------------------- inference


def _product_bars(
    paths: FuturesPaths,
    product: str,
    start: dt.date | None,
    end: dt.date | None,
) -> pl.DataFrame:
    """(trading_date, rel_min) unique pairs of the product's top-2 daily contracts.

    Restricted to [start, end]: bars outside the requested range come from
    partial contract lifetimes (full-lifetime ingest tails) and would produce
    artifact template runs inferred from thin far-month data.
    """
    files = sorted(paths.raw_root.glob(f"exchange=*/product={product}/*.parquet"))
    if not files:
        return pl.DataFrame(schema={"trading_date": pl.Date, "rel_min": pl.Int32})
    lf = pl.scan_parquet([str(f) for f in files])
    if start is not None:
        lf = lf.filter(pl.col("trading_date") >= start)
    if end is not None:
        lf = lf.filter(pl.col("trading_date") <= end)
    lf = lf.select(
        "contract",
        "trading_date",
        "volume",
        (
            (
                pl.col("bob").dt.hour().cast(pl.Int32) * 60
                + pl.col("bob").dt.minute().cast(pl.Int32)
                - ANCHOR_MIN
            ).mod(_MINUTES_PER_DAY)
        ).alias("rel_min"),
    )
    day_vol = lf.group_by("contract", "trading_date").agg(pl.col("volume").sum())
    top2 = (
        day_vol.with_columns(
            pl.col("volume")
            .rank(method="ordinal", descending=True)
            .over("trading_date")
            .alias("rank")
        )
        .filter(pl.col("rank") <= 2)
        .select("contract", "trading_date")
    )
    return (
        lf.join(top2, on=["contract", "trading_date"], how="semi")
        .select("trading_date", "rel_min")
        .unique()
        .collect(engine="streaming")
        .sort("trading_date", "rel_min")
    )


@dataclass(slots=True)
class _ProductSessions:
    product: str
    runs: list[tuple[dt.date, dt.date, tuple[tuple[int, int], ...], int]]
    # (date_start, date_end, interval rel-endpoints tuple, n_days)
    night_exceptions: list[dt.date]
    day_dates: np.ndarray  # trading dates (epoch days) with day-session bars


def _infer_product(product: str, bars: pl.DataFrame, cfg: CalendarConfig) -> _ProductSessions:
    dates = bars["trading_date"].unique().sort()
    date_days = dates.cast(pl.Int32).to_numpy()
    d_index = {int(v): i for i, v in enumerate(date_days)}
    n_days = len(date_days)

    m = np.zeros((n_days, N_BUCKETS), dtype=bool)
    rows = bars["trading_date"].cast(pl.Int32).to_numpy()
    cols = (bars["rel_min"].to_numpy() // cfg.bucket_minutes).astype(np.int64)
    m[[d_index[int(r)] for r in rows], cols] = True

    smoothed = smooth_majority(m, cfg.smoothing_days)
    runs = rle_runs(smoothed, cfg.min_run_days)

    out_runs: list[tuple[dt.date, dt.date, tuple[tuple[int, int], ...], int]] = []
    exceptions: list[dt.date] = []
    epoch = dt.date(1970, 1, 1)

    for s, e in runs:
        vec = run_template_row(smoothed, s, e)
        run_dates = date_days[s : e + 1]
        run_bars = bars.filter(
            pl.col("trading_date").cast(pl.Int32).is_between(int(run_dates[0]), int(run_dates[-1]))
        )
        intervals: list[tuple[int, int]] = []
        for b0, b1 in _blocks(vec):
            lo_rel, hi_rel = b0 * cfg.bucket_minutes, (b1 + 1) * cfg.bucket_minutes
            block = run_bars.filter(pl.col("rel_min").is_between(lo_rel, hi_rel - 1))
            if block.is_empty():
                continue
            per_day = block.group_by("trading_date").agg(
                pl.col("rel_min").min().alias("lo"),
                (pl.col("rel_min").max() + 1).alias("hi"),
            )
            intervals.append((_mode(per_day["lo"]), _mode(per_day["hi"])))
        intervals.sort()
        if not intervals:
            continue  # no-data stretch (thin partial-lifetime bars) — not a template

        # night exceptions: raw (unsmoothed) presence inside a night-template run
        if vec[NIGHT_BUCKETS].any():
            night_any = m[s : e + 1, NIGHT_BUCKETS].any(axis=1)
            day_any = m[s : e + 1, DAY_BUCKETS].any(axis=1)
            for i in np.flatnonzero(day_any & ~night_any):
                exceptions.append(epoch + dt.timedelta(days=int(run_dates[i])))

        out_runs.append(
            (
                epoch + dt.timedelta(days=int(run_dates[0])),
                epoch + dt.timedelta(days=int(run_dates[-1])),
                tuple(intervals),
                int(e - s + 1),
            )
        )

    day_mask = m[:, DAY_BUCKETS].any(axis=1)
    return _ProductSessions(
        product=product,
        runs=out_runs,
        night_exceptions=exceptions,
        day_dates=date_days[day_mask],
    )


# ------------------------------------------------------------------ build stage


def build_calendar(
    paths: FuturesPaths,
    *,
    products: Sequence[str] | None = None,
    cfg: CalendarConfig | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> CalendarReport:
    cfg = cfg or CalendarConfig()
    if products is None:
        products = sorted(
            p.name.removeprefix("product=") for p in paths.raw_root.glob("exchange=*/product=*")
        )
    products = [p.upper() for p in products]

    inferred: list[_ProductSessions] = []
    for product in products:
        bars = _product_bars(paths, product, start, end)
        if bars.is_empty():
            logger.warning("calendar: no ingested bars for {} — skipped", product)
            continue
        inferred.append(_infer_product(product, bars, cfg))
        logger.info(
            "calendar {}: {} runs, {} night exceptions",
            product,
            len(inferred[-1].runs),
            len(inferred[-1].night_exceptions),
        )

    if not inferred:
        raise CalendarInferenceError("no products with ingested bars")

    _apply_overrides(inferred, cfg.overrides)

    # ---- global template canonicalization
    all_tuples = sorted({run[2] for ps in inferred for run in ps.runs})
    template_id = {tup: i for i, tup in enumerate(all_tuples)}

    template_rows = [
        {
            "template_id": template_id[tup],
            "seq": seq,
            "start_time": rel_to_wall(lo),
            "end_time": rel_to_wall(hi % _MINUTES_PER_DAY),
            "bars": hi - lo,
        }
        for tup in all_tuples
        for seq, (lo, hi) in enumerate(tup)
    ]
    sessions_rows = [
        {
            "product": ps.product,
            "date_start": d0,
            "date_end": d1,
            "template_id": template_id[tup],
            "n_days": n,
        }
        for ps in inferred
        for (d0, d1, tup, n) in ps.runs
    ]
    exception_rows = [
        {"product": ps.product, "trading_date": d} for ps in inferred for d in ps.night_exceptions
    ]

    # ---- trading days: full recompute from ingested day-session bars
    counts: dict[int, int] = {}
    for ps in inferred:
        for day in ps.day_dates:
            counts[int(day)] = counts.get(int(day), 0) + 1
    required = min(cfg.anchors_required, len(inferred))
    trading = sorted(day for day, c in counts.items() if c >= required)
    epoch = dt.date(1970, 1, 1)
    trading_dates = [epoch + dt.timedelta(days=d) for d in trading]

    _check_against_bootstrap(paths, trading_dates)

    # ---- writes (atomic)
    paths.calendar_dir.mkdir(parents=True, exist_ok=True)
    _write(
        pl.DataFrame(
            template_rows,
            schema={
                "template_id": pl.Int16,
                "seq": pl.Int8,
                "start_time": pl.Time,
                "end_time": pl.Time,
                "bars": pl.Int16,
            },
        ).sort("template_id", "seq"),
        paths.session_templates,
    )
    _write(
        pl.DataFrame(
            sessions_rows,
            schema={
                "product": pl.Utf8,
                "date_start": pl.Date,
                "date_end": pl.Date,
                "template_id": pl.Int16,
                "n_days": pl.Int32,
            },
        ).sort("product", "date_start"),
        paths.product_sessions,
    )
    _write(
        pl.DataFrame(
            exception_rows,
            schema={"product": pl.Utf8, "trading_date": pl.Date},
        ).sort("product", "trading_date"),
        paths.night_exceptions,
    )
    _write(
        pl.DataFrame({"trading_date": trading_dates}, schema={"trading_date": pl.Date}),
        paths.trading_days,
    )

    return CalendarReport(
        products=[ps.product for ps in inferred],
        n_templates=len(all_tuples),
        n_trading_days=len(trading_dates),
        n_night_exceptions=len(exception_rows),
    )


def _apply_overrides(
    inferred: list[_ProductSessions], overrides: Sequence[Mapping[str, Any]]
) -> None:
    """Manual escape hatch (configs/calendar.yaml): replace the template for a
    (product, date range) after inference so a mis-inference never blocks the
    pipeline. Overlapping inferred runs are trimmed; n_days of trimmed runs
    becomes a calendar-day approximation (informational only)."""
    for ov in overrides:
        product = str(ov["product"]).upper()
        d0 = dt.date.fromisoformat(str(ov["date_start"]))
        d1 = dt.date.fromisoformat(str(ov["date_end"]))
        intervals_raw = ov["intervals"]
        intervals: list[tuple[int, int]] = []
        for lo_s, hi_s in intervals_raw:
            lo_t = dt.time.fromisoformat(str(lo_s))
            hi_t = dt.time.fromisoformat(str(hi_s))
            lo = (lo_t.hour * 60 + lo_t.minute - ANCHOR_MIN) % _MINUTES_PER_DAY
            hi = (hi_t.hour * 60 + hi_t.minute - ANCHOR_MIN) % _MINUTES_PER_DAY
            intervals.append((lo, hi if hi > lo else hi + _MINUTES_PER_DAY))
        tup = tuple(sorted(intervals))

        for ps in inferred:
            if ps.product != product:
                continue
            new_runs: list[tuple[dt.date, dt.date, tuple[tuple[int, int], ...], int]] = []
            for r0, r1, rtup, n in ps.runs:
                if r1 < d0 or r0 > d1:
                    new_runs.append((r0, r1, rtup, n))
                    continue
                if r0 < d0:
                    new_runs.append((r0, d0 - dt.timedelta(days=1), rtup, (d0 - r0).days))
                if r1 > d1:
                    new_runs.append((d1 + dt.timedelta(days=1), r1, rtup, (r1 - d1).days))
            new_runs.append((d0, d1, tup, (d1 - d0).days + 1))
            new_runs.sort(key=lambda r: r[0])
            ps.runs = new_runs
            logger.info("calendar override applied: {} [{} → {}]", product, d0, d1)


def _check_against_bootstrap(paths: FuturesPaths, trading_dates: list[dt.date]) -> None:
    """Every full trading day inside the bootstrap's range must be in the bootstrap."""
    if not paths.trading_days_bootstrap.exists() or not trading_dates:
        return
    boot = set(pl.read_parquet(paths.trading_days_bootstrap)["trading_date"].to_list())
    lo, hi = min(boot), max(boot)
    missing = [d for d in trading_dates if lo <= d <= hi and d not in boot]
    if missing:
        raise CalendarInferenceError(
            f"{len(missing)} inferred trading days absent from bootstrap "
            f"(first: {missing[:5]}) — inconsistent day-session inference"
        )


def _write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ read API


@dataclass(frozen=True, slots=True)
class SessionTable:
    """Read-side view: intervals_for(product, date) → wall-clock interval tuple."""

    templates: Mapping[int, tuple[tuple[dt.time, dt.time], ...]]
    runs: pl.DataFrame  # product, date_start, date_end, template_id

    def intervals_for(self, product: str, date: dt.date) -> tuple[tuple[dt.time, dt.time], ...]:
        hit = self.runs.filter(
            (pl.col("product") == product.upper())
            & (pl.col("date_start") <= date)
            & (pl.col("date_end") >= date)
        )
        if hit.is_empty():
            return ()
        return self.templates[int(hit["template_id"][0])]

    def template_for(self, product: str, date: dt.date) -> int | None:
        hit = self.runs.filter(
            (pl.col("product") == product.upper())
            & (pl.col("date_start") <= date)
            & (pl.col("date_end") >= date)
        )
        return None if hit.is_empty() else int(hit["template_id"][0])


def load_session_table(paths: FuturesPaths) -> SessionTable:
    tdf = pl.read_parquet(paths.session_templates)
    templates: dict[int, tuple[tuple[dt.time, dt.time], ...]] = {}
    for tid, group in tdf.group_by("template_id", maintain_order=True):
        pairs = tuple(
            (row["start_time"], row["end_time"]) for row in group.sort("seq").iter_rows(named=True)
        )
        templates[int(tid[0])] = pairs
    return SessionTable(templates=templates, runs=pl.read_parquet(paths.product_sessions))


def load_trading_days(paths: FuturesPaths) -> np.ndarray:
    """Sorted epoch-day int32 array of full trading days."""
    return (
        pl.read_parquet(paths.trading_days)["trading_date"]
        .cast(pl.Int32)
        .to_numpy()
        .astype(np.int32)
    )
