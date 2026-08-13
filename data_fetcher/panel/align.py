"""L2a: union minute grid + channel/mask memmaps.

Grid semantics (calendar-derived, deterministic): for year Y, one row per
(trading day, rel-minute) where at least one selected product's session
template covers that minute. ``rel`` is the 20:30-anchored minute-of-day, so
within a trading day rows run night → next-morning day session in true
chronological order. Night-exception days keep their full template rows —
``m_traded`` carries the per-minute truth.

Segmentation (bob_index):
- ``seg_session`` (i32): +1 at every session boundary of ANY selected product
  (incl. the 10:15–10:30 mini-break), at any grid gap, and at day starts —
  the lead-lag scan's blocking source. Conservative by construction: a break
  for one product breaks the segment for all.
- ``seg_macro`` (i8): 0 before the 2025-07/2026-01 data hole, 1 after — GP
  window-reset and forced-flat source.

Limit-lock mask (trap 7, CAUSAL): a bar is limit-locked when it sits in a run
of ≥ ``min_run_bars`` consecutive traded bars whose close equals the RUNNING
(causal, within-day) session max (or min) with zero 1-bar return.

Memory: per-product scatter into memmaps — never a full pivot. One product-
year slice is ~15 MB; channels live as OS-paged memmaps.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from data_fetcher.calendar import ANCHOR_MIN, SessionTable, load_session_table
from data_fetcher.continuous import HOLE_LAST_DAY
from data_fetcher.exceptions import StorageError
from data_fetcher.panel import memmap_io
from futures_common.manifest import config_hash
from futures_common.paths import FuturesPaths

_MIN_PER_DAY = 1440
_EPOCH = dt.date(1970, 1, 1)

CHANNELS = (
    "open_adj",
    "close_adj",
    "high_adj",
    "low_adj",
    "open_raw",
    "close_raw",
    "volume",
    "position",
)
MASKS = ("m_traded", "m_limit", "roll_flag")


@dataclass(slots=True)
class PanelReport:
    years: list[int]
    n_products: int
    rows_total: int
    bars_scattered: int
    bars_unmatched: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "years": self.years,
            "n_products": self.n_products,
            "rows_total": self.rows_total,
            "bars_scattered": self.bars_scattered,
            "bars_unmatched": self.bars_unmatched,
        }


def wall_to_rel(minute_of_day: int) -> int:
    return (minute_of_day - ANCHOR_MIN) % _MIN_PER_DAY


def intervals_to_rel(
    intervals: Sequence[tuple[dt.time, dt.time]],
) -> list[tuple[int, int]]:
    """Wall-clock intervals → [rel_start, rel_end) pairs (end may wrap +1440)."""
    out: list[tuple[int, int]] = []
    for lo, hi in intervals:
        r0 = wall_to_rel(lo.hour * 60 + lo.minute)
        r1 = wall_to_rel(hi.hour * 60 + hi.minute)
        if r1 <= r0:
            r1 += _MIN_PER_DAY
        out.append((r0, r1))
    return sorted(out)


def day_grid(
    per_product_intervals: Sequence[Sequence[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Union rel-minutes + break flags for one trading day.

    Returns (rels sorted int32, is_break bool — True where a new session
    segment starts: any product's interval start, or a gap in the union grid).
    """
    covered = np.zeros(2 * _MIN_PER_DAY, dtype=bool)
    starts: set[int] = set()
    for intervals in per_product_intervals:
        for r0, r1 in intervals:
            covered[r0:r1] = True
            starts.add(r0)
    rels = np.flatnonzero(covered).astype(np.int32)
    if rels.size == 0:
        return rels, np.zeros(0, dtype=bool)
    is_break = np.zeros(rels.size, dtype=bool)
    is_break[0] = True
    gap = np.diff(rels) > 1
    is_break[1:][gap] = True
    start_arr = np.array(sorted(starts), dtype=np.int32)
    is_break |= np.isin(rels, start_arr)
    return rels, is_break


def _grid_for_year(
    year: int,
    products: Sequence[str],
    table: SessionTable,
    trading_days: np.ndarray,
) -> pl.DataFrame:
    """bob_index frame: trading_date, rel_min, bob, seg_session, seg_macro."""
    day_lo = (dt.date(year, 1, 1) - _EPOCH).days
    day_hi = (dt.date(year, 12, 31) - _EPOCH).days
    days = trading_days[(trading_days >= day_lo) & (trading_days <= day_hi)]

    frames: list[dict[str, Any]] = []
    seg = -1
    for day_epoch in days:
        day = _EPOCH + dt.timedelta(days=int(day_epoch))
        per_product = []
        for p in products:
            ivs = table.intervals_for(p, day)
            if ivs:
                per_product.append(intervals_to_rel(ivs))
        if not per_product:
            continue
        rels, is_break = day_grid(per_product)
        # previous trading day (for night-bar calendar dates)
        idx = int(np.searchsorted(trading_days, day_epoch))
        prev_epoch = int(trading_days[idx - 1]) if idx > 0 else int(day_epoch) - 1
        prev_day = _EPOCH + dt.timedelta(days=prev_epoch)
        for rel, brk in zip(rels.tolist(), is_break.tolist(), strict=True):
            if brk:
                seg += 1
            wall = (rel + ANCHOR_MIN) % _MIN_PER_DAY
            r = rel % _MIN_PER_DAY
            if r >= 390:  # day side: 03:00 → 20:30 wall
                cal = day
            elif r < 210:  # 20:30 → 24:00 wall, previous trading day's evening
                cal = prev_day
            else:  # 00:00 → 03:00 wall, morning after the previous trading day
                cal = prev_day + dt.timedelta(days=1)
            frames.append(
                {
                    "trading_date": day,
                    "rel_min": rel,
                    "bob": dt.datetime.combine(cal, dt.time(wall // 60, wall % 60)),
                    "seg_session": seg,
                    "seg_macro": 0 if day <= HOLE_LAST_DAY else 1,
                }
            )
    if not frames:
        return pl.DataFrame(
            schema={
                "trading_date": pl.Date,
                "rel_min": pl.Int32,
                "bob": pl.Datetime("us", "Asia/Shanghai"),
                "seg_session": pl.Int32,
                "seg_macro": pl.Int8,
            }
        )
    return pl.DataFrame(frames).with_columns(
        pl.col("rel_min").cast(pl.Int32),
        pl.col("bob").dt.replace_time_zone("Asia/Shanghai"),
        pl.col("seg_session").cast(pl.Int32),
        pl.col("seg_macro").cast(pl.Int8),
    )


def _limit_mask(close: np.ndarray, day_start_flags: np.ndarray, min_run_bars: int) -> np.ndarray:
    """Causal limit-lock heuristic on one product's TRADED bar sequence."""
    n = close.size
    locked = np.zeros(n, dtype=bool)
    if n == 0:
        return locked
    day_starts = np.flatnonzero(day_start_flags)
    for k, s in enumerate(day_starts):
        e = day_starts[k + 1] if k + 1 < day_starts.size else n
        c = close[s:e]
        run_hi = np.maximum.accumulate(c)
        run_lo = np.minimum.accumulate(c)
        ret_zero = np.empty(e - s, dtype=bool)
        ret_zero[0] = False
        ret_zero[1:] = c[1:] == c[:-1]
        candidate = ((c == run_hi) | (c == run_lo)) & ret_zero
        # runs of consecutive candidates ≥ min_run_bars
        run_id = np.cumsum(~candidate)
        if candidate.any():
            counts = np.bincount(run_id[candidate])
            good = np.flatnonzero(counts >= min_run_bars)
            locked[s:e] = candidate & np.isin(run_id, good)
    return locked


def build_panel(
    paths: FuturesPaths,
    *,
    products: Sequence[str],
    years: Sequence[int],
    limit_cfg: Mapping[str, Any] | None = None,
) -> PanelReport:
    products = sorted(p.upper() for p in products)
    table = load_session_table(paths)
    trading_days_arr = pl.read_parquet(paths.trading_days)["trading_date"].cast(pl.Int32).to_numpy()
    min_run_bars = int(((limit_cfg or {}).get("heuristic") or {}).get("min_run_bars", 30))

    report = PanelReport(
        years=list(years),
        n_products=len(products),
        rows_total=0,
        bars_scattered=0,
        bars_unmatched=0,
    )

    for year in years:
        grid = _grid_for_year(year, products, table, trading_days_arr)
        if grid.is_empty():
            logger.warning("panel {}: empty grid — skipped", year)
            continue
        t_rows = grid.height
        n = len(products)
        report.rows_total += t_rows

        # sorted composite key for O(log T) bar → row lookup
        grid_key = (
            grid["trading_date"].cast(pl.Int32).to_numpy().astype(np.int64) * _MIN_PER_DAY * 2
            + grid["rel_min"].to_numpy()
        )
        assert np.all(np.diff(grid_key) > 0), "grid keys must be strictly increasing"

        arrays = {
            name: memmap_io.create_array(paths, year, "channel", name, (t_rows, n))
            for name in CHANNELS
        }
        masks = {
            name: memmap_io.create_array(paths, year, "mask", name, (t_rows, n)) for name in MASKS
        }

        for i, product in enumerate(products):
            path = paths.continuous_year(product, year)
            if not path.exists():
                continue
            bars = pl.read_parquet(path).sort("bob")
            rel = (
                (
                    bars["bob"].dt.hour().cast(pl.Int32) * 60
                    + bars["bob"].dt.minute().cast(pl.Int32)
                    - ANCHOR_MIN
                )
                % _MIN_PER_DAY
            ).to_numpy()
            # night rels sort below 210 rel but grid stores them un-wrapped;
            # match grid encoding: rels are within [0, 1440) already ✓
            key = (
                bars["trading_date"].cast(pl.Int32).to_numpy().astype(np.int64) * _MIN_PER_DAY * 2
                + rel
            )
            pos = np.searchsorted(grid_key, key)
            ok = (pos < grid_key.size) & (grid_key[np.minimum(pos, grid_key.size - 1)] == key)
            report.bars_unmatched += int((~ok).sum())
            rows = pos[ok]

            adj = bars["adj_factor"].to_numpy()[ok]
            for src, dst in (
                ("open", "open_adj"),
                ("close", "close_adj"),
                ("high", "high_adj"),
                ("low", "low_adj"),
            ):
                arrays[dst][rows, i] = (bars[src].cast(pl.Float64).to_numpy()[ok] * adj).astype(
                    np.float32
                )
            arrays["open_raw"][rows, i] = bars["open"].to_numpy()[ok]
            arrays["close_raw"][rows, i] = bars["close"].to_numpy()[ok]
            arrays["volume"][rows, i] = bars["volume"].to_numpy()[ok].astype(np.float32)
            arrays["position"][rows, i] = bars["position"].to_numpy()[ok].astype(np.float32)
            masks["m_traded"][rows, i] = 1
            masks["roll_flag"][rows, i] = bars["roll_flag"].to_numpy()[ok].astype(np.uint8)

            td = bars["trading_date"].cast(pl.Int32).to_numpy()[ok]
            day_start = np.empty(td.size, dtype=bool)
            if td.size:
                day_start[0] = True
                day_start[1:] = td[1:] != td[:-1]
            locked = _limit_mask(
                bars["close"].cast(pl.Float64).to_numpy()[ok], day_start, min_run_bars
            )
            masks["m_limit"][rows, i] = locked.astype(np.uint8)
            report.bars_scattered += int(ok.sum())

        for arr in (*arrays.values(), *masks.values()):
            arr.flush()
        del arrays, masks

        grid.with_row_index("row").write_parquet(paths.bob_index(year))
        memmap_io.write_meta(
            paths,
            {
                "version": memmap_io.PANEL_VERSION,
                "year": year,
                "T": t_rows,
                "N": n,
                "products": list(products),
                "channels": list(CHANNELS),
                "masks": list(MASKS),
                "features": [],
                "config_hash": config_hash(
                    {"products": list(products), "year": year, "min_run_bars": min_run_bars}
                ),
            },
        )
        logger.info("panel {}: T={} N={} rows", year, t_rows, n)

    if report.rows_total == 0:
        raise StorageError("panel build produced no rows — is the calendar built?")
    return report
