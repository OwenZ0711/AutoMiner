"""Panel align tests: grid, scatter, masks, segmentation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from data_fetcher.calendar import CalendarConfig, build_calendar
from data_fetcher.continuous import build_continuous
from data_fetcher.panel import memmap_io
from data_fetcher.panel.align import _limit_mask, build_panel, day_grid, intervals_to_rel
from data_fetcher.tests.test_calendar import DAY_ONLY, NIGHT_23, _weekdays, _write_contract
from futures_common.paths import FuturesPaths


def test_intervals_to_rel_night_wraps() -> None:
    rel = intervals_to_rel([(dt.time(21, 0), dt.time(1, 0))])
    assert rel == [(30, 270)]  # 21:00 → rel 30; 01:00 → rel 270 (wrapped)


def test_day_grid_union_and_breaks() -> None:
    a = intervals_to_rel(DAY_ONLY)  # 09:00-10:15, 10:30-11:30, 13:30-15:00
    b = intervals_to_rel([(dt.time(9, 30), dt.time(11, 30)), (dt.time(13, 0), dt.time(15, 0))])
    rels, is_break = day_grid([a, b])
    # union covers 09:00 → 11:30 with NO gap at 10:15-10:30 (b bridges it)...
    r_1015 = (10 * 60 + 15 - (20 * 60 + 30)) % 1440
    assert r_1015 in rels
    # ...but a segment break still fires at 10:30 (a's interval start)
    r_1030 = (10 * 60 + 30 - (20 * 60 + 30)) % 1440
    assert bool(is_break[np.searchsorted(rels, r_1030)])
    # gap between 11:30 and 13:00 → break at 13:00
    r_1300 = (13 * 60 - (20 * 60 + 30)) % 1440
    assert bool(is_break[np.searchsorted(rels, r_1300)])


def test_limit_mask_fires_at_30_not_29() -> None:
    def seq(n_locked: int) -> np.ndarray:
        rising = np.arange(10, dtype=float)
        locked = np.full(n_locked, 9.0)
        return np.concatenate([rising, locked])

    day_start = np.zeros(10 + 30, dtype=bool)
    day_start[0] = True
    locked30 = _limit_mask(seq(30), day_start, 30)
    assert locked30.sum() == 30
    day_start = np.zeros(10 + 29, dtype=bool)
    day_start[0] = True
    locked29 = _limit_mask(seq(29), day_start, 30)
    assert locked29.sum() == 0


def _build_slice(tmp_paths: FuturesPaths) -> list[dt.date]:
    days = _weekdays(dt.date(2020, 6, 1), 40)
    _write_contract(tmp_paths, "RB", "RB2101", days, {d: NIGHT_23 for d in days})
    _write_contract(tmp_paths, "I", "I2101", days, {d: DAY_ONLY for d in days})
    build_calendar(tmp_paths, products=["RB", "I"], cfg=CalendarConfig(anchors_required=1))
    build_continuous(tmp_paths, products=["RB", "I"])
    build_panel(tmp_paths, products=["RB", "I"], years=[2020])
    return days


def test_panel_grid_and_masks_match_templates(tmp_paths: FuturesPaths) -> None:
    days = _build_slice(tmp_paths)
    meta = memmap_io.read_meta(tmp_paths, 2020)
    assert meta["products"] == ["I", "RB"]
    grid = pl.read_parquet(tmp_paths.bob_index(2020))
    # union grid per day: RB's night (120) + shared day (225) = 345 rows
    per_day = grid.group_by("trading_date").len()
    assert set(per_day["len"].to_list()) == {345}
    assert grid.height == len(days) * 345

    m_traded = memmap_io.open_array(tmp_paths, 2020, "mask", "m_traded")
    i_col = meta["products"].index("I")
    rb_col = meta["products"].index("RB")
    rel = grid["rel_min"].to_numpy()
    night_rows = rel < 390
    # I never trades at night; RB trades everywhere
    assert m_traded[night_rows, i_col].sum() == 0
    assert m_traded[night_rows, rb_col].all()
    assert m_traded[~night_rows, i_col].all()

    close = memmap_io.open_array(tmp_paths, 2020, "channel", "close_adj")
    assert np.isfinite(close[~night_rows, i_col]).all()
    assert np.isnan(close[night_rows, i_col]).all()


def test_panel_segmentation_breaks(tmp_paths: FuturesPaths) -> None:
    _build_slice(tmp_paths)
    grid = pl.read_parquet(tmp_paths.bob_index(2020))
    one_day = grid.filter(pl.col("trading_date") == grid["trading_date"][0])
    segs = one_day["seg_session"].unique().len()
    # night block + 09:00 + 10:30 + 13:30 = 4 segments per day
    assert segs == 4
    # seg ids strictly increase across days (never reused)
    assert grid["seg_session"].is_sorted()
