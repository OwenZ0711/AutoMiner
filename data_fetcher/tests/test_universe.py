"""Tests for universe.py — cached event log + PIT (additions-only) lookup.

The behaviour we want, given the v1 limitation:
  * Cache the event log on disk under root/universe/source={src}/{name}/.
  * Refresh if older than refresh_days.
  * `members_on(name, on_date)` returns 6-digit codes (or normalised SH/SZ
    if requested) filtered by `(in_date <= on_date) AND (out_date IS NULL OR
    out_date > on_date)` — works correctly for both v1 (out_date always
    NULL) and a v1.5 Source that exposes real removal events.
  * On the first lookup with `on_date < today − 1y`, log a one-line
    survivorship-bias warning.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from data_fetcher.exceptions import UniverseUnavailableError
from data_fetcher.universe import UniverseCache


# ---------- Fakes ----------


class _FakeSource:
    """Source stub: only implements list_universe_event_log."""

    name = "fake"

    def __init__(self, event_log: pl.DataFrame, raise_on: str | None = None) -> None:
        self._log = event_log
        self._raise_on = raise_on
        self.calls: list[str] = []

    def list_universe_event_log(self, name: str) -> pl.DataFrame:
        self.calls.append(name)
        if self._raise_on == name:
            raise UniverseUnavailableError(f"fake says no for {name}")
        return self._log


def _v1_event_log() -> pl.DataFrame:
    """A 5-row 'today's roster' event log: out_date always NULL (v1 shape)."""
    rows = [
        ("600519", dt.date(2005, 4, 8), None),     # in since inception
        ("000001", dt.date(2005, 4, 8), None),
        ("300750", dt.date(2018, 12, 14), None),   # joined 2018 rebalance
        ("688981", dt.date(2020, 7, 13), None),    # joined 2020
        ("002594", dt.date(2021, 6, 15), None),    # joined 2021
    ]
    return pl.DataFrame(
        rows,
        schema={
            "stock_code": pl.Utf8,
            "in_date": pl.Date,
            "out_date": pl.Date,
        },
        orient="row",
    )


def _v15_event_log_with_removals() -> pl.DataFrame:
    """Forward-compatible: a 4-row event log including real removal events.
    Mirrors the v1.5 Source that we'll build later. Same downstream logic
    must work.
    """
    rows = [
        ("600519", dt.date(2018, 1, 1), None),                  # still in
        ("000001", dt.date(2018, 1, 1), dt.date(2024, 6, 14)),  # left mid-window
        ("300750", dt.date(2018, 12, 14), None),
        ("000333", dt.date(2019, 6, 17), dt.date(2022, 12, 12)),  # left
    ]
    return pl.DataFrame(
        rows,
        schema={
            "stock_code": pl.Utf8,
            "in_date": pl.Date,
            "out_date": pl.Date,
        },
        orient="row",
    )


# ---------- Cache + load ----------


def test_first_call_fetches_from_source(tmp_data_root: Path) -> None:
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)
    log = uc.event_log("csi300")
    assert log.height == 5
    assert src.calls == ["csi300"]
    # File written under root/universe/source=fake/csi300/membership_history.parquet
    cache_path = (
        tmp_data_root
        / "universe"
        / "source=fake"
        / "csi300"
        / "membership_history.parquet"
    )
    assert cache_path.exists()


def test_second_call_reads_from_cache(tmp_data_root: Path) -> None:
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)
    uc.event_log("csi300")
    uc.event_log("csi300")
    # Source called only once.
    assert len(src.calls) == 1


def test_stale_cache_refetches(tmp_data_root: Path) -> None:
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src, refresh_days=7)
    uc.event_log("csi300")
    # Age the file out.
    cache_path = (
        tmp_data_root
        / "universe"
        / "source=fake"
        / "csi300"
        / "membership_history.parquet"
    )
    import os

    old = (dt.datetime.now() - dt.timedelta(days=14)).timestamp()
    os.utime(cache_path, (old, old))
    uc.event_log("csi300")
    assert len(src.calls) == 2


def test_force_refresh(tmp_data_root: Path) -> None:
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)
    uc.event_log("csi300")
    uc.event_log("csi300", force_refresh=True)
    assert len(src.calls) == 2


# ---------- members_on logic ----------


def test_members_on_v1_filters_by_in_date_only(tmp_data_root: Path) -> None:
    """v1 invariant: no out_date data; in_date filter is the only PIT signal."""
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)

    # As of 2018-06-30: 600519 + 000001 are in; 300750 (Dec 2018), 688981
    # (Jul 2020), 002594 (Jun 2021) had not joined yet.
    members = uc.members_on("csi300", dt.date(2018, 6, 30), normalise=False)
    assert set(members) == {"600519", "000001"}

    # As of 2020-01-01: + 300750.
    members = uc.members_on("csi300", dt.date(2020, 1, 1), normalise=False)
    assert set(members) == {"600519", "000001", "300750"}

    # As of 2024-01-01: all 5.
    members = uc.members_on("csi300", dt.date(2024, 1, 1), normalise=False)
    assert len(members) == 5


def test_members_on_normalise_default(tmp_data_root: Path) -> None:
    """By default, members_on returns canonical SH/SZ-prefixed symbols."""
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)
    members = uc.members_on("csi300", dt.date(2024, 1, 1))
    # All should be SH or SZ prefixed.
    assert all(m.startswith(("SH", "SZ")) for m in members)
    assert "SH600519" in members
    assert "SZ000001" in members
    assert "SZ300750" in members


def test_members_on_v15_handles_removals(tmp_data_root: Path) -> None:
    """Forward-compatibility: filter respects out_date when present."""
    src = _FakeSource(_v15_event_log_with_removals())
    uc = UniverseCache(root=tmp_data_root, source=src)

    # As of 2024-12-31: 000001 left in 2024-06, 000333 left in 2022. So 600519, 300750 only.
    members = uc.members_on("csi300", dt.date(2024, 12, 31), normalise=False)
    assert set(members) == {"600519", "300750"}

    # As of 2020-01-01: 000001 still in (out_date=2024 is in future), 000333 still in
    # (in_date=2019, out_date=2022). 300750 yes (Dec 2018). 600519 yes.
    members = uc.members_on("csi300", dt.date(2020, 1, 1), normalise=False)
    assert set(members) == {"600519", "000001", "300750", "000333"}


def test_members_on_old_date_logs_survivorship_warning(
    tmp_data_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Old dates trigger a once-per-process survivorship-bias warning."""
    import sys
    from loguru import logger

    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)

    # Wire loguru → caplog so we can assert.
    handler_id = logger.add(sys.stderr, level="DEBUG")
    try:
        records: list[str] = []
        logger.add(lambda m: records.append(str(m)), level="WARNING")

        old_date = dt.date.today() - dt.timedelta(days=400)
        uc.members_on("csi300", old_date)
        assert any("survivorship" in r.lower() for r in records)

        # A second call shouldn't re-log.
        before = len(records)
        uc.members_on("csi300", old_date)
        new_records = records[before:]
        assert not any("survivorship" in r.lower() for r in new_records), (
            "warning should fire only once per (universe) per process"
        )
    finally:
        logger.remove(handler_id)


def test_members_on_recent_date_no_warning(
    tmp_data_root: Path,
) -> None:
    """Dates within the last year don't fire the bias warning."""
    import sys
    from loguru import logger

    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)

    handler_id = logger.add(sys.stderr, level="DEBUG")
    try:
        records: list[str] = []
        logger.add(lambda m: records.append(str(m)), level="WARNING")

        recent = dt.date.today() - dt.timedelta(days=30)
        uc.members_on("csi300", recent)
        assert not any("survivorship" in r.lower() for r in records)
    finally:
        logger.remove(handler_id)


def test_unknown_universe_raises(tmp_data_root: Path) -> None:
    src = _FakeSource(_v1_event_log(), raise_on="hsi")
    uc = UniverseCache(root=tmp_data_root, source=src)
    with pytest.raises(UniverseUnavailableError):
        uc.event_log("hsi")


def test_members_on_empty_for_pre_inception_date(tmp_data_root: Path) -> None:
    src = _FakeSource(_v1_event_log())
    uc = UniverseCache(root=tmp_data_root, source=src)
    # CSI 300 launched 2005-04-08. Earlier date → empty.
    members = uc.members_on("csi300", dt.date(2000, 1, 1), normalise=False)
    assert members == []


def test_ever_members_returns_full_roster(tmp_data_root: Path) -> None:
    """`ever_members` returns the union of stocks that ever appeared in
    the event log — used by Fetcher.pull to know which symbols to backfill.
    """
    src = _FakeSource(_v15_event_log_with_removals())
    uc = UniverseCache(root=tmp_data_root, source=src)
    ever = uc.ever_members("csi300", normalise=False)
    assert set(ever) == {"600519", "000001", "300750", "000333"}
