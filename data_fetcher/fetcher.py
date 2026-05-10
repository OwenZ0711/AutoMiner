"""Fetcher — top-level orchestrator.

Composes Source + ParquetStore + UniverseCache + trade-calendar cache into
the public API every downstream pipeline calls. This is the layer that:

  * Knows about disk layout (`~/AutoLLM_data/`).
  * Iterates per-symbol pulls with rate-limit-friendly sleeps.
  * Translates "universe + on_date" → list of symbols (PIT-aware).
  * Hands back lazy Polars frames for cheap downstream slicing.

Intentionally has no business logic of its own — it's a thin façade.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl
from loguru import logger

from data_fetcher import trade_calendar as tc
from data_fetcher.akshare_source import AKshareSource
from data_fetcher.exceptions import (
    DataFetcherError,
    EmptyResponseError,
    UniverseUnavailableError,
)
from data_fetcher.source import Source
from data_fetcher.storage import OHLCV_SCHEMA, ParquetStore
from data_fetcher.symbols import normalise
from data_fetcher.universe import UniverseCache


def _coerce_date(d: dt.date | pd.Timestamp | str) -> dt.date:
    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        return d
    return pd.Timestamp(d).date()


class Fetcher:
    def __init__(
        self,
        source: Source | None = None,
        root: Path | str = "~/AutoLLM_data",
        adjust: str = "qfq",
        sleep_s: float = 0.1,
        universe_refresh_days: int = 7,
        calendar_refresh_days: int = 7,
    ) -> None:
        self.source: Source = source if source is not None else AKshareSource()
        self.root = Path(str(root)).expanduser()
        self.adjust = adjust
        self.sleep_s = sleep_s
        self.store = ParquetStore(root=self.root, source=self.source.name, adjust=adjust)
        self.universe_cache = UniverseCache(
            root=self.root, source=self.source, refresh_days=universe_refresh_days
        )
        self._calendar_path = (
            self.root / "calendar" / f"source={self.source.name}" / "trade_dates.parquet"
        )
        self._calendar_refresh_days = calendar_refresh_days

    # ---------- Backfill ----------

    def pull(
        self,
        universe: str | None = None,
        symbols: Iterable[str] | None = None,
        start: dt.date | pd.Timestamp | str = "2018-01-01",
        end: dt.date | pd.Timestamp | str = "today",
        adjust: str | None = None,
    ) -> dict[str, int]:
        """Backfill OHLCV. Either `universe` (name) OR `symbols` (explicit list)
        must be given. Idempotent: re-running with the same args writes 0 new
        rows.

        Returns: {'pulled', 'skipped', 'failed'} — row counts and failure tally.
        """
        if (universe is None) == (symbols is None):
            raise ValueError("pass exactly one of `universe` or `symbols`")
        if universe is not None:
            try:
                target = self.universe_cache.ever_members(universe, normalise=True)
            except UniverseUnavailableError as e:
                raise UniverseUnavailableError(
                    f"can't pull universe {universe!r}: {e}"
                ) from e
        else:
            target = [normalise(s) for s in symbols]  # type: ignore[arg-type]

        end_ts = pd.Timestamp.today().normalize() if end == "today" else pd.Timestamp(end)
        start_ts = pd.Timestamp(start)
        adj = adjust if adjust is not None else self.adjust

        pulled = skipped = failed = 0
        for sym in target:
            try:
                df = self.source.pull_ohlcv(
                    symbol=sym, start=start_ts, end=end_ts, adjust=adj
                )
            except EmptyResponseError as e:
                logger.warning(f"pull_ohlcv({sym}): empty response — {e}")
                failed += 1
                continue
            except DataFetcherError as e:
                logger.error(f"pull_ohlcv({sym}): {e}")
                failed += 1
                continue
            result = self.store.write_ohlcv(df)
            pulled += result["written"]
            skipped += result["skipped"]
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        logger.info(
            f"pull complete: {pulled} new rows, {skipped} skipped, {failed} failed"
        )
        return {"pulled": pulled, "skipped": skipped, "failed": failed}

    # ---------- Read paths ----------

    def panel(
        self,
        universe: list[str] | str,
        fields: list[str],
        start: dt.date | pd.Timestamp | str,
        end: dt.date | pd.Timestamp | str,
        on_date: dt.date | pd.Timestamp | str | None = None,
        freq: str = "day",
    ) -> pl.LazyFrame:
        """Return a LazyFrame with columns `['date', 'symbol', *fields]` over
        the requested universe and date range.

        `universe` may be:
          * A list of symbols (canonical SH/SZ form or any form symbols.normalise
            accepts).
          * A universe name (e.g. "csi300"); in that case `on_date` selects the
            PIT roster snapshot to scope by.
        """
        if freq != "day":
            raise NotImplementedError(f"freq={freq!r} not supported in v1")

        start_d = _coerce_date(start)
        end_d = _coerce_date(end)

        if isinstance(universe, str):
            if on_date is None:
                on_date = end_d
            on_d = _coerce_date(on_date)
            symbols = self.universe_cache.members_on(universe, on_d, normalise=True)
        else:
            symbols = [normalise(s) for s in universe]

        for f in fields:
            if f not in OHLCV_SCHEMA:
                raise ValueError(
                    f"unknown field {f!r}; must be in {sorted(OHLCV_SCHEMA)}"
                )

        lf = self.store.read_ohlcv(symbols, start_d, end_d)
        keep = ["date", "symbol", *fields]
        return lf.select(keep)

    def universe(self, name: str, on_date: dt.date | pd.Timestamp | str) -> list[str]:
        """PIT membership for `name` as of `on_date` (canonical SH/SZ symbols).

        v1 caveat: see CLAUDE.md "CSI 300 membership". `on_date` more than a
        year in the past triggers a one-time survivorship-bias warning.
        """
        return self.universe_cache.members_on(name, _coerce_date(on_date))

    def calendar(self, force_refresh: bool = False) -> pd.DatetimeIndex:
        """All known trading dates from the active source (cached on disk)."""
        if not force_refresh:
            cached = tc.load_cached(
                self._calendar_path, max_age_days=self._calendar_refresh_days
            )
            if cached is not None:
                return cached
        cal = self.source.trade_calendar()
        tc.save_cache(cal, self._calendar_path)
        return cal
