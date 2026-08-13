"""Index-membership cache and PIT lookup.

Caches a Source's event log under
    root/universe/source={src.name}/{name}/membership_history.parquet
and provides:

  * `event_log(name, force_refresh=False)` — full DataFrame, refetch when stale.
  * `members_on(name, on_date, normalise=True)` — list of stock codes that
    are members on `on_date` per the (in_date, out_date) filter.
  * `ever_members(name, normalise=True)` — every code that ever appeared in
    the log, used by `Fetcher.pull` to determine the backfill set.

In v1, the AKshareSource always reports `out_date = NULL` (it has no
removal data), so `members_on` is a strict in_date filter. The same code
correctly handles a v1.5 Source that exposes real out_dates — we don't
re-write this layer when paid data lands.

Survivorship-bias warning: the first time `members_on` is called for a
date older than 1 year, log a one-time warning per universe. Downstream
backtests on early years are biased toward today's survivors; consumers
should know.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
from loguru import logger

from data_fetcher.source import Source
from data_fetcher.symbols import normalise as _norm


class UniverseCache:
    def __init__(
        self,
        root: Path,
        source: Source,
        refresh_days: int = 7,
    ) -> None:
        self.root = Path(root)
        self.source = source
        self.refresh_days = refresh_days
        self._warned: set[str] = set()  # universe names we've already warned about

    # ---------- Cache path / freshness ----------

    def _cache_path(self, name: str) -> Path:
        return (
            self.root
            / "universe"
            / f"source={self.source.name}"
            / name
            / "membership_history.parquet"
        )

    def _is_stale(self, path: Path) -> bool:
        if not path.exists():
            return True
        age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
        return age.days > self.refresh_days

    # ---------- Event log ----------

    def event_log(self, name: str, force_refresh: bool = False) -> pl.DataFrame:
        path = self._cache_path(name)
        if force_refresh or self._is_stale(path):
            df = self.source.list_universe_event_log(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
            return df
        return pl.read_parquet(path)

    # ---------- Lookups ----------

    def members_on(
        self, name: str, on_date: dt.date, normalise: bool = True
    ) -> list[str]:
        """Members of `name` on `on_date`.

        Filter: `in_date <= on_date AND (out_date IS NULL OR out_date > on_date)`.
        In v1 (AKshareSource), out_date is always NULL; the second clause is a
        no-op. In v1.5+ (Source with real removal events), it does the right thing.

        Logs a one-time survivorship-bias warning per universe if `on_date` is
        more than a year in the past.
        """
        log = self.event_log(name)
        d = on_date if isinstance(on_date, dt.date) else dt.date(*str(on_date).split("-"))  # type: ignore[arg-type]
        members_df = log.filter(
            (pl.col("in_date") <= d)
            & (pl.col("out_date").is_null() | (pl.col("out_date") > d))
        )
        codes = members_df["stock_code"].to_list()

        if normalise:
            codes = [_norm(c) for c in codes]

        # Survivorship warning, once per (universe).
        if name not in self._warned:
            cutoff = dt.date.today() - dt.timedelta(days=365)
            if d < cutoff:
                self._warned.add(name)
                logger.warning(
                    f"Survivorship bias: members_on({name!r}, {d}) returned "
                    f"{len(codes)} members. v1 has no removal data; stocks that "
                    f"were in the index then but have since left are silently "
                    f"absent. See CLAUDE.md 'CSI 300 membership' for the v1.5 "
                    f"mitigation path."
                )
        return codes

    def ever_members(self, name: str, normalise: bool = True) -> list[str]:
        """Every stock code that ever appeared in the event log for `name`.

        Used by `Fetcher.pull` to compute the union of symbols we need to
        backfill OHLCV for.
        """
        log = self.event_log(name)
        codes = log["stock_code"].unique().to_list()
        if normalise:
            codes = [_norm(c) for c in codes]
        return codes
