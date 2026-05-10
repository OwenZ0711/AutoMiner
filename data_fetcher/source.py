"""Source — pluggable data-source Protocol.

Sources own *only* "raw upstream → polars.DataFrame in our schema" conversion.
They do not own storage, normalisation policy, retry strategy beyond
their own knowledge of upstream error modes, or the public API.

Adding a new source is ~30 lines: subclass / structurally implement, plug
into Fetcher via the constructor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl


@runtime_checkable
class Source(Protocol):
    """Implementors return Polars DataFrames in the canonical schemas defined
    in storage.OHLCV_SCHEMA and the event-log shape below.

    On error, raise data_fetcher.exceptions.* rather than upstream library
    exceptions — the caller decides retry / skip / abort policy and shouldn't
    have to know which library you wrap.
    """

    name: str

    def pull_ohlcv(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjust: str = "qfq",
    ) -> pl.DataFrame:
        """Daily OHLCV for `symbol` between `start` and `end` inclusive.

        `symbol` is in canonical SH/SZ + 6-digit form (this module never
        sees raw forms; `Fetcher` normalises before calling).

        `adjust` ∈ {"" (raw), "qfq" (forward-adjusted), "hfq" (back-adjusted)}.
        Default qfq is what backtests usually want.

        Returned DataFrame matches storage.OHLCV_SCHEMA exactly.

        Raises:
            EmptyResponseError if the response is empty (no trading days in
                range, or upstream is missing this symbol's data).
            SymbolNotFound if the symbol is delisted or not in the catalog.
            RateLimitError after retries exhausted.
            FetchTimeoutError on a hard timeout.
        """
        ...

    def list_universe_event_log(self, name: str) -> pl.DataFrame:
        """Return the add/drop event log for the named index/universe.

        For "csi300" this wraps `ak.index_stock_hist(symbol='sh000300')`
        and is the source of truth for PIT membership.

        Schema (mandatory):
            stock_code  : pl.Utf8 — bare 6-digit code (no SH/SZ prefix)
            in_date     : pl.Date
            out_date    : pl.Date — null/None if the stock is still a member

        Raises UniverseUnavailableError if `name` is not supported.
        """
        ...

    def trade_calendar(self) -> pd.DatetimeIndex:
        """All known SHSE/SZSE trading dates."""
        ...
