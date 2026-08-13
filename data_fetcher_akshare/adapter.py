"""DataAdapter — the typed protocol every downstream pipeline depends on.

Per proposal §8.3:

    class DataAdapter(Protocol):
        def panel(...)
        def calendar()
        def universe(name, on_date)
        def windows(asset, start, end, length)
        def context_bundle(on_date)

This module:
  1. Defines the `DataAdapter` Protocol (so downstream code can type against it).
  2. Provides `AKshareParquetAdapter`, the v1 implementation that wraps `Fetcher`
     and reads only from cached parquet (no network calls at adapter level).

When paid data arrives, swap in a different adapter — every Pipeline
consumes `DataAdapter`, never `Fetcher` directly.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import polars as pl

from data_fetcher.fetcher import Fetcher
from data_fetcher.symbols import normalise


_DEFAULT_WINDOW_FIELDS = ("open", "high", "low", "close", "volume", "vwap")


@runtime_checkable
class DataAdapter(Protocol):
    """Public contract every pipeline programs against."""

    def panel(
        self,
        universe: list[str] | str,
        fields: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str = "day",
    ) -> pl.LazyFrame:
        ...

    def calendar(self) -> pd.DatetimeIndex:
        ...

    def universe(self, name: str, on_date: pd.Timestamp) -> list[str]:
        ...

    def windows(
        self,
        asset: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        length: int,
    ) -> Iterator[np.ndarray]:
        ...

    def context_bundle(self, on_date: pd.Timestamp) -> dict:
        ...


class AKshareParquetAdapter:
    """v1 adapter — reads from `Fetcher`'s parquet store. No network calls.

    Pipelines should never see this class directly; type their function
    signatures against `DataAdapter` so v1.5+ adapters drop in unchanged.
    """

    def __init__(
        self, fetcher: Fetcher, window_fields: tuple[str, ...] = _DEFAULT_WINDOW_FIELDS
    ) -> None:
        self.fetcher = fetcher
        self.window_fields = window_fields

    # ---------- Forwarding ----------

    def panel(
        self,
        universe: list[str] | str,
        fields: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str = "day",
    ) -> pl.LazyFrame:
        return self.fetcher.panel(
            universe=universe,
            fields=fields,
            start=start,
            end=end,
            freq=freq,
        )

    def calendar(self) -> pd.DatetimeIndex:
        return self.fetcher.calendar()

    def universe(self, name: str, on_date: pd.Timestamp) -> list[str]:
        return self.fetcher.universe(name, on_date)

    # ---------- Windows ----------

    def windows(
        self,
        asset: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        length: int,
    ) -> Iterator[np.ndarray]:
        """Yield successive (length, len(window_fields)) numpy arrays for `asset`,
        chronological, sliding by 1 trading day. Yields nothing if there are
        fewer than `length` days of data.
        """
        sym = normalise(asset)
        df = (
            self.fetcher.panel(
                universe=[sym],
                fields=list(self.window_fields),
                start=start,
                end=end,
            )
            .collect()
            .sort("date")
        )
        if df.height < length:
            return
        # Drop the symbol/date columns; keep the feature matrix.
        feat = df.select(list(self.window_fields)).to_numpy()
        for i in range(0, feat.shape[0] - length + 1):
            yield feat[i : i + length].copy()

    # ---------- Stubs for v2 ----------

    def context_bundle(self, on_date: pd.Timestamp) -> dict:
        """Stub for Pipeline D's LLM Idea Agent (proposal §8.3). v1 returns
        only the as-of date; v2 will assemble news/event/regime context.
        """
        d = on_date if isinstance(on_date, dt.date) else pd.Timestamp(on_date).date()
        return {"as_of": d, "version": "v1-stub"}
