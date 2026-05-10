"""data_fetcher — pluggable data ingest for AutoLLM.

Public API (the four things every downstream pipeline cares about):

    Fetcher                     orchestrator
    AKshareParquetAdapter       v1 DataAdapter implementation
    DataAdapter                 the typed protocol for any adapter
    Source                      the typed protocol for any data source

Helpers:

    normalise(symbol)           SH/SZ + 6-digit canonicalisation
    exceptions                  typed errors

Phase-1 README + the proposal §8 are the spec; CLAUDE.md has the
conventions.
"""

from __future__ import annotations

from data_fetcher import exceptions, trade_calendar
from data_fetcher.adapter import AKshareParquetAdapter, DataAdapter
from data_fetcher.akshare_source import AKshareSource
from data_fetcher.fetcher import Fetcher
from data_fetcher.source import Source
from data_fetcher.symbols import normalise, to_akshare_bare, to_akshare_with_market

__all__ = [
    "AKshareParquetAdapter",
    "AKshareSource",
    "DataAdapter",
    "Fetcher",
    "Source",
    "exceptions",
    "trade_calendar",
    "normalise",
    "to_akshare_bare",
    "to_akshare_with_market",
]
