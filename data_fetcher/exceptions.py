"""Typed exceptions for the data_fetcher module.

The downstream contract: every error from a Source surfaces as one of these,
never as a raw `requests.HTTPError`, `pandas.errors.EmptyDataError`, or `KeyError`.
This lets `Fetcher.pull` decide policy (retry / skip / abort) without inspecting
upstream tracebacks.
"""

from __future__ import annotations


class DataFetcherError(Exception):
    """Base class for all data_fetcher exceptions."""


class InvalidSymbolError(DataFetcherError, ValueError):
    """Raised when a symbol cannot be normalised to SH/SZ + 6-digit form."""


class SymbolNotFound(DataFetcherError):
    """Upstream returned no rows for a syntactically valid symbol.

    Most likely the symbol was delisted before the requested range, or the
    symbol is valid in our normaliser but not present in the upstream catalog.
    """


class EmptyResponseError(DataFetcherError):
    """Upstream returned a non-error response with zero rows.

    Distinct from SymbolNotFound: the symbol exists, but the requested
    date range happens to contain no trading days, or the upstream is
    temporarily missing data for this symbol.
    """


class RateLimitError(DataFetcherError):
    """Upstream returned a rate-limit signal (HTTP 429 or its API equivalent).

    Surfaced after retries are exhausted; the Fetcher catches it and logs.
    """


class FetchTimeoutError(DataFetcherError):
    """Upstream request did not return within the configured timeout."""


class StorageError(DataFetcherError):
    """A parquet read/write failed in a way that's not just missing data."""


class UniverseUnavailableError(DataFetcherError):
    """Upstream membership-history endpoint returned no data, or the named
    universe is not supported by this Source."""
