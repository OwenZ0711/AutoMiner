"""Typed exception hierarchy for the futures data band."""

from __future__ import annotations


class DataFetcherError(Exception):
    """Base class for all data-band errors."""


class RawLayoutError(DataFetcherError):
    """A raw tree path does not match the expected layout."""


class SchemaDriftError(DataFetcherError):
    """A raw CSV's columns/values do not match the profiled schema."""


class CalendarInferenceError(DataFetcherError):
    """Session-calendar inference produced an inconsistent result."""


class StorageError(DataFetcherError):
    """A parquet/memmap artifact is malformed or inconsistent."""
