"""Shared raw-CSV normalization — every L0 trap filter lives here, nowhere else.

Raw layouts differ (per-contract historical files vs per-day 2026 files, 12 vs
13 columns, mixed filename case), so each layout gets a reader Strategy; both
feed this single :func:`normalize_raw` path so the trap handling exists once.

Traps handled here (numbers from ``gp_cta/proposal.md`` § traps):
- (4) zero-volume padded bars   → ``volume > 0`` filter, unconditional
- (5) literal-"nan" rows        → ``null_values`` at scan; drop null bob/close
- pseudo-contracts (8888/9999/9998) → never yielded by the readers
- mixed-case 2026 filenames     → symbols upper-cased here
- ``type``/``sequence``/``eob`` → dropped at scan (never read into memory)

The trading-date rule (trap 3) needs the trading-day set and is applied later
by ``data_fetcher.ingest``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import polars as pl

from data_fetcher.exceptions import SchemaDriftError

# Real contracts only: 1-2 letter product + 4-digit YYMM (verified across all
# six exchanges, incl. CZCE historical files like TA0702.csv).
CONTRACT_RE = re.compile(r"^([A-Z]{1,2})(\d{4})$")
PSEUDO_RE = re.compile(r"(8888|9999|9998)$")

RAW_COLUMNS_12 = (
    "exchange",
    "symbol",
    "open",
    "close",
    "high",
    "low",
    "amount",
    "volume",
    "position",
    "bob",
    "eob",
    "type",
)
RAW_COLUMNS_13 = (*RAW_COLUMNS_12, "sequence")

_SCAN_OVERRIDES: dict[str, pl.DataType | type[pl.DataType]] = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "open": pl.Float32,
    "close": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "amount": pl.Float64,
    "volume": pl.Float64,  # raw CSVs carry "31867.0"; cast to Int64 post-filter
    "position": pl.Float64,
    "bob": pl.Utf8,
}

NULL_VALUES = ["nan", "NaN", ""]

# Normalized output schema of a RawBatch lazyframe (before trading_date).
# `symbol` (the raw in-file symbol, upper-cased) rides along so ingest can
# assert symbol == contract-from-filename before dropping it.
NORMALIZED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "product": pl.Utf8,
    "contract": pl.Utf8,
    "delivery_ym": pl.Int32,
    "open": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "close": pl.Float32,
    "amount": pl.Float64,
    "volume": pl.Int64,
    "position": pl.Int64,
    "bob": pl.Datetime("us", "Asia/Shanghai"),
}


class RawBatch(NamedTuple):
    """One contract's rows from one raw tree, normalized but not yet dated."""

    product: str  # "RB"
    contract: str  # "RB2001" (canonical upper)
    sources: tuple[Path, ...]  # every raw file contributing rows
    lf: pl.LazyFrame  # NORMALIZED_SCHEMA


@runtime_checkable
class RawReader(Protocol):
    """Strategy interface: one implementation per raw-tree layout."""

    name: str

    def discover(self, products: Sequence[str] | None) -> list[str]:
        """Product codes present in this tree (upper-case), optionally filtered."""
        ...

    def iter_product(self, product: str) -> Iterator[RawBatch]:
        """Yield one RawBatch per real contract of `product` (pseudo-contracts skipped)."""
        ...


def is_pseudo_contract(stem: str) -> bool:
    return bool(PSEUDO_RE.search(stem.upper()))


def split_contract(symbol: str, source: Path) -> tuple[str, int]:
    """'RB2001' → ('RB', 202001). Raises SchemaDriftError on anything unexpected."""
    m = CONTRACT_RE.match(symbol)
    if m is None:
        raise SchemaDriftError(f"{source}: symbol {symbol!r} does not match XXYYMM")
    product = m.group(1)
    yymm = m.group(2)
    yy, mm = int(yymm[:2]), int(yymm[2:])
    if not 1 <= mm <= 12:
        raise SchemaDriftError(f"{source}: symbol {symbol!r} has month {mm}")
    return product, (2000 + yy) * 100 + mm


def scan_raw_csv(paths: Sequence[Path], *, has_sequence: bool) -> pl.LazyFrame:
    """Lazy-scan raw CSV file(s) with the profiled 12/13-column schema."""
    expected = RAW_COLUMNS_13 if has_sequence else RAW_COLUMNS_12
    lf = pl.scan_csv(
        [str(p) for p in paths],
        schema_overrides=_SCAN_OVERRIDES,
        null_values=NULL_VALUES,
        has_header=True,
    )
    names = tuple(lf.collect_schema().names())
    if names != expected:
        raise SchemaDriftError(
            f"{paths[0]}: columns {names} != expected {'13' if has_sequence else '12'}-col layout"
        )
    return lf


def normalize_raw(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Shared normalization path (see module docstring). Output: NORMALIZED_SCHEMA

    minus product/contract/delivery_ym, which need per-batch constants — the
    readers attach those after asserting the file holds exactly one symbol.
    """
    return (
        lf.select(
            "exchange",
            "symbol",
            "open",
            "close",
            "high",
            "low",
            "amount",
            "volume",
            "position",
            "bob",
        )
        # trap 5: literal-"nan" rows parsed to nulls
        .filter(pl.col("bob").is_not_null() & pl.col("close").is_not_null())
        # trap 4: zero-volume padded bars carry no trade information
        .filter(pl.col("volume") > 0)
        .with_columns(
            pl.col("exchange").str.to_uppercase(),
            pl.col("symbol").str.to_uppercase(),
            pl.col("bob")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_unit="us")
            .dt.convert_time_zone("Asia/Shanghai"),
            pl.col("volume").cast(pl.Int64),
            pl.col("position").cast(pl.Int64),
        )
    )
