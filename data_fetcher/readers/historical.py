"""Reader for the historical tree: ``2005-202506/EXCHANGE/PRODUCT/XXYYMM.csv``.

One file per real contract (12 columns, no ``sequence``), UPPERCASE names.
Contract lifetime ≈ 14 months before its delivery month, which allows
file-level pruning against a requested date range without parsing anything.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl
from loguru import logger

from data_fetcher.exceptions import RawLayoutError, SchemaDriftError
from data_fetcher.readers.base import (
    RawBatch,
    is_pseudo_contract,
    normalize_raw,
    scan_raw_csv,
    split_contract,
)

# A contract can print bars this many months before its delivery month.
CONTRACT_LIFETIME_MONTHS = 14


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def contract_overlaps_range(delivery_ym: int, start: dt.date | None, end: dt.date | None) -> bool:
    """File-level pruning: does [delivery - lifetime, delivery] intersect [start, end]?"""
    delivery_idx = _month_index(delivery_ym // 100, delivery_ym % 100)
    first_idx = delivery_idx - CONTRACT_LIFETIME_MONTHS
    if start is not None and delivery_idx < _month_index(start.year, start.month):
        return False
    if end is not None and first_idx > _month_index(end.year, end.month):
        return False
    return True


class HistoricalTreeReader:
    """Iterate real contracts of the per-contract historical tree."""

    name = "historical"

    def __init__(
        self,
        root: Path,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> None:
        if not root.is_dir():
            raise RawLayoutError(f"historical tree not found: {root}")
        self.root = root
        self.start = start
        self.end = end
        self._product_dirs: dict[str, Path] = {}
        for exchange_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for product_dir in sorted(p for p in exchange_dir.iterdir() if p.is_dir()):
                code = product_dir.name.upper()
                if code in self._product_dirs:
                    raise RawLayoutError(
                        f"product {code} appears under two exchanges: "
                        f"{self._product_dirs[code]} and {product_dir}"
                    )
                self._product_dirs[code] = product_dir

    def discover(self, products: Sequence[str] | None) -> list[str]:
        known = sorted(self._product_dirs)
        if products is None:
            return known
        wanted = {p.upper() for p in products}
        return [p for p in known if p in wanted]

    def iter_product(self, product: str) -> Iterator[RawBatch]:
        product = product.upper()
        product_dir = self._product_dirs.get(product)
        if product_dir is None:
            return
        for csv_path in sorted(product_dir.glob("*.csv")):
            stem = csv_path.stem.upper()
            if is_pseudo_contract(stem):
                continue
            file_product, delivery_ym = split_contract(stem, csv_path)
            if file_product != product:
                raise SchemaDriftError(
                    f"{csv_path}: filename product {file_product!r} != directory {product!r}"
                )
            if not contract_overlaps_range(delivery_ym, self.start, self.end):
                continue
            lf = normalize_raw(scan_raw_csv([csv_path], has_sequence=False))
            lf = lf.with_columns(
                pl.lit(product).alias("product"),
                pl.lit(stem).alias("contract"),
                pl.lit(delivery_ym, dtype=pl.Int32).alias("delivery_ym"),
            ).select(
                "exchange",
                "symbol",
                "product",
                "contract",
                "delivery_ym",
                "open",
                "high",
                "low",
                "close",
                "amount",
                "volume",
                "position",
                "bob",
            )
            yield RawBatch(product=product, contract=stem, sources=(csv_path,), lf=lf)
        if not any(product_dir.glob("*.csv")):
            logger.warning("historical product dir {} holds no CSVs", product_dir)
