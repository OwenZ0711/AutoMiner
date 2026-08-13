"""Reader for the 2026 tree: ``2026/YYYYMM/YYYYMMDD/<contract>.csv``.

One file per (calendar day, contract), 13 columns (trailing ``sequence``).
Filenames are mixed-case: SHFE/DCE/INE/GFEX lowercase (``rb2605.csv``),
CFFEX/CZCE uppercase (``IF2603.csv``), pseudo-contracts uppercase
(``RB8888.csv``) — matching is case-insensitive on the FULL stem regex so
prefix collisions (``A`` vs ``AG``/``AL``/``AP``, ``P`` vs ``PB``/``PP``)
cannot happen. Saturday folders hold only post-midnight Friday-night tails;
no folder-level date logic is needed here because every row carries a full
timestamp and the trading-date rule reassigns them downstream.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl

from data_fetcher.exceptions import RawLayoutError
from data_fetcher.readers.base import (
    RawBatch,
    is_pseudo_contract,
    normalize_raw,
    scan_raw_csv,
    split_contract,
)
from data_fetcher.readers.historical import contract_overlaps_range

_CONTRACT_FILE_RE = re.compile(r"^([A-Za-z]{1,2})(\d{4})\.csv$")
_DAY_DIR_RE = re.compile(r"^\d{8}$")


class DailyTreeReader:
    """Iterate real contracts of the per-day 2026 tree, grouped per contract."""

    name = "2026"

    def __init__(
        self,
        root: Path,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> None:
        if not root.is_dir():
            raise RawLayoutError(f"2026 tree not found: {root}")
        self.root = root
        self.start = start
        self.end = end
        # contract (upper) → sorted list of day files; built once, cheap
        # (a directory walk over ~60 trading days * ~1k files).
        self._files_by_contract: dict[str, list[Path]] = {}
        self._products: set[str] = set()
        for month_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for day_dir in sorted(p for p in month_dir.iterdir() if p.is_dir()):
                if not _DAY_DIR_RE.match(day_dir.name):
                    raise RawLayoutError(f"unexpected day folder name: {day_dir}")
                for csv_path in day_dir.iterdir():
                    m = _CONTRACT_FILE_RE.match(csv_path.name)
                    if m is None:
                        continue  # 8888/9999/9998 and anything else non-contract
                    stem = csv_path.stem.upper()
                    if is_pseudo_contract(stem):
                        continue
                    self._files_by_contract.setdefault(stem, []).append(csv_path)
                    self._products.add(split_contract(stem, csv_path)[0])
        for files in self._files_by_contract.values():
            files.sort()

    def discover(self, products: Sequence[str] | None) -> list[str]:
        known = sorted(self._products)
        if products is None:
            return known
        wanted = {p.upper() for p in products}
        return [p for p in known if p in wanted]

    def iter_product(self, product: str) -> Iterator[RawBatch]:
        product = product.upper()
        for contract in sorted(self._files_by_contract):
            file_product, delivery_ym = split_contract(
                contract, self._files_by_contract[contract][0]
            )
            if file_product != product:
                continue
            if not contract_overlaps_range(delivery_ym, self.start, self.end):
                continue
            paths = self._files_by_contract[contract]
            lf = normalize_raw(scan_raw_csv(paths, has_sequence=True))
            lf = lf.with_columns(
                pl.lit(product).alias("product"),
                pl.lit(contract).alias("contract"),
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
            yield RawBatch(product=product, contract=contract, sources=tuple(paths), lf=lf)
