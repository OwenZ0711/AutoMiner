"""Partitioned parquet OHLCV store.

Layout:
    root / "raw" / "source={src}" / "adjust={adj}" / "year={Y}" / "symbol={SYM}.parquet"

One small parquet per (source, adjust, year, symbol). Cheap to scan when the
caller asks "give me SH600519 + SZ000001 for 2018-01..2018-12" — we touch
exactly the four files involved.

Idempotency: write_ohlcv merges with whatever's already on disk for each
(year, symbol) pair, deduping by (symbol, date). Re-running a pull that
overlaps existing data writes zero new rows. Counts of written vs skipped
are reported back to the caller.

Schema (every column required):
    date           : pl.Date
    symbol         : pl.Utf8 (canonical SH/SZ form)
    open, high, low, close, volume, turnover, vwap, return_pct, turnover_rate : pl.Float64
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import polars as pl

from data_fetcher.exceptions import StorageError


OHLCV_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "turnover": pl.Float64,
    "vwap": pl.Float64,
    "return_pct": pl.Float64,
    "turnover_rate": pl.Float64,
}


class ParquetStore:
    """Per-source, per-adjust parquet store. Construct one per (source, adjust)
    combination; these have *different* numerical values and must not be mixed.
    """

    def __init__(self, root: Path, source: str, adjust: str = "qfq") -> None:
        self.root = Path(root)
        self.source = source
        self.adjust = adjust
        self._raw_root = self.root / "raw" / f"source={source}" / f"adjust={adjust}"

    # ---------- Path helpers ----------

    def year_dir(self, year: int) -> Path:
        return self._raw_root / f"year={year}"

    def partition_path(self, symbol: str, year: int) -> Path:
        return self.year_dir(year) / f"symbol={symbol}.parquet"

    def has_partition(self, symbol: str, year: int) -> bool:
        return self.partition_path(symbol, year).exists()

    def list_symbols_with_data(self) -> set[str]:
        """All distinct symbols that have at least one partition under this store."""
        if not self._raw_root.exists():
            return set()
        out: set[str] = set()
        for year_dir in self._raw_root.glob("year=*"):
            for f in year_dir.glob("symbol=*.parquet"):
                # 'symbol=SH600519.parquet' -> 'SH600519'
                stem = f.name.removeprefix("symbol=").removesuffix(".parquet")
                out.add(stem)
        return out

    # ---------- Writes ----------

    def write_ohlcv(self, df: pl.DataFrame) -> dict[str, int]:
        """Write `df` (OHLCV_SCHEMA) into the partitioned store, deduping on
        (symbol, date) against any existing data. Returns a dict with the
        keys 'written' (rows newly persisted) and 'skipped' (rows that were
        duplicates and thus not written).
        """
        if df.is_empty():
            return {"written": 0, "skipped": 0}
        self._validate_schema(df)
        # Group by (year, symbol) → batch writes.
        df_with_year = df.with_columns(pl.col("date").dt.year().alias("_year"))
        written_total = 0
        skipped_total = 0
        for (year, symbol), batch in df_with_year.group_by(
            ["_year", "symbol"], maintain_order=True
        ):
            new_rows = batch.drop("_year")
            written, skipped = self._write_partition(symbol, int(year), new_rows)
            written_total += written
            skipped_total += skipped
        return {"written": written_total, "skipped": skipped_total}

    def _write_partition(
        self, symbol: str, year: int, new_rows: pl.DataFrame
    ) -> tuple[int, int]:
        path = self.partition_path(symbol, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pl.read_parquet(path)
            existing_dates = set(existing["date"].to_list())
            new_only = new_rows.filter(~pl.col("date").is_in(list(existing_dates)))
            if new_only.is_empty():
                return 0, new_rows.height
            merged = (
                pl.concat([existing, new_only], how="vertical")
                .sort(["date", "symbol"])
                .unique(subset=["date", "symbol"], keep="first")
            )
            merged.write_parquet(path)
            return new_only.height, new_rows.height - new_only.height
        else:
            new_rows.sort(["date", "symbol"]).write_parquet(path)
            return new_rows.height, 0

    # ---------- Reads ----------

    def read_ohlcv(
        self,
        symbols: Iterable[str],
        start: dt.date,
        end: dt.date,
    ) -> pl.LazyFrame:
        """Return a LazyFrame of all rows for `symbols` between `start` and
        `end` inclusive. Missing partitions are silently treated as empty.
        """
        symbols = list(symbols)
        if not symbols:
            return pl.LazyFrame(schema=OHLCV_SCHEMA)
        if start > end:
            return pl.LazyFrame(schema=OHLCV_SCHEMA)
        years = range(start.year, end.year + 1)
        paths: list[Path] = []
        for y in years:
            for s in symbols:
                p = self.partition_path(s, y)
                if p.exists():
                    paths.append(p)
        if not paths:
            return pl.LazyFrame(schema=OHLCV_SCHEMA)
        return (
            pl.scan_parquet([str(p) for p in paths])
            .filter(pl.col("date").is_between(start, end, closed="both"))
            .filter(pl.col("symbol").is_in(symbols))
        )

    # ---------- Validation ----------

    @staticmethod
    def _validate_schema(df: pl.DataFrame) -> None:
        missing = set(OHLCV_SCHEMA) - set(df.columns)
        if missing:
            raise StorageError(
                f"OHLCV frame missing columns {sorted(missing)}; "
                f"got {df.columns}"
            )
        for col, expected_dtype in OHLCV_SCHEMA.items():
            actual = df.schema[col]
            if actual != expected_dtype:
                raise StorageError(
                    f"column {col!r}: expected dtype {expected_dtype}, got {actual}"
                )
