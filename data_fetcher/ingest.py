"""L0 ingest: raw CSV trees → one parquet per real contract.

Output: ``~/AutoLLM_data/futures/raw/exchange={E}/product={P}/{CONTRACT}.parquet``
sorted by bob, plus ``raw/_manifest.parquet`` for idempotency.

Loop unit (memory rule 1): one contract at a time — the largest raw contract
CSV is ~35 MB (INE/SC), so the eager per-contract frame stays well under the
1 GB ingest budget. A contract spanning both trees (e.g. RB2605) is built
from the historical file plus all its 2026 day files in one pass, deduped on
bob (keep-first, historical wins).

Range semantics: [start, end] selects WHICH contracts to touch (lifetime
intersection); a touched contract's parquet always holds its FULL lifetime of
valid rows. Overlapping slice runs therefore never shrink a contract, and the
per-contract fingerprint depends only on its sources + code version.

Idempotency: a contract is rebuilt only when its fingerprint — source paths,
mtimes, sizes, ingest code version — differs from the manifest row, or with
``force=True``. Writes are atomic (tmp → os.replace) and deterministically
sorted, so an unchanged re-run touches nothing and a rebuild from identical
inputs is byte-identical.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from loguru import logger

from data_fetcher.exceptions import SchemaDriftError
from data_fetcher.readers import DailyTreeReader, HistoricalTreeReader, RawBatch, RawReader
from data_fetcher.trading_days import TradingDays, assign_trading_date, build_bootstrap_trading_days
from futures_common.paths import FuturesPaths

INGEST_VERSION = "1"

OUTPUT_COLUMNS = (
    "exchange",
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
    "trading_date",
)

_MANIFEST_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "contract": pl.Utf8,
    "exchange": pl.Utf8,
    "product": pl.Utf8,
    "fingerprint": pl.Utf8,
    "n_rows": pl.Int64,
    "written_at": pl.Utf8,
}


@dataclass(slots=True)
class IngestReport:
    contracts_written: int = 0
    contracts_skipped: int = 0
    contracts_empty: int = 0
    rows_written: int = 0
    rows_dropped_unmapped: int = 0
    products: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "contracts_written": self.contracts_written,
            "contracts_skipped": self.contracts_skipped,
            "contracts_empty": self.contracts_empty,
            "rows_written": self.rows_written,
            "rows_dropped_unmapped": self.rows_dropped_unmapped,
            "products": self.products,
        }


def _fingerprint(sources: Iterable[Path], params_hash: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(sources):
        st = p.stat()
        h.update(f"{p}|{st.st_mtime_ns}|{st.st_size}\n".encode())
    h.update(f"v={INGEST_VERSION}|{params_hash}".encode())
    return h.hexdigest()


def _load_manifest(paths: FuturesPaths) -> pl.DataFrame:
    if paths.raw_manifest.exists():
        return pl.read_parquet(paths.raw_manifest)
    return pl.DataFrame(schema=_MANIFEST_SCHEMA)


def _save_manifest(paths: FuturesPaths, manifest: pl.DataFrame) -> None:
    paths.raw_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.raw_manifest.with_suffix(".parquet.tmp")
    manifest.sort("contract").write_parquet(tmp)
    os.replace(tmp, paths.raw_manifest)


def _collect_contract(batches: Sequence[RawBatch]) -> pl.DataFrame:
    frames = [b.lf.collect(engine="streaming") for b in batches]
    df = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")
    contract = batches[0].contract
    if df.is_empty():
        return df.drop("symbol")
    bad = df.filter(pl.col("symbol") != contract)
    if not bad.is_empty():
        raise SchemaDriftError(
            f"{batches[0].sources[0]}: {bad.height} rows whose symbol "
            f"{bad['symbol'][0]!r} != contract {contract!r}"
        )
    return (
        df.drop("symbol")
        .sort("bob", maintain_order=True)
        .unique(subset=["bob"], keep="first", maintain_order=True)
    )


def _write_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def make_readers(
    raw_historical: Path,
    raw_2026: Path,
    *,
    tree: str,
    start: dt.date | None,
    end: dt.date | None,
) -> list[RawReader]:
    readers: list[RawReader] = []
    if tree in ("historical", "both") and raw_historical.is_dir():
        readers.append(HistoricalTreeReader(raw_historical, start=start, end=end))
    if tree in ("2026", "both") and raw_2026.is_dir():
        readers.append(DailyTreeReader(raw_2026, start=start, end=end))
    if not readers:
        raise FileNotFoundError(
            f"no raw tree found for tree={tree!r} under {raw_historical} / {raw_2026}"
        )
    return readers


def ingest(
    paths: FuturesPaths,
    *,
    raw_historical: Path,
    raw_2026: Path,
    products: Sequence[str] | None,
    start: dt.date,
    end: dt.date,
    tree: str = "both",
    anchors: Sequence[str] = ("CU", "AL", "RU", "M", "C", "TA", "CF", "SR", "RB", "IF"),
    anchors_required: int = 3,
    force: bool = False,
    trading_days: TradingDays | None = None,
) -> IngestReport:
    """Ingest the requested slice; see module docstring for semantics."""
    readers = make_readers(raw_historical, raw_2026, tree=tree, start=start, end=end)
    if trading_days is None:
        trading_days = build_bootstrap_trading_days(
            readers,
            anchors=anchors,
            start=start,
            end=end,
            anchors_required=anchors_required,
            paths=paths,
        )

    params_hash = ""  # fingerprint = sources + code version only (full-lifetime writes)
    manifest = _load_manifest(paths)
    known_fp: dict[str, str] = dict(zip(manifest["contract"], manifest["fingerprint"], strict=True))
    touched: list[dict[str, object]] = []

    report = IngestReport()
    all_products = sorted({p for r in readers for p in r.discover(products)})
    report.products = all_products

    for product in all_products:
        # gather every batch (across trees) for this product, grouped by contract
        by_contract: dict[str, list[RawBatch]] = {}
        for reader in readers:
            for batch in reader.iter_product(product):
                by_contract.setdefault(batch.contract, []).append(batch)

        for contract in sorted(by_contract):
            batches = by_contract[contract]
            sources = [p for b in batches for p in b.sources]
            fp = _fingerprint(sources, params_hash)
            if not force and known_fp.get(contract) == fp:
                report.contracts_skipped += 1
                continue

            df = _collect_contract(batches)
            exchange = df["exchange"][0] if not df.is_empty() else "UNKNOWN"
            df, dropped = assign_trading_date(df, trading_days)
            report.rows_dropped_unmapped += dropped
            df = df.select(OUTPUT_COLUMNS)

            out_path = paths.raw_contract(exchange, product, contract)
            if df.is_empty():
                report.contracts_empty += 1
                if out_path.exists():
                    out_path.unlink()
            else:
                _write_atomic(df, out_path)
                report.contracts_written += 1
                report.rows_written += df.height
            touched.append(
                {
                    "contract": contract,
                    "exchange": exchange,
                    "product": product,
                    "fingerprint": fp,
                    "n_rows": df.height,
                    "written_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
        logger.info(
            "ingest {}: {} contracts done ({} skipped so far)",
            product,
            len(by_contract),
            report.contracts_skipped,
        )

    if touched:
        touched_df = pl.DataFrame(touched, schema=_MANIFEST_SCHEMA)
        manifest = pl.concat(
            [
                manifest.filter(~pl.col("contract").is_in(touched_df["contract"].implode())),
                touched_df,
            ],
            how="vertical",
        )
        _save_manifest(paths, manifest)
    return report
