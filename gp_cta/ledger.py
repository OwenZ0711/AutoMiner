"""Append-only trial ledger — the band-3 DSR gate's denominator.

EVERY hypothesis examined is a row: each (pair, window) the lead-lag scan
looks at (significant or not), each GP formula backtested, each one-time sign
flip. Rows are buffered and flushed to ``results/trial_ledger/part-*.parquet``;
nothing is ever rewritten.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from typing import Any

import polars as pl

from futures_common.manifest import canonical_json
from futures_common.paths import FuturesPaths

_SCHEMA: dict[str, Any] = {
    "trial_id": pl.Utf8,
    "kind": pl.Utf8,  # ll_pair_scan | gp_formula | gp_sign_flip
    "asof_date": pl.Date,
    "detail": pl.Utf8,  # "I→RB w=2019..2021" or ast_hash
    "seed_id": pl.Utf8,
    "split": pl.Utf8,
    "stat": pl.Float64,
    "metrics_json": pl.Utf8,
    "config_hash": pl.Utf8,
    "ts": pl.Datetime("us"),
}


class TrialLedger:
    def __init__(self, paths: FuturesPaths, config_hash: str = "") -> None:
        self.paths = paths
        self.config_hash = config_hash
        self._buffer: list[dict[str, Any]] = []

    def log(
        self,
        kind: str,
        detail: str,
        *,
        asof_date: dt.date | None = None,
        seed_id: str | None = None,
        split: str | None = None,
        stat: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        trial_id = hashlib.blake2b(
            f"{kind}|{detail}|{self.config_hash}".encode(), digest_size=12
        ).hexdigest()
        self._buffer.append(
            {
                "trial_id": trial_id,
                "kind": kind,
                "asof_date": asof_date,
                "detail": detail,
                "seed_id": seed_id,
                "split": split,
                "stat": stat,
                "metrics_json": canonical_json(metrics) if metrics else None,
                "config_hash": self.config_hash,
                "ts": dt.datetime.now(),
            }
        )
        if len(self._buffer) >= 5000:
            self.flush()
        return trial_id

    def flush(self) -> None:
        if not self._buffer:
            return
        out_dir = self.paths.trial_ledger_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        part = out_dir / f"part-{uuid.uuid4().hex[:8]}.parquet"
        pl.DataFrame(self._buffer, schema=_SCHEMA).write_parquet(part)
        self._buffer.clear()

    def count(self, kind: str | None = None) -> int:
        """Total trials on disk + buffered (the DSR trial count)."""
        n = sum(1 for r in self._buffer if kind is None or r["kind"] == kind)
        if self.paths.trial_ledger_dir.exists():
            parts = list(self.paths.trial_ledger_dir.glob("part-*.parquet"))
            if parts:
                lf = pl.scan_parquet([str(p) for p in parts])
                if kind is not None:
                    lf = lf.filter(pl.col("kind") == kind)
                n += int(lf.select(pl.len()).collect().item())
        return n

    def __enter__(self) -> TrialLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()


def scan_ledger(paths: FuturesPaths) -> pl.LazyFrame:
    parts = list(paths.trial_ledger_dir.glob("part-*.parquet"))
    if not parts:
        return pl.LazyFrame(schema=_SCHEMA)
    return pl.scan_parquet([str(p) for p in parts])


__all__ = ["TrialLedger", "scan_ledger"]
