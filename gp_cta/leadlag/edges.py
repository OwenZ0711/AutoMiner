"""L4: append-only, as-of dated edge table.

One parquet per append under ``~/AutoLLM_data/futures/edges/`` — never
rewritten; re-scans append under a new config_hash. ``as_of(date)`` is THE
single sanctioned path by which GP touches foreign series (look-ahead safety
§4): only rows with ``asof_date ≤ date − 1 trading day`` are visible, latest
asof per (leader, follower) wins, leaders ranked per follower by |a_stat|.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from futures_common.paths import FuturesPaths

EDGE_SCHEMA: dict[str, object] = {
    "asof_date": pl.Date,
    "leader": pl.Utf8,
    "follower": pl.Utf8,
    "lag_bars": pl.Int16,
    "strength": pl.Float32,
    "sign": pl.Int8,
    "a_stat": pl.Float32,
    "p_value": pl.Float32,
    "q_value": pl.Float32,
    "family": pl.Utf8,
    "n_eff": pl.Int64,
    "stability": pl.Float32,
    "half_life": pl.Float32,
    "regime_flags": pl.Utf8,
    "window_start": pl.Date,
    "window_end": pl.Date,
    "config_hash": pl.Utf8,
}


@dataclass(frozen=True, slots=True)
class Edge:
    leader: str
    follower: str
    lag_bars: int
    strength: float
    sign: int
    a_stat: float
    q_value: float
    asof_date: dt.date


class EdgeStore:
    def __init__(self, paths: FuturesPaths) -> None:
        self.paths = paths

    def append(self, edges: pl.DataFrame, asof: dt.date, config_hash: str) -> None:
        missing = set(EDGE_SCHEMA) - set(edges.columns) - {"asof_date", "config_hash"}
        if missing:
            raise ValueError(f"edge frame missing columns: {sorted(missing)}")
        out = edges.with_columns(
            pl.lit(asof).alias("asof_date"), pl.lit(config_hash).alias("config_hash")
        ).select(list(EDGE_SCHEMA))
        out = out.cast(EDGE_SCHEMA)  # type: ignore[arg-type]
        path = self.paths.edges_file(asof, config_hash)
        if path.exists():
            raise FileExistsError(f"edge file already exists (append-only): {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path)

    def _scan(self) -> pl.LazyFrame:
        files = sorted(self.paths.edges_dir.glob("asof=*.parquet"))
        if not files:
            return pl.LazyFrame(schema=EDGE_SCHEMA)  # type: ignore[arg-type]
        return pl.scan_parquet([str(f) for f in files])

    def as_of(
        self,
        date: dt.date,
        *,
        q_max: float = 0.10,
        min_stability: float = 0.0,
        max_staleness_days: int | None = 45,
    ) -> pl.DataFrame:
        """Validated edges visible at `date` (asof strictly before `date`).

        Latest asof_date per (leader, follower) wins; ties broken by
        config_hash for determinism.
        """
        lf = self._scan().filter(pl.col("asof_date") < date)
        if max_staleness_days is not None:
            lf = lf.filter(pl.col("asof_date") >= date - dt.timedelta(days=max_staleness_days))
        df = (
            lf.sort(["asof_date", "config_hash"], descending=[True, True])
            .group_by(["leader", "follower"], maintain_order=True)
            .first()
            .collect()
        )
        return df.filter((pl.col("q_value") <= q_max) & (pl.col("stability") >= min_stability))

    def leaders_for(
        self,
        follower: str,
        date: dt.date,
        *,
        top_k: int = 3,
        q_max: float = 0.10,
        min_stability: float = 0.0,
        max_staleness_days: int | None = 45,
    ) -> list[Edge]:
        df = (
            self.as_of(
                date,
                q_max=q_max,
                min_stability=min_stability,
                max_staleness_days=max_staleness_days,
            )
            .filter(pl.col("follower") == follower.upper())
            .with_columns(pl.col("a_stat").abs().alias("_rank"))
            .sort("_rank", descending=True)
            .head(top_k)
        )
        return [
            Edge(
                leader=r["leader"],
                follower=r["follower"],
                lag_bars=int(r["lag_bars"]),
                strength=float(r["strength"]),
                sign=int(r["sign"]),
                a_stat=float(r["a_stat"]),
                q_value=float(r["q_value"]),
                asof_date=r["asof_date"],
            )
            for r in df.iter_rows(named=True)
        ]
