"""Read-side contract with the data band's panel artifacts.

Reads ``panel/year=YYYY/{meta.json, bob_index.parquet, channels/*.f32,
masks/*.u8, features/*.f32}`` directly (json + raw C-order memmaps) so the
mining band never imports data-band code. The layout is pinned in
``gp_cta/proposal.md`` § Target architecture; the e2e test guards drift.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from futures_common.paths import FuturesPaths


class PanelError(RuntimeError):
    """Panel artifact missing or inconsistent."""


@dataclass(slots=True)
class PanelReader:
    paths: FuturesPaths
    years: tuple[int, ...]
    _meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    _index: dict[int, pl.DataFrame] = field(default_factory=dict)

    def __init__(self, paths: FuturesPaths, years: Sequence[int]) -> None:
        self.paths = paths
        self.years = tuple(sorted(years))
        self._meta = {}
        self._index = {}
        products: list[str] | None = None
        for year in self.years:
            meta_path = paths.panel_meta(year)
            if not meta_path.exists():
                raise PanelError(f"panel meta missing for {year}: {meta_path}")
            meta = json.loads(meta_path.read_text())
            if products is None:
                products = list(meta["products"])
            elif products != list(meta["products"]):
                raise PanelError(f"product order differs across years: {year}")
            self._meta[year] = meta
        if products is None:
            raise PanelError("no years given")

    @property
    def products(self) -> tuple[str, ...]:
        return tuple(self._meta[self.years[0]]["products"])

    @property
    def n_products(self) -> int:
        return len(self.products)

    def shape(self, year: int) -> tuple[int, int]:
        m = self._meta[year]
        return int(m["T"]), int(m["N"])

    def _open(
        self, year: int, kind: Literal["channel", "mask", "feature"], name: str
    ) -> np.ndarray:
        if kind == "channel":
            path = self.paths.panel_channel(year, name)
        elif kind == "mask":
            path = self.paths.panel_mask(year, name)
        else:
            path = self.paths.panel_feature(year, name)
        if not path.exists():
            raise PanelError(f"missing panel array: {path}")
        dtype = np.uint8 if kind == "mask" else np.float32
        shape = self.shape(year)
        expected = shape[0] * shape[1] * np.dtype(dtype).itemsize
        if path.stat().st_size != expected:
            raise PanelError(f"{path}: size mismatch for shape {shape}")
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def feature(self, name: str, year: int) -> np.ndarray:
        return self._open(year, "feature", name)

    def channel(self, name: str, year: int) -> np.ndarray:
        return self._open(year, "channel", name)

    def mask(self, name: str, year: int) -> np.ndarray:
        return self._open(year, "mask", name)

    def index(self, year: int) -> pl.DataFrame:
        if year not in self._index:
            self._index[year] = pl.read_parquet(self.paths.bob_index(year))
        return self._index[year]

    def segment_spans(self, year: int) -> list[tuple[int, int]]:
        """[start, end) row spans of seg_session — the GEMM blocking source."""
        seg = self.index(year)["seg_session"].to_numpy()
        if seg.size == 0:
            return []
        changes = np.flatnonzero(np.diff(seg)) + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [seg.size]])
        return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]

    def trading_dates(self, year: int) -> np.ndarray:
        """Per-row trading date as epoch days (int32)."""
        return self.index(year)["trading_date"].cast(pl.Int32).to_numpy()
