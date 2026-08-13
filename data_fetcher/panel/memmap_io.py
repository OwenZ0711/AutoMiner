"""Single owner of the panel binary layout.

Per year, per channel/feature/mask: one raw C-order memmap file of shape
(T, N) — float32 (``.f32``, NaN = invalid) or uint8 (``.u8``, 0/1). Product
order is pinned in ``meta.json`` (sorted alphabetically); both bands read the
order from meta and never assume it. Time-major layout: the lead-lag GEMM
``Z[:-l].T @ Z[l:]`` wants contiguous time rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np

from data_fetcher.exceptions import StorageError
from futures_common.paths import FuturesPaths

PANEL_VERSION = 1


class PanelMeta(TypedDict):
    version: int
    year: int
    T: int
    N: int
    products: list[str]
    channels: list[str]
    masks: list[str]
    features: list[str]
    config_hash: str


def write_meta(paths: FuturesPaths, meta: PanelMeta) -> None:
    path = paths.panel_meta(meta["year"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp, path)


def read_meta(paths: FuturesPaths, year: int) -> PanelMeta:
    path = paths.panel_meta(year)
    if not path.exists():
        raise StorageError(f"panel meta missing: {path}")
    raw = json.loads(path.read_text())
    if raw.get("version") != PANEL_VERSION:
        raise StorageError(f"{path}: version {raw.get('version')} != {PANEL_VERSION}")
    return cast("PanelMeta", raw)


def _mmap_path(paths: FuturesPaths, year: int, kind: str, name: str) -> Path:
    if kind == "channel":
        return paths.panel_channel(year, name)
    if kind == "mask":
        return paths.panel_mask(year, name)
    if kind == "feature":
        return paths.panel_feature(year, name)
    raise ValueError(kind)


def create_array(
    paths: FuturesPaths,
    year: int,
    kind: Literal["channel", "mask", "feature"],
    name: str,
    shape: tuple[int, int],
) -> np.memmap:
    """Create (or overwrite) a memmap, fill with NaN (f32) / 0 (u8)."""
    path = _mmap_path(paths, year, kind, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if kind == "mask" else np.float32
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    arr[:] = 0 if kind == "mask" else np.nan
    return arr


def open_array(
    paths: FuturesPaths,
    year: int,
    kind: Literal["channel", "mask", "feature"],
    name: str,
    *,
    mode: Literal["r", "r+"] = "r",
) -> np.memmap:
    meta = read_meta(paths, year)
    path = _mmap_path(paths, year, kind, name)
    if not path.exists():
        raise StorageError(f"panel array missing: {path}")
    dtype = np.uint8 if kind == "mask" else np.float32
    shape = (meta["T"], meta["N"])
    expected = shape[0] * shape[1] * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise StorageError(f"{path}: size {actual} != expected {expected} for shape {shape}")
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
