"""Shared mining-band fixtures: synthetic panels written in the on-disk format."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from futures_common.paths import FuturesPaths


@pytest.fixture()
def tmp_paths(tmp_path: Path) -> FuturesPaths:
    return FuturesPaths(data_root=tmp_path / "data", results_root=tmp_path / "results")


def write_panel_year(
    paths: FuturesPaths,
    year: int,
    products: list[str],
    x_std: np.ndarray,  # (T, N) f32, NaN = invalid
    m_traded: np.ndarray,  # (T, N) bool
    trading_dates: np.ndarray,  # (T,) epoch days int32
    seg_session: np.ndarray,  # (T,) int32
    *,
    extra_features: dict[str, np.ndarray] | None = None,
    extra_channels: dict[str, np.ndarray] | None = None,
    m_limit: np.ndarray | None = None,
    roll_flag: np.ndarray | None = None,
    rel_min: np.ndarray | None = None,
    seg_macro: np.ndarray | None = None,
) -> None:
    """Write a synthetic panel year in the exact on-disk format."""
    t_rows, n = x_std.shape
    pdir = paths.panel_dir(year)
    (pdir / "features").mkdir(parents=True, exist_ok=True)
    (pdir / "masks").mkdir(parents=True, exist_ok=True)
    (pdir / "channels").mkdir(parents=True, exist_ok=True)

    def _write(sub: str, name: str, arr: np.ndarray, dtype: type) -> None:
        suffix = "u8" if dtype is np.uint8 else "f32"
        mm = np.memmap(pdir / sub / f"{name}.{suffix}", dtype=dtype, mode="w+", shape=(t_rows, n))
        mm[:] = arr.astype(dtype)
        mm.flush()

    _write("features", "x_std", x_std, np.float32)
    for name, arr in (extra_features or {}).items():
        _write("features", name, arr, np.float32)
    for name, arr in (extra_channels or {}).items():
        _write("channels", name, arr, np.float32)
    _write("masks", "m_traded", m_traded.astype(np.uint8), np.uint8)
    _write(
        "masks",
        "m_limit",
        (m_limit if m_limit is not None else np.zeros((t_rows, n))).astype(np.uint8),
        np.uint8,
    )
    _write(
        "masks",
        "roll_flag",
        (roll_flag if roll_flag is not None else np.zeros((t_rows, n))).astype(np.uint8),
        np.uint8,
    )

    epoch = dt.date(1970, 1, 1)
    feature_names = ["x_std", *(extra_features or {})]
    channel_names = list(extra_channels or {})
    if rel_min is None:
        # synthesize a within-day sequence
        rel = np.zeros(t_rows, dtype=np.int32)
        _, counts = np.unique(trading_dates, return_counts=True)
        pos = 0
        for c in counts:
            rel[pos : pos + c] = 750 + np.arange(c)
            pos += c
        rel_min = rel
    idx = pl.DataFrame(
        {
            "row": np.arange(t_rows, dtype=np.uint32),
            "trading_date": [epoch + dt.timedelta(days=int(d)) for d in trading_dates],
            "rel_min": rel_min.astype(np.int32),
            "bob": [
                dt.datetime.combine(epoch + dt.timedelta(days=int(d)), dt.time(9, 0))
                + dt.timedelta(minutes=int(k))
                for k, d in zip(rel_min - rel_min.min(), trading_dates, strict=True)
            ],
            "seg_session": seg_session.astype(np.int32),
            "seg_macro": (seg_macro if seg_macro is not None else np.zeros(t_rows)).astype(np.int8),
        }
    ).with_columns(pl.col("bob").dt.replace_time_zone("Asia/Shanghai"))
    idx.write_parquet(paths.bob_index(year))

    meta = {
        "version": 1,
        "year": year,
        "T": t_rows,
        "N": n,
        "products": products,
        "channels": channel_names,
        "masks": ["m_traded", "m_limit", "roll_flag"],
        "features": feature_names,
        "config_hash": "test",
    }
    paths.panel_meta(year).write_text(json.dumps(meta))
