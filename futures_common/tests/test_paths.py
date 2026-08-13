"""Path-convention tests: round-trip under tmp_path, env override, casing."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from futures_common.paths import FuturesPaths, default_paths


def _paths(tmp_path: Path) -> FuturesPaths:
    return FuturesPaths(data_root=tmp_path / "data", results_root=tmp_path / "results")


def test_raw_contract_layout_and_uppercase(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    path = p.raw_contract("shfe", "rb", "rb2001")
    assert path == tmp_path / "data" / "raw" / "exchange=SHFE" / "product=RB" / "RB2001.parquet"


def test_panel_layout(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    assert p.panel_channel(2020, "close_adj").name == "close_adj.f32"
    assert p.panel_mask(2020, "m_traded").name == "m_traded.u8"
    assert p.panel_feature(2020, "x_std").parent.name == "features"
    assert p.panel_meta(2020).parent == p.panel_dir(2020)
    assert "year=2020" in str(p.panel_dir(2020))


def test_edges_file_naming(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    f = p.edges_file(dt.date(2024, 6, 30), "abcdef0123456789")
    assert f.name == "asof=2024-06-30_cfg=abcdef012345.parquet"
    assert f.parent == p.edges_dir


def test_results_paths(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    assert p.run_dir("ingest", "r1") == tmp_path / "results" / "runs" / "ingest" / "r1"
    assert p.latest_runs.name == "latest.json"


def test_env_override(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AUTOLLM_DATA_ROOT", str(tmp_path / "elsewhere"))
    p = default_paths()
    assert p.data_root == tmp_path / "elsewhere"
