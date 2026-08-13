"""Manifest write/read round-trip and latest-pointer semantics."""

from __future__ import annotations

from pathlib import Path

from futures_common.manifest import (
    RunManifest,
    config_hash,
    make_run_id,
    read_latest,
    write_manifest,
)
from futures_common.paths import FuturesPaths


def _manifest(stage: str, run_id: str) -> RunManifest:
    snapshot = {"products": ["RB"], "seed": 42}
    return RunManifest(
        stage=stage,
        run_id=run_id,
        started_at="2026-07-13T10:00:00+08:00",
        finished_at="2026-07-13T10:05:00+08:00",
        seed=42,
        argv=["ingest", "--slice", "acceptance"],
        config_snapshot=snapshot,
        config_hash=config_hash(snapshot),
        git_hash="deadbeef",
        git_dirty=True,
        peak_rss_bytes=123_456_789,
        report={"rows_written": 10},
    )


def test_write_and_read_latest(tmp_path: Path) -> None:
    paths = FuturesPaths(data_root=tmp_path / "d", results_root=tmp_path / "r")
    m = _manifest("ingest", "run1")
    out = write_manifest(m, paths)
    assert out.exists()

    back = read_latest("ingest", paths)
    assert back is not None
    assert back.run_id == "run1"
    assert back.peak_rss_bytes == 123_456_789
    assert back.report == {"rows_written": 10}


def test_latest_pointer_tracks_stages_independently(tmp_path: Path) -> None:
    paths = FuturesPaths(data_root=tmp_path / "d", results_root=tmp_path / "r")
    write_manifest(_manifest("ingest", "run1"), paths)
    write_manifest(_manifest("panel", "run2"), paths)
    write_manifest(_manifest("ingest", "run3"), paths)

    latest_ingest = read_latest("ingest", paths)
    latest_panel = read_latest("panel", paths)
    assert latest_ingest is not None and latest_ingest.run_id == "run3"
    assert latest_panel is not None and latest_panel.run_id == "run2"
    assert read_latest("mine", paths) is None


def test_make_run_id_embeds_hash_prefix() -> None:
    h = config_hash({"a": 1})
    rid = make_run_id(h)
    assert rid.endswith(h[:8])
