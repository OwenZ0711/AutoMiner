"""Per-run manifest: config, seed, git state, peak RSS — reproducibility record.

Every stage writes exactly one manifest under
``results/runs/{stage}/{run_id}/manifest.json`` and updates the
``results/runs/latest.json`` pointer. ``config_hash`` here is the same value
stamped into the L4 edge table and the trial ledger — one definition.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from futures_common.paths import FuturesPaths, repo_root


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, dates/paths/tuples normalized."""
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"))


def _normalize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [_normalize(v) for v in obj]
        return sorted(items, key=repr) if isinstance(obj, (set, frozenset)) else items
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return _normalize(asdict(obj))
    return repr(obj)


def config_hash(snapshot: Any) -> str:
    """sha256 hex of the canonical JSON of a resolved config snapshot."""
    return hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()


def git_state() -> tuple[str, bool]:
    """(HEAD hash, dirty?) — ("unknown", False) outside a repo or on failure."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return head, bool(porcelain.strip())
    except (subprocess.SubprocessError, OSError):
        return "unknown", False


def _package_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("polars", "numpy", "pandas", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:  # pragma: no cover
            pass
    return out


@dataclass(frozen=True, slots=True)
class RunManifest:
    stage: str
    run_id: str
    started_at: str
    finished_at: str
    seed: int
    argv: list[str]
    config_snapshot: dict[str, Any]
    config_hash: str
    git_hash: str
    git_dirty: bool
    peak_rss_bytes: int
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=_package_versions)
    exit_status: str = "ok"
    report: dict[str, Any] = field(default_factory=dict)


def make_run_id(cfg_hash: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    return f"{now:%Y%m%d-%H%M%S}_{cfg_hash[:8]}"


def write_manifest(m: RunManifest, paths: FuturesPaths) -> Path:
    run_dir = paths.run_dir(m.stage, m.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "manifest.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_normalize(asdict(m)), indent=2, sort_keys=True))
    os.replace(tmp, out)
    _update_latest(m.stage, m.run_id, paths)
    return out


def _update_latest(stage: str, run_id: str, paths: FuturesPaths) -> None:
    latest = paths.latest_runs
    latest.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, str] = {}
    if latest.exists():
        try:
            current = json.loads(latest.read_text())
        except json.JSONDecodeError:
            current = {}
    current[stage] = run_id
    tmp = latest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True))
    os.replace(tmp, latest)


def read_latest(stage: str, paths: FuturesPaths) -> RunManifest | None:
    latest = paths.latest_runs
    if not latest.exists():
        return None
    pointer = json.loads(latest.read_text())
    run_id = pointer.get(stage)
    if run_id is None:
        return None
    manifest_path = paths.run_dir(stage, run_id) / "manifest.json"
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_text())
    known = {f for f in RunManifest.__dataclass_fields__}
    return RunManifest(**{k: v for k, v in raw.items() if k in known})
