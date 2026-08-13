"""Shared stage-CLI plumbing: common flags, slice resolution, the run wrapper.

Every band subcommand follows the same shape:

    setup_logging → resolve StageContext → PeakTracker(budget) → work
    → write_manifest (config, seed, git state, peak RSS)

Exit codes: 0 ok, 1 crash, 2 completed-with-failures, 3 memory budget.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from futures_common.config import Defaults, load_defaults
from futures_common.log import setup_logging
from futures_common.manifest import (
    RunManifest,
    config_hash,
    git_state,
    make_run_id,
    write_manifest,
)
from futures_common.memory import MemoryBudgetError, PeakTracker
from futures_common.paths import FuturesPaths, default_paths

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_PARTIAL = 2
EXIT_MEMORY = 3


def add_common_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--slice", dest="slice_name", help="slice from configs/defaults.toml")
    group.add_argument("--products", help="comma list, e.g. RB,I,TA")
    parser.add_argument("--start", help="YYYY-MM-DD (with --products)")
    parser.add_argument("--end", help="YYYY-MM-DD (with --products)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    parser.add_argument("--budget-gb", type=float, default=None, help="override memory budget")
    parser.add_argument("--log-level", default="INFO")


@dataclass(frozen=True, slots=True)
class StageContext:
    stage: str
    paths: FuturesPaths
    defaults: Defaults
    products: tuple[str, ...]
    start: dt.date
    end: dt.date
    seed: int
    force: bool
    budget_gb: float


def resolve_context(args: argparse.Namespace, stage: str) -> StageContext:
    defaults = load_defaults()
    slice_name = args.slice_name or ("acceptance" if not args.products else None)
    if slice_name is not None:
        if slice_name not in defaults.slices:
            raise SystemExit(f"unknown slice {slice_name!r}; defined: {sorted(defaults.slices)}")
        sl = defaults.slices[slice_name]
        products, start, end = sl.products, sl.start, sl.end
    else:
        if not (args.start and args.end):
            raise SystemExit("--products requires --start and --end")
        products = tuple(p.strip().upper() for p in args.products.split(",") if p.strip())
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)

    paths = default_paths(data_root=args.data_root)
    seed = args.seed if args.seed is not None else defaults.project.seed
    budget = args.budget_gb if args.budget_gb is not None else defaults.memory.budget_for(stage)
    return StageContext(
        stage=stage,
        paths=paths,
        defaults=defaults,
        products=products,
        start=start,
        end=end,
        seed=seed,
        force=bool(args.force),
        budget_gb=budget,
    )


def run_stage(
    stage: str,
    args: argparse.Namespace,
    work: Callable[[StageContext], Mapping[str, Any]],
    *,
    extra_config: Mapping[str, Any] | None = None,
) -> int:
    """Run one stage under logging + memory tracking + manifest writing."""
    setup_logging(stage, level=getattr(args, "log_level", "INFO"))
    ctx = resolve_context(args, stage)

    snapshot: dict[str, Any] = {
        "stage": stage,
        "products": list(ctx.products),
        "start": ctx.start,
        "end": ctx.end,
        "seed": ctx.seed,
        "force": ctx.force,
        **(dict(extra_config) if extra_config else {}),
    }
    cfg_hash = config_hash(snapshot)
    run_id = make_run_id(cfg_hash)
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    exit_code = EXIT_OK
    report: Mapping[str, Any] = {}
    tracker = PeakTracker(stage, budget_gb=ctx.budget_gb, hard=False)
    try:
        with tracker:
            report = work(ctx)
        if tracker.peak_rss_bytes > ctx.budget_gb * 1024**3:
            exit_code = EXIT_MEMORY
    except MemoryBudgetError:
        logger.exception("{} exceeded its memory budget", stage)
        exit_code = EXIT_MEMORY
    except Exception:
        logger.error("stage {} crashed:\n{}", stage, traceback.format_exc())
        exit_code = EXIT_CRASH

    head, dirty = git_state()
    manifest = RunManifest(
        stage=stage,
        run_id=run_id,
        started_at=started,
        finished_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        seed=ctx.seed,
        argv=sys.argv[1:],
        config_snapshot=snapshot,
        config_hash=cfg_hash,
        git_hash=head,
        git_dirty=dirty,
        peak_rss_bytes=tracker.peak_rss_bytes,
        exit_status={
            EXIT_OK: "ok",
            EXIT_CRASH: "failed",
            EXIT_PARTIAL: "partial",
            EXIT_MEMORY: "killed:memory",
        }[exit_code],
        report=dict(report),
    )
    manifest_path = write_manifest(manifest, ctx.paths)
    logger.info(
        "{} finished: exit={} peak_rss={:.2f} GB manifest={}",
        stage,
        exit_code,
        tracker.peak_rss_bytes / 1024**3,
        manifest_path,
    )
    return exit_code


def repo_raw_roots() -> tuple[Path, Path]:
    from futures_common.paths import DEFAULT_RAW_2026, DEFAULT_RAW_HISTORICAL

    return DEFAULT_RAW_HISTORICAL, DEFAULT_RAW_2026
