"""Umbrella orchestrator: run pipeline stages as one subprocess each.

Subprocess-per-stage is a memory-safety decision, not an ergonomic one:
CPython rarely returns freed pages to the OS, so a single in-process pipeline
would accumulate allocator arenas across stages and make the "< 4 GB per
stage" budget unmeasurable. Each stage starts at a small baseline, its peak
RSS is its own, and :func:`futures_common.memory.watch_child` can kill a
runaway child at ``budget * kill_factor`` without taking the orchestrator down.

    python -m futures_common run --slice acceptance [--stages ingest,panel]
                                 [--from panel] [--check] [--seed 42]
    python -m futures_common status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from loguru import logger

from futures_common.config import load_defaults
from futures_common.log import setup_logging
from futures_common.memory import watch_child
from futures_common.paths import default_paths

# Canonical stage order: (package, subcommand)
STAGES: tuple[tuple[str, str], ...] = (
    ("data_fetcher", "ingest"),
    ("data_fetcher", "calendar"),
    ("data_fetcher", "continuous"),
    ("data_fetcher", "universe"),
    ("data_fetcher", "panel"),
    ("data_fetcher", "features"),
    ("gp_cta", "scan"),
    ("gp_cta", "validate"),
    ("gp_cta", "mine"),
)
STAGE_NAMES = tuple(sub for _, sub in STAGES)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="futures_common", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run pipeline stages sequentially, one subprocess each")
    run.add_argument("--slice", default="acceptance", help="slice name from configs/defaults.toml")
    run.add_argument("--stages", default=None, help="comma list; default: all in canonical order")
    run.add_argument("--from", dest="from_stage", default=None, help="resume from this stage")
    run.add_argument("--check", action="store_true", help="run each stage's `check` afterwards")
    run.add_argument("--seed", type=int, default=None, help="override configs seed")
    run.add_argument("--data-root", default=None)

    sub.add_parser("status", help="tabulate latest run manifests")
    return p


def _select_stages(stages_arg: str | None, from_stage: str | None) -> list[tuple[str, str]]:
    selected = list(STAGES)
    if stages_arg:
        wanted = [s.strip() for s in stages_arg.split(",") if s.strip()]
        unknown = set(wanted) - set(STAGE_NAMES)
        if unknown:
            raise SystemExit(f"unknown stage(s) {sorted(unknown)}; valid: {STAGE_NAMES}")
        selected = [(pkg, sub) for pkg, sub in STAGES if sub in wanted]
    if from_stage:
        if from_stage not in STAGE_NAMES:
            raise SystemExit(f"unknown --from stage {from_stage!r}; valid: {STAGE_NAMES}")
        idx = next(i for i, (_, sub) in enumerate(selected) if sub == from_stage)
        selected = selected[idx:]
    return selected


def _run_stage(
    pkg: str, sub: str, extra_argv: list[str], budget_gb: float, kill_factor: float
) -> int:
    argv = [sys.executable, "-m", pkg, sub, *extra_argv]
    logger.info("→ {} (budget {:.1f} GB)", " ".join(argv[2:]), budget_gb)
    child = subprocess.Popen(argv)
    watch = watch_child(child.pid, budget_gb, kill_factor)
    rc = child.wait()
    watch.stop()
    if watch.killed_for_memory:
        logger.error("stage {} killed for exceeding memory budget", sub)
        return 3
    return rc


def cmd_run(args: argparse.Namespace) -> int:
    defaults = load_defaults()
    if args.slice not in defaults.slices:
        raise SystemExit(f"unknown slice {args.slice!r}; defined: {sorted(defaults.slices)}")
    seed = args.seed if args.seed is not None else defaults.project.seed

    common = ["--slice", args.slice, "--seed", str(seed)]
    if args.data_root:
        common += ["--data-root", args.data_root]

    for pkg, sub in _select_stages(args.stages, args.from_stage):
        budget = defaults.memory.budget_for(sub)
        rc = _run_stage(pkg, sub, common, budget, defaults.memory.kill_factor)
        if rc != 0:
            logger.error("stage {} exited with {} — stopping the chain", sub, rc)
            return rc
        if args.check:
            rc = _run_stage(pkg, "check", [sub, *common], budget, defaults.memory.kill_factor)
            if rc != 0:
                logger.error("check {} failed with {} — stopping the chain", sub, rc)
                return rc
    logger.info("pipeline complete")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    paths = default_paths()
    latest = paths.latest_runs
    if not latest.exists():
        print("no runs recorded yet")
        return 0
    pointer: dict[str, str] = json.loads(latest.read_text())
    header = f"{'stage':<12} {'run_id':<28} {'peak_rss':>9} {'status':<12}"
    print(header)
    print("-" * len(header))
    for stage, run_id in sorted(pointer.items()):
        manifest = paths.run_dir(stage, run_id) / "manifest.json"
        peak, status = "?", "?"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            peak = f"{m.get('peak_rss_bytes', 0) / 1024**3:.2f}G"
            status = str(m.get("exit_status", "?"))
        print(f"{stage:<12} {run_id:<28} {peak:>9} {status:<12}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging("umbrella")
    if args.command == "run":
        return cmd_run(args)
    return cmd_status(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
