"""Data-band CLI: ``python -m data_fetcher {ingest,calendar,continuous,universe,panel,features,check}``.

Handlers only dispatch; logic lives in the stage modules. Later phases fill
the stages that currently exit with a "not built yet" message.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from data_fetcher.ingest import ingest
from futures_common.config import DEFAULT_CONFIG_DIR, load_yaml
from futures_common.stage import StageContext, add_common_args, repo_raw_roots, run_stage

_NOT_BUILT = "stage {stage!r} is not built yet (arrives in a later phase)"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="data_fetcher", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="L0: raw CSV trees → per-contract parquets")
    add_common_args(ing)
    ing.add_argument("--tree", choices=("historical", "2026", "both"), default="both")

    for name, help_text in (
        ("calendar", "L1a: inferred session tables"),
        ("continuous", "L1b: dominant contracts + PIT-adjusted series"),
        ("universe", "L1c: listings, era flags, liquidity"),
        ("panel", "L2a: aligned memmap panel + masks"),
        ("features", "L2b: causal feature memmaps"),
    ):
        sp = sub.add_parser(name, help=help_text)
        add_common_args(sp)

    chk = sub.add_parser("check", help="acceptance assertions for a built stage")
    chk.add_argument("target", choices=("ingest", "calendar", "continuous", "universe", "panel"))
    add_common_args(chk)
    return p


def _cmd_ingest(args: argparse.Namespace) -> int:
    def work(ctx: StageContext) -> Mapping[str, Any]:
        calendar_cfg = load_yaml(DEFAULT_CONFIG_DIR / "calendar.yaml")
        raw_hist, raw_2026 = repo_raw_roots()
        report = ingest(
            ctx.paths,
            raw_historical=raw_hist,
            raw_2026=raw_2026,
            products=list(ctx.products),
            start=ctx.start,
            end=ctx.end,
            tree=args.tree,
            anchors=[str(a) for a in calendar_cfg.get("bootstrap_products", [])],
            anchors_required=int(calendar_cfg.get("anchors_required", 3)),
            force=ctx.force,
        )
        return report.as_dict()

    return run_stage("ingest", args, work, extra_config={"tree": args.tree})


def _cmd_calendar(args: argparse.Namespace) -> int:
    from data_fetcher.calendar import CalendarConfig, build_calendar

    def work(ctx: StageContext) -> Mapping[str, Any]:
        raw = load_yaml(DEFAULT_CONFIG_DIR / "calendar.yaml")
        cfg = CalendarConfig(
            bucket_minutes=int(raw.get("bucket_minutes", 5)),
            smoothing_days=int(raw.get("smoothing_days", 15)),
            min_run_days=int(raw.get("min_run_days", 10)),
            anchors_required=int(raw.get("anchors_required", 3)),
            overrides=tuple(raw.get("overrides") or ()),
        )
        return build_calendar(
            ctx.paths, products=list(ctx.products), cfg=cfg, start=ctx.start, end=ctx.end
        ).as_dict()

    return run_stage("calendar", args, work)


def _cmd_continuous(args: argparse.Namespace) -> int:
    from data_fetcher.continuous import build_continuous

    def work(ctx: StageContext) -> Mapping[str, Any]:
        return build_continuous(
            ctx.paths, products=list(ctx.products), start=ctx.start, end=ctx.end
        ).as_dict()

    return run_stage("continuous", args, work)


def _cmd_universe(args: argparse.Namespace) -> int:
    from data_fetcher.universe import build_universe
    from futures_common.config import load_costs

    def work(ctx: StageContext) -> Mapping[str, Any]:
        return build_universe(
            ctx.paths,
            load_costs(),
            products=list(ctx.products),
            universe_cfg=load_yaml(DEFAULT_CONFIG_DIR / "universe.yaml"),
            volume_cfg=load_yaml(DEFAULT_CONFIG_DIR / "volume_counting.yaml"),
        ).as_dict()

    return run_stage("universe", args, work)


def _cmd_panel(args: argparse.Namespace) -> int:
    from data_fetcher.panel.align import build_panel

    def work(ctx: StageContext) -> Mapping[str, Any]:
        years = list(range(ctx.start.year, ctx.end.year + 1))
        return build_panel(
            ctx.paths,
            products=list(ctx.products),
            years=years,
            limit_cfg=load_yaml(DEFAULT_CONFIG_DIR / "limits.yaml"),
        ).as_dict()

    return run_stage("panel", args, work)


def _cmd_features(args: argparse.Namespace) -> int:
    from data_fetcher.panel.features import build_features

    def work(ctx: StageContext) -> Mapping[str, Any]:
        years = list(range(ctx.start.year, ctx.end.year + 1))
        return build_features(
            ctx.paths,
            products=list(ctx.products),
            years=years,
            volume_cfg=load_yaml(DEFAULT_CONFIG_DIR / "volume_counting.yaml"),
        ).as_dict()

    return run_stage("features", args, work)


def _cmd_check(args: argparse.Namespace) -> int:
    from data_fetcher.checks import run_check

    def work(ctx: StageContext) -> Mapping[str, Any]:
        return run_check(args.target, ctx)

    return run_stage(f"check-{args.target}", args, work)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "calendar":
        return _cmd_calendar(args)
    if args.command == "continuous":
        return _cmd_continuous(args)
    if args.command == "universe":
        return _cmd_universe(args)
    if args.command == "panel":
        return _cmd_panel(args)
    if args.command == "features":
        return _cmd_features(args)
    if args.command == "check":
        return _cmd_check(args)
    raise SystemExit(_NOT_BUILT.format(stage=args.command))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
