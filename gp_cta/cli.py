"""Mining-band CLI: ``python -m gp_cta {scan,validate,mine,check}``."""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Mapping
from typing import Any

from futures_common.config import DEFAULT_CONFIG_DIR, load_yaml
from futures_common.stage import StageContext, add_common_args, run_stage

_NOT_BUILT = "stage {stage!r} is not built yet (arrives in a later phase)"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gp_cta", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("scan", "L3: lagged-correlation scan → saved tensors"),
        ("validate", "L3/L4: nulls + FDR + stability → edge table append"),
        ("mine", "GP: warm-start mining round → candidate pool"),
    ):
        sp = sub.add_parser(name, help=help_text)
        add_common_args(sp)
        if name == "validate":
            sp.add_argument("--draws", type=int, default=None, help="override null draws")
        if name == "mine":
            sp.add_argument(
                "--init",
                choices=("warmstart", "random"),
                default="warmstart",
                help="population init (random = baseline for the A/B check)",
            )
            sp.add_argument("--round-id", default=None)
    chk = sub.add_parser("check", help="acceptance assertions")
    chk.add_argument("target", choices=("leadlag", "gp"))
    add_common_args(chk)
    return p


def _years(ctx: StageContext) -> list[int]:
    return list(range(ctx.start.year, ctx.end.year + 1))


def _cmd_scan(args: argparse.Namespace) -> int:
    import numpy as np

    from gp_cta.leadlag.scan import ScanConfig, save_scan, scan_lagged_corr
    from gp_cta.panel_io import PanelReader

    def work(ctx: StageContext) -> Mapping[str, Any]:
        raw = load_yaml(DEFAULT_CONFIG_DIR / "scan.yaml")
        cfg = ScanConfig(
            max_lag=int(raw.get("max_lag_bars", 60)),
            min_joint_bars=int(raw.get("min_joint_bars", 20_000)),
        )
        panel = PanelReader(ctx.paths, _years(ctx))
        t0 = dt.datetime.now()
        result = scan_lagged_corr(panel, cfg, window=(ctx.start, ctx.end))
        wall_s = (dt.datetime.now() - t0).total_seconds()
        out = ctx.paths.leadlag_scan_dir(cfg.hash) / "full.npz"
        save_scan(result, out)
        return {
            "products": list(panel.products),
            "wall_s": round(wall_s, 2),
            "max_abs_rho": float(np.nanmax(np.abs(result.rho))),
            "pairs_with_data": int((result.n.max(axis=0) > 0).sum()),
            "out": str(out),
        }

    return run_stage("scan", args, work)


def _cmd_validate(args: argparse.Namespace) -> int:
    import polars as pl

    from gp_cta.leadlag.edges import EdgeStore
    from gp_cta.leadlag.scan import ScanConfig, scan_lagged_corr
    from gp_cta.leadlag.validate import (
        PriorSet,
        apply_fdr,
        pair_statistics,
        permutation_null,
        stability_metrics,
    )
    from gp_cta.ledger import TrialLedger
    from gp_cta.panel_io import PanelReader

    def _window_stats(
        panel: Any,
        cfg: Any,
        window: tuple[dt.date, dt.date],
        lag_grid: tuple[int, ...],
        n_draws: int,
        min_shift: int,
        priors: Any,
        q_a: float,
        q_b: float,
        seed: int,
        stream: str,
    ) -> Any:
        scan = scan_lagged_corr(panel, cfg, window=window, lags=lag_grid)
        stats = pair_statistics(scan, lag_grid=lag_grid)
        stats = permutation_null(
            panel,
            stats,
            cfg,
            lag_grid=lag_grid,
            n_draws=n_draws,
            min_shift_days=min_shift,
            seed=seed,
            window=window,
            stream=stream,
        )
        return apply_fdr(stats, priors, q_a=q_a, q_b=q_b)

    def work(ctx: StageContext) -> Mapping[str, Any]:
        raw = load_yaml(DEFAULT_CONFIG_DIR / "scan.yaml")
        cfg = ScanConfig(
            max_lag=int(raw.get("max_lag_bars", 60)),
            min_joint_bars=int(raw.get("min_joint_bars", 20_000)),
        )
        lag_grid = tuple(int(x) for x in raw.get("lag_grid", [])) or DEFAULT_GRID
        null_cfg = raw.get("null", {})
        n_draws = args.draws if args.draws else int(null_cfg.get("n_draws", 200))
        monthly_draws = int(null_cfg.get("monthly_draws", 50))
        min_shift = int(null_cfg.get("min_shift_days", 5))
        rolling_cfg = raw.get("rolling", {})
        q_a = float(raw.get("fdr", {}).get("q_family_a", 0.05))
        q_b = float(raw.get("fdr", {}).get("q_family_b", 0.10))
        priors = PriorSet.from_config(
            load_yaml(DEFAULT_CONFIG_DIR / "leadlag_priors.yaml").get("pairs", [])
        )

        panel = PanelReader(ctx.paths, _years(ctx))
        ledger = TrialLedger(ctx.paths, config_hash=cfg.hash)
        store = EdgeStore(ctx.paths)

        # ---- monthly rolling windows: PIT edge history, each with its OWN null
        from gp_cta.leadlag.scan import monthly_windows

        monthly_frames: list[pl.DataFrame] = []
        monthly_appended = 0
        for w_start, w_end in monthly_windows(
            panel,
            window_months=int(rolling_cfg.get("window_months", 24)),
            step_months=int(rolling_cfg.get("step_months", 1)),
        ):
            w_start = max(w_start, ctx.start)
            w_end = min(w_end, ctx.end)
            if w_start >= w_end:
                continue
            m_stats = _window_stats(
                panel,
                cfg,
                (w_start, w_end),
                lag_grid,
                monthly_draws,
                min_shift,
                priors,
                q_a,
                q_b,
                ctx.seed,
                f"leadlag.null.m{w_end}",
            )
            if m_stats.is_empty() or m_stats["n_eff"].max() == 0:
                continue
            for r in m_stats.iter_rows(named=True):
                ledger.log(
                    "ll_pair_scan",
                    f"{r['leader']}→{r['follower']} w={w_start}..{w_end}",
                    asof_date=w_end,
                    stat=r["a_stat"],
                )
            # trailing PIT stability: only windows ALREADY processed (≤ w_end)
            prior = (
                pl.concat(monthly_frames, how="vertical")
                if monthly_frames
                else pl.DataFrame(
                    schema={
                        "leader": pl.Utf8,
                        "follower": pl.Utf8,
                        "a_stat": pl.Float64,
                        "q_value": pl.Float64,
                        "window_end": pl.Date,
                    }
                )
            )
            m_stats = stability_metrics(
                prior,
                m_stats,
                lookback=int(rolling_cfg.get("stability_lookback", 12)),
                q_cut=q_b,
            )
            monthly_frames.append(
                m_stats.select("leader", "follower", "a_stat", "q_value").with_columns(
                    pl.lit(w_end).alias("window_end")
                )
            )
            m_edges = m_stats.with_columns(
                pl.lit("monthly").alias("regime_flags"),
                pl.lit(w_start).alias("window_start"),
                pl.lit(w_end).alias("window_end"),
            )
            try:
                store.append(m_edges, asof=w_end, config_hash=cfg.hash)
                monthly_appended += 1
            except FileExistsError:
                pass  # idempotent re-run: monthly file already written

        # ---- full window: the headline test with the deep null
        stats = _window_stats(
            panel,
            cfg,
            (ctx.start, ctx.end),
            lag_grid,
            n_draws,
            min_shift,
            priors,
            q_a,
            q_b,
            ctx.seed,
            "leadlag.null.full",
        )
        for r in stats.iter_rows(named=True):
            ledger.log(
                "ll_pair_scan",
                f"{r['leader']}→{r['follower']} w={ctx.start}..{ctx.end}",
                asof_date=ctx.end,
                stat=r["a_stat"],
            )

        monthly = (
            pl.concat(monthly_frames, how="vertical")
            if monthly_frames
            else pl.DataFrame(
                schema={
                    "leader": pl.Utf8,
                    "follower": pl.Utf8,
                    "a_stat": pl.Float64,
                    "q_value": pl.Float64,
                    "window_end": pl.Date,
                }
            )
        )
        stats = stability_metrics(
            monthly,
            stats,
            lookback=int(rolling_cfg.get("stability_lookback", 12)),
            q_cut=q_b,
        )

        edges = stats.with_columns(
            pl.lit("").alias("regime_flags"),
            pl.lit(ctx.start).alias("window_start"),
            pl.lit(ctx.end).alias("window_end"),
        )
        try:
            store.append(edges, asof=ctx.end, config_hash=cfg.hash)
        except FileExistsError:
            pass
        ledger.flush()

        validated = stats.filter(pl.col("validated"))
        return {
            "pairs_scanned": stats.height,
            "monthly_windows": monthly_appended,
            "validated": validated.height,
            "validated_pairs": [
                f"{r['leader']}→{r['follower']} lag={r['lag_bars']} q={r['q_value']:.3f} "
                f"stab={r['stability']:.2f}"
                for r in validated.iter_rows(named=True)
            ],
        }

    return run_stage("validate", args, work)


DEFAULT_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60)


def _cmd_check(args: argparse.Namespace) -> int:
    from gp_cta.checks import run_check

    def work(ctx: StageContext) -> Mapping[str, Any]:
        return run_check(args.target, ctx)

    return run_stage(f"check-{args.target}", args, work)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "scan":
        return _cmd_scan(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "mine":
        try:
            from gp_cta.cli_mine import cmd_mine
        except ImportError as exc:  # Phase 5 delivers the GP engine
            raise SystemExit(_NOT_BUILT.format(stage="mine")) from exc
        result: int = cmd_mine(args)
        return result
    raise SystemExit(_NOT_BUILT.format(stage=args.command))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
