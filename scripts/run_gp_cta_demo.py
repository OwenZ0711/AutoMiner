#!/usr/bin/env python
"""Run the GP-CTA baseline on A-share OHLCV data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_fetcher import AKshareParquetAdapter, AKshareSource, Fetcher
from gp_cta.evaluator import EvaluationConfig, FormulaEvaluator, split_by_date
from gp_cta.features import RAW_COLUMNS, build_features
from gp_cta.gp import GPConfig, GPEngine


DEFAULT_FIXTURE = Path("data_fetcher/tests/fixtures/ohlcv_5sym_2018_2024.parquet")
DEFAULT_FIELDS = list(RAW_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-mode",
        choices=("fixture", "cache"),
        default="fixture",
        help="fixture reads the bundled parquet; cache reads ~/AutoLLM_data via DataAdapter.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--limit-symbols", type=int, default=30)
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.5,
        help="Seconds to sleep between AKShare symbol pulls in cache mode.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="AKShare retry attempts per symbol in cache mode.",
    )
    parser.add_argument(
        "--base-backoff-s",
        type=float,
        default=1.0,
        help="Initial AKShare retry backoff seconds in cache mode.",
    )
    parser.add_argument(
        "--pull-akshare",
        action="store_true",
        help="Pull selected symbols into ~/AutoLLM_data before reading cache mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/gp_cta"))
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol filter. In cache mode this overrides --universe.",
    )
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.population_size < 1:
        raise SystemExit("--population-size must be >= 1")
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.max_depth < 1:
        raise SystemExit("--max-depth must be >= 1")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.limit_symbols < 1:
        raise SystemExit("--limit-symbols must be >= 1")

    panel = load_panel(args)

    feature_panel = build_features(panel)
    splits = split_by_date(feature_panel, train_end=args.train_end, val_end=args.val_end)
    validate_splits(splits)
    evaluator = FormulaEvaluator(EvaluationConfig())
    engine = GPEngine(
        evaluator=evaluator,
        config=GPConfig(
            population_size=args.population_size,
            generations=args.generations,
            max_depth=args.max_depth,
            seed=args.seed,
            elite_size=min(5, args.population_size),
        ),
    )
    ranked = engine.evolve(splits["train"])
    seen: set[str] = set()
    top = []
    for candidate in ranked:
        formula = candidate.result.formula
        if formula in seen:
            continue
        seen.add(formula)
        top.append(candidate)
        if len(top) >= args.top_k:
            break
    if not top:
        raise SystemExit("GP search produced no candidate formulas")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = []
    metrics_rows = []
    for rank, candidate in enumerate(top, start=1):
        train = candidate.result.as_dict()
        validation = evaluator.evaluate(candidate.expr, splits["validation"]).as_dict()
        test = evaluator.evaluate(candidate.expr, splits["test"]).as_dict()
        top_rows.append({"rank": rank, **train})
        for split_name, row in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            metrics_rows.append({"rank": rank, "split": split_name, **row})

    pl.DataFrame(top_rows).write_csv(args.output_dir / "top_formulas.csv")
    pl.DataFrame(metrics_rows).write_csv(args.output_dir / "metrics.csv")
    best_equity = evaluator.portfolio_equity_curve(top[0].expr, feature_panel)
    pl.from_pandas(best_equity).write_csv(args.output_dir / "equity_curve.csv")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )

    print(f"Best formula: {top[0].result.formula}")
    print(f"Train fitness: {top[0].result.fitness:.4f}")
    print(f"Wrote results to {args.output_dir}")
    return 0


def load_panel(args: argparse.Namespace) -> pl.DataFrame:
    if args.data_mode == "fixture":
        panel = pl.read_parquet(args.input)
        if args.symbols:
            symbols = [
                symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()
            ]
            panel = panel.filter(pl.col("symbol").is_in(symbols))
        if panel.is_empty():
            raise SystemExit("input panel is empty after applying filters")
        return panel

    fetcher = Fetcher(
        source=AKshareSource(
            max_retries=args.max_retries,
            base_backoff_s=args.base_backoff_s,
        ),
        sleep_s=args.sleep_s,
    )
    adapter = AKshareParquetAdapter(fetcher)
    if args.symbols:
        symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    else:
        symbols = fetcher.universe(args.universe, on_date=args.end)[: args.limit_symbols]

    if args.pull_akshare:
        result = fetcher.pull(symbols=symbols, start=args.start, end=args.end)
        if result["failed"]:
            print(f"Warning: AKShare pull had {result['failed']} failed symbols")
        if result["pulled"] == 0 and result["skipped"] == 0 and result["failed"]:
            raise SystemExit(
                "AKShare did not return usable OHLCV for any requested symbol. "
                "Try a smaller LIMIT_SYMBOLS value, larger SLEEP_S/MAX_RETRIES, "
                "or rerun later when the upstream endpoint is stable."
            )

    panel = adapter.panel(
        universe=symbols,
        fields=DEFAULT_FIELDS,
        start=args.start,
        end=args.end,
    ).collect()
    if panel.is_empty():
        raise SystemExit(
            "cache panel is empty; rerun with --pull-akshare or choose symbols already "
            "cached under ~/AutoLLM_data"
        )
    loaded_symbols = panel.select("symbol").unique().height
    min_date = panel.select(pl.col("date").min()).item()
    max_date = panel.select(pl.col("date").max()).item()
    print(
        f"Loaded {panel.height} rows for {loaded_symbols} symbols "
        f"from cache ({min_date}..{max_date})"
    )
    return panel


def validate_splits(splits: dict[str, pl.DataFrame]) -> None:
    empty = [name for name, frame in splits.items() if frame.is_empty()]
    if empty:
        sizes = {name: frame.height for name, frame in splits.items()}
        raise SystemExit(
            "empty data split(s): "
            f"{', '.join(empty)}. Split row counts: {sizes}. "
            "Adjust --start/--end/--train-end/--val-end."
        )


if __name__ == "__main__":
    raise SystemExit(main())
