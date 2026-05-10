"""CLI: `python -m data_fetcher <subcommand> ...`.

Subcommands:
    pull      — backfill OHLCV
    panel     — print a small panel slice (smoke / debug)
    universe  — print PIT membership for a date
    calendar  — print first/last/N trading dates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from data_fetcher.fetcher import Fetcher


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m data_fetcher",
        description="AutoLLM data_fetcher — AKshare → parquet pipeline.",
    )
    p.add_argument(
        "--root",
        default="~/AutoLLM_data",
        help="Storage root (default: ~/AutoLLM_data).",
    )
    p.add_argument(
        "--adjust",
        default="qfq",
        choices=["", "qfq", "hfq"],
        help="Price adjustment mode (default: qfq).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # pull
    pull = sub.add_parser("pull", help="Backfill OHLCV.")
    g = pull.add_mutually_exclusive_group(required=True)
    g.add_argument("--universe", help="Universe name, e.g. csi300.")
    g.add_argument(
        "--symbols",
        help="Comma-separated explicit symbol list, e.g. SH600519,SZ000001.",
    )
    pull.add_argument("--start", default="2018-01-01")
    pull.add_argument("--end", default="today")
    pull.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between per-symbol pulls (default: 0.1).",
    )

    # panel
    panel = sub.add_parser("panel", help="Print a panel slice (smoke).")
    pg = panel.add_mutually_exclusive_group(required=True)
    pg.add_argument("--universe", help="Universe name (PIT-resolved with --on-date).")
    pg.add_argument("--symbols", help="Comma-separated symbol list.")
    panel.add_argument(
        "--fields",
        default="close,vwap",
        help="Comma-separated fields (default: close,vwap).",
    )
    panel.add_argument("--start", required=True)
    panel.add_argument("--end", required=True)
    panel.add_argument("--on-date", help="PIT snapshot date if --universe is set.")
    panel.add_argument("--limit", type=int, default=20, help="Print first N rows.")

    # universe
    universe = sub.add_parser("universe", help="Print PIT membership.")
    universe.add_argument("name", help="Universe name, e.g. csi300.")
    universe.add_argument("--on-date", required=True)

    # calendar
    cal = sub.add_parser("calendar", help="Print trading calendar summary.")
    cal.add_argument("--head", type=int, default=5)
    cal.add_argument("--tail", type=int, default=5)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    f = Fetcher(root=Path(args.root).expanduser(), adjust=args.adjust)

    if args.cmd == "pull":
        f.sleep_s = args.sleep
        if args.universe:
            result = f.pull(universe=args.universe, start=args.start, end=args.end)
        else:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
            result = f.pull(symbols=symbols, start=args.start, end=args.end)
        print(
            f"pull: {result['pulled']} rows pulled, "
            f"{result['skipped']} skipped (already on disk), "
            f"{result['failed']} failed"
        )
        return 0 if result["failed"] == 0 else 2

    if args.cmd == "panel":
        fields = [s.strip() for s in args.fields.split(",") if s.strip()]
        if args.universe:
            lf = f.panel(
                universe=args.universe,
                fields=fields,
                start=args.start,
                end=args.end,
                on_date=args.on_date,
            )
        else:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
            lf = f.panel(
                universe=symbols,
                fields=fields,
                start=args.start,
                end=args.end,
            )
        df = lf.collect()
        with pl.Config(tbl_rows=args.limit, tbl_cols=20):
            print(df.head(args.limit))
        print(f"...{df.height} rows total")
        return 0

    if args.cmd == "universe":
        members = f.universe(args.name, args.on_date)
        for m in sorted(members):
            print(m)
        print(f"# {len(members)} members as of {args.on_date}", file=sys.stderr)
        return 0

    if args.cmd == "calendar":
        cal = f.calendar()
        print(f"first: {cal[: args.head].to_list()}")
        print(f"last : {cal[-args.tail :].to_list()}")
        print(f"# {len(cal)} trading days total")
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
