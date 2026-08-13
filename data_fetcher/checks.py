"""Acceptance checks (`python -m data_fetcher check <stage>`).

Same functions back the pytest `acceptance` marker — one source of truth for
the proposal's build-order "done when" column. Each check returns a report
mapping and raises AssertionError on failure (run_stage maps that to exit 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl
from loguru import logger

from data_fetcher.readers.base import PSEUDO_RE
from futures_common.manifest import read_latest
from futures_common.stage import StageContext

_GB = 1024**3


def run_check(target: str, ctx: StageContext) -> Mapping[str, Any]:
    if target == "ingest":
        return check_ingest(ctx)
    if target == "calendar":
        return check_calendar(ctx)
    if target == "continuous":
        return check_continuous(ctx)
    if target == "universe":
        return check_universe(ctx)
    if target == "panel":
        return check_panel(ctx)
    raise SystemExit(f"check {target!r} is not built yet (arrives in a later phase)")


def check_ingest(ctx: StageContext) -> Mapping[str, Any]:
    raw_root = ctx.paths.raw_root
    assert raw_root.is_dir(), f"raw store missing: {raw_root}"
    assert ctx.paths.raw_manifest.exists(), "raw/_manifest.parquet missing"

    files = sorted(raw_root.glob("exchange=*/product=*/*.parquet"))
    assert files, f"no contract parquets under {raw_root}"

    manifest = pl.read_parquet(ctx.paths.raw_manifest)
    manifest_contracts = set(manifest["contract"].to_list())

    n_rows_total = 0
    for f in files:
        contract = f.stem
        assert not PSEUDO_RE.search(contract), f"pseudo-contract ingested: {f}"
        assert contract in manifest_contracts, f"{contract} missing from manifest"
        product_dir = f.parent.name.removeprefix("product=")
        df = pl.read_parquet(f)
        assert df.height > 0, f"{f} is empty"
        assert (df["product"] == product_dir).all(), f"{f}: product column != path"
        assert (df["contract"] == contract).all(), f"{f}: contract column != filename"
        assert df["bob"].is_sorted(), f"{f}: not sorted by bob"
        assert df["bob"].n_unique() == df.height, f"{f}: duplicate bob rows"
        assert (df["volume"] > 0).all(), f"{f}: zero-volume rows survived"
        # full-lifetime semantics: rows may extend past the slice, but every
        # row must carry a mapped trading_date
        assert df["trading_date"].null_count() == 0, f"{f}: unmapped trading_date rows"
        n_rows_total += df.height

    latest = read_latest("ingest", ctx.paths)
    assert latest is not None, "no ingest run manifest found"
    peak_gb = latest.peak_rss_bytes / _GB
    budget = ctx.defaults.memory.budget_for("ingest")
    assert peak_gb < budget, f"ingest peak RSS {peak_gb:.2f} GB >= budget {budget} GB"
    assert latest.exit_status == "ok", f"latest ingest run status: {latest.exit_status}"

    logger.info(
        "check ingest OK: {} contracts, {} rows, last peak {:.2f} GB",
        len(files),
        n_rows_total,
        peak_gb,
    )
    return {
        "contracts": len(files),
        "rows": n_rows_total,
        "last_peak_rss_gb": round(peak_gb, 3),
    }


def check_calendar(ctx: StageContext) -> Mapping[str, Any]:
    """Trading days sane (no weekends, Golden Weeks excluded); COVID night gap
    visible when the slice covers 2020H1; RB 2016 night change when covered."""
    import datetime as dt

    from data_fetcher.calendar import load_session_table

    td = pl.read_parquet(ctx.paths.trading_days)["trading_date"].to_list()
    assert td, "trading_days.parquet is empty"
    assert all(d.weekday() < 5 for d in td), "weekend day in trading_days"
    for year in range(ctx.start.year, ctx.end.year + 1):
        gw = {dt.date(year, 10, d) for d in range(1, 8)}
        if dt.date(year, 10, 1) >= ctx.start and dt.date(year, 10, 7) <= ctx.end:
            assert not (set(td) & gw), f"Golden Week {year} present in trading_days"

    table = load_session_table(ctx.paths)
    report: dict[str, Any] = {"n_trading_days": len(td)}

    if "RB" in ctx.products:
        if ctx.start <= dt.date(2020, 3, 15) <= ctx.end:
            covid = table.intervals_for("RB", dt.date(2020, 3, 15))
            assert covid and covid[0][0] == dt.time(9, 0), (
                f"COVID gap: RB 2020-03-15 should be day-only, got {covid}"
            )
            report["covid_gap"] = "detected"
        if ctx.start <= dt.date(2016, 2, 1) and ctx.end >= dt.date(2016, 8, 1):
            before = table.intervals_for("RB", dt.date(2016, 2, 1))
            after = table.intervals_for("RB", dt.date(2016, 8, 1))
            assert before[0] == (dt.time(21, 0), dt.time(1, 0)), before
            assert after[0] == (dt.time(21, 0), dt.time(23, 0)), after
            report["rb_2016_night_change"] = "detected"
    logger.info("check calendar OK: {}", report)
    return report


def check_continuous(ctx: StageContext) -> Mapping[str, Any]:
    """RB dominant follows 01/05/10; no roll return spikes in adjusted series."""
    import numpy as np

    rc = pl.read_parquet(ctx.paths.roll_calendar).filter(
        pl.col("trading_date").is_between(ctx.start, ctx.end)
    )
    report: dict[str, Any] = {"rolls": rc.filter(pl.col("reason") == "volume").height}

    if "RB" in ctx.products:
        rb = rc.filter((pl.col("product") == "RB") & (pl.col("reason") == "volume"))
        months = {c[-2:] for c in rb["new_contract"].to_list()}
        assert months <= {"01", "05", "10"}, f"RB dominant months {months} != 01/05/10"
        report["rb_months"] = sorted(months)

    worst = 0.0
    for product in ctx.products:
        year_files = sorted(ctx.paths.continuous_dir.glob(f"product={product}/year=*.parquet"))
        assert year_files, f"no continuous years for {product}"
        for f in year_files:
            df = pl.read_parquet(f).sort("bob")
            adj = (df["close"].cast(pl.Float64) * df["adj_factor"]).to_numpy()
            rets = np.abs(np.diff(np.log(adj)))
            seg = df["segment_id"].to_numpy()
            same_seg = seg[1:] == seg[:-1]
            roll_next = df["roll_flag"].to_numpy()[1:]
            if not roll_next.any():
                continue
            sigma = float(np.std(rets[same_seg & ~roll_next])) or 1e-9
            worst_roll = float(np.max(rets[same_seg & roll_next]))
            worst = max(worst, worst_roll / sigma)
            assert worst_roll < max(5 * sigma, 0.02), (
                f"{f}: roll-bar |return| {worst_roll:.4f} vs sigma {sigma:.5f}"
            )
    report["worst_roll_sigma"] = round(worst, 2)
    logger.info("check continuous OK: {}", report)
    return report


def check_panel(ctx: StageContext) -> Mapping[str, Any]:
    """Masks agree with the session table; features present; budgets respected."""
    import datetime as dt

    import numpy as np

    from data_fetcher.calendar import load_session_table
    from data_fetcher.panel import memmap_io
    from data_fetcher.panel.features import FEATURES

    table = load_session_table(ctx.paths)
    years = list(range(ctx.start.year, ctx.end.year + 1))
    report: dict[str, Any] = {"years": years}
    epoch = dt.date(1970, 1, 1)

    for year in years:
        meta = memmap_io.read_meta(ctx.paths, year)
        grid = pl.read_parquet(ctx.paths.bob_index(year))
        assert grid.height == meta["T"], f"{year}: bob_index rows != meta.T"
        m_traded = memmap_io.open_array(ctx.paths, year, "mask", "m_traded")
        assert m_traded.shape == (meta["T"], meta["N"])

        # no traded bar outside the product's session template (sample 30 days)
        td = grid["trading_date"].cast(pl.Int32).to_numpy()
        rel = grid["rel_min"].to_numpy()
        sample_days = np.unique(td)[:: max(1, np.unique(td).size // 30)]
        from data_fetcher.panel.align import intervals_to_rel

        for day_epoch in sample_days:
            day = epoch + dt.timedelta(days=int(day_epoch))
            rows = np.flatnonzero(td == day_epoch)
            for i, product in enumerate(meta["products"]):
                ivs = intervals_to_rel(table.intervals_for(product, day))
                if not ivs:
                    continue
                traded_rels = rel[rows][m_traded[rows, i] > 0]
                in_session = np.zeros(traded_rels.size, dtype=bool)
                for r0, r1 in ivs:
                    in_session |= (traded_rels >= r0) & (traded_rels < r1)
                assert in_session.all(), (
                    f"{year} {product} {day}: traded bars outside session template"
                )
        for name in FEATURES:
            assert ctx.paths.panel_feature(year, name).exists(), f"{year}: missing {name}"
        report[f"T_{year}"] = meta["T"]

    latest = read_latest("panel", ctx.paths)
    if latest is not None:
        peak_gb = latest.peak_rss_bytes / _GB
        budget = ctx.defaults.memory.budget_for("panel")
        assert peak_gb < budget, f"panel peak {peak_gb:.2f} GB >= {budget} GB"
        report["panel_peak_gb"] = round(peak_gb, 2)
    logger.info("check panel OK: {}", report)
    return report


def check_universe(ctx: StageContext) -> Mapping[str, Any]:
    """Inferred multipliers agree with declared costs.yaml (verify-costs)."""
    from futures_common.config import load_costs

    products = pl.read_parquet(ctx.paths.universe_products)
    costs = load_costs()
    mismatches: list[str] = []
    for row in products.iter_rows(named=True):
        declared = costs.for_product(row["product"])
        if not declared.declared or declared.multiplier is None:
            continue
        if row["multiplier_confidence"] <= 0:
            continue
        rel = abs(row["multiplier"] - declared.multiplier) / declared.multiplier
        if rel > costs.inference.multiplier_rel_tol:
            mismatches.append(f"{row['product']}: {row['multiplier']} vs {declared.multiplier}")
    assert not mismatches, f"multiplier mismatches: {mismatches}"
    logger.info("check universe OK: {} products verified", products.height)
    return {"products": products.height}
