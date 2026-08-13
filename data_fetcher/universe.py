"""L1c: listings, era flags, multiplier/tick inference, liquidity table.

Everything here is *descriptive metadata* (sanctioned non-causal, look-ahead
safety §6): contract specs, exchange conventions, listing dates. The one
market-derived table — the ADV liquidity proxy — is computed as a trailing
statistic so PIT consumers can use it as-of.

Multiplier inference (trap-6-aware cross-check of configs/costs.yaml):
``multiplier ≈ amount / (volume * close)``. The ratio is invariant to the
2020 double→single volume-counting switch (amount and volume are counted the
same way), so it doubles as a fee-table verification.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import polars as pl
from loguru import logger

from data_fetcher.continuous import build_daily_stats
from futures_common.config import CostsConfig
from futures_common.paths import FuturesPaths

STANDARD_MULTIPLIERS = (
    0.1,
    0.5,
    1,
    2,
    5,
    10,
    15,
    20,
    30,
    50,
    60,
    100,
    200,
    300,
    500,
    1000,
    10000,
)


@dataclass(slots=True)
class UniverseReport:
    products: list[str]
    multiplier_mismatches: list[str]
    breaks_confirmed: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "products": self.products,
            "multiplier_mismatches": self.multiplier_mismatches,
            "breaks_confirmed": self.breaks_confirmed,
        }


def _liquid_bars(paths: FuturesPaths, product: str, n: int = 1000) -> pl.DataFrame:
    files = sorted(paths.raw_root.glob(f"exchange=*/product={product}/*.parquet"))
    if not files:
        return pl.DataFrame()
    return (
        pl.scan_parquet([str(f) for f in files])
        .select("exchange", "contract", "bob", "close", "amount", "volume")
        .sort("volume", descending=True)
        .head(n)
        .collect(engine="streaming")
    )


def infer_multiplier(bars: pl.DataFrame) -> tuple[float, float]:
    """(multiplier, confidence). Snapped to the standard set within 5%."""
    med = (bars["amount"] / (bars["volume"] * bars["close"].cast(pl.Float64))).median()
    if med is None:
        return float("nan"), 0.0
    ratio = float(cast("float", med))
    if ratio <= 0:
        return float("nan"), 0.0
    snapped = min(STANDARD_MULTIPLIERS, key=lambda m: abs(m - ratio) / m)
    rel_gap = abs(snapped - ratio) / snapped
    if rel_gap <= 0.05:
        return float(snapped), float(1.0 - rel_gap)
    return ratio, 0.5


def infer_tick(paths: FuturesPaths, product: str) -> float:
    """Mode of positive close-diffs on the product's highest-volume contract."""
    files = sorted(paths.raw_root.glob(f"exchange=*/product={product}/*.parquet"))
    if not files:
        return float("nan")
    best_file, best_vol = None, -1
    for f in files:
        vol = pl.scan_parquet(f).select(pl.col("volume").sum()).collect().item()
        if vol > best_vol:
            best_file, best_vol = f, vol
    diffs = (
        pl.scan_parquet(str(best_file))
        .sort("bob")
        .select(pl.col("close").cast(pl.Float64).diff().round(6).alias("d"))
        .filter(pl.col("d") > 0)
        .collect(engine="streaming")
    )
    if diffs.is_empty():
        return float("nan")
    mode = diffs["d"].value_counts().sort(["count", "d"], descending=[True, False])["d"][0]
    return float(mode)


def volume_break_ratio(
    daily: pl.DataFrame, break_date: dt.date, window_days: int = 60
) -> float | None:
    """median(day_volume / last_position) after vs before the break (pooled)."""
    lo = break_date - dt.timedelta(days=window_days * 2)
    hi = break_date + dt.timedelta(days=window_days * 2)
    d = daily.filter(
        pl.col("trading_date").is_between(lo, hi) & (pl.col("last_position") > 0)
    ).with_columns((pl.col("day_volume") / pl.col("last_position")).alias("v_oi"))
    before = d.filter(pl.col("trading_date") < break_date)["v_oi"].median()
    after = d.filter(pl.col("trading_date") >= break_date)["v_oi"].median()
    if before is None or after is None:
        return None
    before_f, after_f = float(cast("float", before)), float(cast("float", after))
    if before_f == 0:
        return None
    return after_f / before_f


def build_universe(
    paths: FuturesPaths,
    costs: CostsConfig,
    *,
    products: Sequence[str] | None = None,
    universe_cfg: Mapping[str, Any] | None = None,
    volume_cfg: Mapping[str, Any] | None = None,
) -> UniverseReport:
    universe_cfg = universe_cfg or {}
    volume_cfg = volume_cfg or {}
    delisted = {str(p).upper() for p in universe_cfg.get("delisted", [])}

    if products is None:
        products = sorted(
            p.name.removeprefix("product=") for p in paths.raw_root.glob("exchange=*/product=*")
        )
    products = [p.upper() for p in products]

    product_rows: list[dict[str, Any]] = []
    liquidity_frames: list[pl.DataFrame] = []
    mismatches: list[str] = []
    daily_by_exchange: dict[str, list[pl.DataFrame]] = {}

    for product in products:
        daily = (
            pl.read_parquet(paths.daily_stats(product))
            if paths.daily_stats(product).exists()
            else build_daily_stats(paths, product)
        )
        if daily.is_empty():
            logger.warning("universe: no data for {} — skipped", product)
            continue
        bars = _liquid_bars(paths, product)
        exchange = bars["exchange"][0] if not bars.is_empty() else "UNKNOWN"
        multiplier, confidence = infer_multiplier(bars)
        tick = infer_tick(paths, product)

        declared = costs.for_product(product)
        if declared.declared and declared.multiplier is not None and confidence > 0:
            rel = abs(multiplier - declared.multiplier) / declared.multiplier
            if rel > costs.inference.multiplier_rel_tol:
                mismatches.append(
                    f"{product}: inferred multiplier {multiplier:g} vs declared "
                    f"{declared.multiplier:g}"
                )

        product_rows.append(
            {
                "product": product,
                "exchange": exchange,
                "first_trading_date": daily["trading_date"].min(),
                "last_trading_date": daily["trading_date"].max(),
                "is_delisted": product in delisted,
                "multiplier": multiplier,
                "multiplier_confidence": confidence,
                "tick_size": tick,
            }
        )
        daily_by_exchange.setdefault(exchange, []).append(daily)

        liquidity_frames.append(
            daily.group_by("trading_date")
            .agg(
                pl.col("day_volume").sum().alias("vol"),
                (pl.col("day_close") * pl.col("day_volume")).sum().alias("pv"),
            )
            .sort("trading_date")
            .with_columns(
                (pl.col("pv") / pl.col("vol")).alias("vwap_proxy"),
            )
            .with_columns(
                (pl.col("vol") * pl.col("vwap_proxy") * multiplier)
                .rolling_median(window_size=60, min_samples=20)
                .alias("adv_proxy"),
                pl.lit(product).alias("product"),
            )
            .select("product", "trading_date", "adv_proxy")
        )

    # --- volume-counting break verification (trap 6)
    breaks_confirmed: dict[str, bool] = {}
    break_rows: list[dict[str, Any]] = []
    verify = volume_cfg.get("verify", {}) or {}
    lo_ok, hi_ok = verify.get("confirm_ratio", [0.35, 0.65])
    for exchange, section in (volume_cfg.get("exchanges", {}) or {}).items():
        bd = section.get("break_date")
        if bd is None or exchange not in daily_by_exchange:
            continue
        break_date = bd if isinstance(bd, dt.date) else dt.date.fromisoformat(str(bd))
        pooled = pl.concat(daily_by_exchange[exchange], how="vertical")
        ratio = volume_break_ratio(pooled, break_date)
        confirmed = ratio is not None and lo_ok <= ratio <= hi_ok
        breaks_confirmed[str(exchange)] = bool(confirmed)
        break_rows.append(
            {
                "exchange": str(exchange),
                "break_date": break_date,
                "ratio": ratio,
                "confirmed": confirmed,
            }
        )
        if ratio is not None and not confirmed:
            logger.warning(
                "volume-break check {}: ratio {:.2f} outside [{}, {}] — config date kept",
                exchange,
                ratio,
                lo_ok,
                hi_ok,
            )

    # --- era flags from config
    era_rows: list[dict[str, Any]] = []
    for ov in universe_cfg.get("era_overrides", []) or []:
        for prod in ov["products"]:
            era_rows.append(
                {
                    "product": str(prod).upper(),
                    "date_start": _as_date(ov["date_start"]),
                    "date_end": _as_date(ov["date_end"]),
                    "flag": str(ov["flag"]),
                    "source": "config",
                }
            )
    for exchange, section in (volume_cfg.get("exchanges", {}) or {}).items():
        bd = section.get("break_date")
        if bd is None:
            continue
        for row in product_rows:
            if row["exchange"] == exchange:
                era_rows.append(
                    {
                        "product": row["product"],
                        "date_start": row["first_trading_date"],
                        "date_end": _as_date(bd) - dt.timedelta(days=1),
                        "flag": "double_counting",
                        "source": "volume_counting.yaml",
                    }
                )

    # --- writes
    paths.universe_dir.mkdir(parents=True, exist_ok=True)
    _write(pl.DataFrame(product_rows).sort("product"), paths.universe_products)
    if era_rows:
        _write(pl.DataFrame(era_rows).sort("product", "date_start"), paths.era_flags)
    if break_rows:
        _write(pl.DataFrame(break_rows).sort("exchange"), paths.volume_break_check)
    if liquidity_frames:
        _write(
            pl.concat(liquidity_frames, how="vertical").sort("product", "trading_date"),
            paths.liquidity,
        )

    return UniverseReport(
        products=[str(r["product"]) for r in product_rows],
        multiplier_mismatches=mismatches,
        breaks_confirmed=breaks_confirmed,
    )


def _as_date(v: object) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v))


def _write(df: pl.DataFrame, path: Any) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def resolve_dev_universe(
    paths: FuturesPaths, universe_cfg: Mapping[str, Any], size: int | None = None
) -> list[str]:
    """Top-N dev candidates by latest ADV (development convenience — NOT PIT)."""
    candidates = [str(p).upper() for p in universe_cfg.get("dev_candidates", [])]
    size = size or int(universe_cfg.get("dev_size", 10))
    if not paths.liquidity.exists():
        return candidates[:size]
    liq = (
        pl.read_parquet(paths.liquidity)
        .filter(pl.col("product").is_in(candidates) & pl.col("adv_proxy").is_not_null())
        .group_by("product")
        .agg(pl.col("adv_proxy").last())
        .sort("adv_proxy", descending=True)
    )
    ranked = liq["product"].to_list()
    return ranked[:size] if ranked else candidates[:size]
