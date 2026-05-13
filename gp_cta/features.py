"""Feature construction for the GP-CTA baseline."""

from __future__ import annotations

import polars as pl


RAW_COLUMNS = ("open", "high", "low", "close", "volume", "turnover", "vwap")

FEATURE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "ret_1d",
    "ret_5d",
    "intraday_ret",
    "hl_range",
    "vwap_gap",
    "ma_gap_5",
    "ma_gap_20",
    "vol_20d",
    "volume_ratio_5_20",
)


def build_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Add simple per-asset price/volume features.

    The output keeps the original OHLCV columns and adds `next_return_1d` for
    evaluation. Formula generation must use only `FEATURE_COLUMNS`, not the
    target column.
    """
    required = {"date", "symbol", *RAW_COLUMNS}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    df = panel.sort(["symbol", "date"])
    df = df.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias(
                "ret_1d"
            ),
            (pl.col("close") / pl.col("close").shift(5).over("symbol") - 1.0).alias(
                "ret_5d"
            ),
            (pl.col("close") / pl.col("open") - 1.0).alias("intraday_ret"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_range"),
            (pl.col("vwap") / pl.col("close") - 1.0).alias("vwap_gap"),
            (
                pl.col("close")
                / pl.col("close").rolling_mean(window_size=5).over("symbol")
                - 1.0
            ).alias("ma_gap_5"),
            (
                pl.col("close")
                / pl.col("close").rolling_mean(window_size=20).over("symbol")
                - 1.0
            ).alias("ma_gap_20"),
            (
                pl.col("volume").rolling_mean(window_size=5).over("symbol")
                / pl.col("volume").rolling_mean(window_size=20).over("symbol")
                - 1.0
            ).alias("volume_ratio_5_20"),
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1.0).alias(
                "next_return_1d"
            ),
        ]
    )
    df = df.with_columns(
        pl.col("ret_1d").rolling_std(window_size=20).over("symbol").alias("vol_20d")
    )
    float_cols = [
        name
        for name, dtype in df.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    return df.with_columns(
        [
            pl.when(pl.col(name).is_infinite())
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
            for name in float_cols
        ]
    )
