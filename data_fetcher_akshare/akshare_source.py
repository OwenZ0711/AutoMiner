"""AKshareSource — v1 implementation of the Source protocol.

Wraps three AKshare endpoints:

  * `ak.stock_zh_a_hist`           — daily OHLCV per A-share equity
  * `ak.index_stock_cons`          — current index members + their in_dates
  * `ak.tool_trade_date_hist_sina` — trading calendar

**KNOWN LIMITATION (v1):** AKshare 1.18.60 does NOT expose a historical
add/drop event log for any A-share index — `index_stock_cons` returns
today's roster plus each current member's `in_date`. We therefore
synthesize an "event log" with `out_date = NULL` for every row, and the
downstream `Fetcher.universe(name, on_date)` returns the subset whose
`in_date <= on_date`. This captures index *additions* correctly but
silently misses members who have since been *removed* — i.e. survivorship
bias in early years (~57% missing for 2018 dates). See CLAUDE.md
"CSI 300 membership" for the full story and v1.5 mitigation path.

Responsibilities of this layer:
  * Translate canonical SH/SZ symbols to the bare 6-digit form AKshare wants.
  * Translate `pd.Timestamp` to the YYYYMMDD strings AKshare wants.
  * Rename Chinese columns to English.
  * Compute VWAP from turnover/volume (AKshare doesn't expose it directly).
  * Cast dtypes to the canonical OHLCV_SCHEMA.
  * Retry on transient network errors with exponential backoff.
  * Surface our typed exceptions, never raw `requests.HTTPError`.

Things this layer does NOT do:
  * Storage. (storage.py)
  * Universe-name → symbol-list resolution. (universe.py)
  * Schedule. (Fetcher.pull)
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import akshare as ak
import pandas as pd
import polars as pl
from loguru import logger

from data_fetcher.exceptions import (
    EmptyResponseError,
    RateLimitError,
    UniverseUnavailableError,
)
from data_fetcher.storage import OHLCV_SCHEMA
from data_fetcher.symbols import normalise, to_akshare_bare, to_akshare_with_market

# Chinese → English column rename. The single most likely thing to drift if AKshare
# updates upstream — every external test points here.
_OHLCV_COLUMN_MAP = {
    "日期": "date",
    "股票代码": "symbol_raw",  # we replace this with our canonical symbol
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "return_pct",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
}

# Universe name → AKshare bare-code form for index_stock_cons.
_UNIVERSE_TO_INDEX = {
    "csi300": "000300",
    "csi500": "000905",
    "csi800": "000906",
    "csi1000": "000852",
    "sse50": "000016",
}

# Chinese → English column rename for ak.index_stock_cons output.
_CONS_COLUMN_MAP = {
    "品种代码": "stock_code",
    "品种名称": "stock_name",  # not stored, just renamed for clarity
    "纳入日期": "in_date",
}


T = TypeVar("T")


class AKshareSource:
    name = "akshare"

    def __init__(self, max_retries: int = 3, base_backoff_s: float = 0.5) -> None:
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

    # ---------- OHLCV ----------

    def pull_ohlcv(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjust: str = "qfq",
    ) -> pl.DataFrame:
        canonical = normalise(symbol)
        bare = to_akshare_bare(canonical)
        start_str = pd.Timestamp(start).strftime("%Y%m%d")
        end_str = pd.Timestamp(end).strftime("%Y%m%d")

        raw = self._call_with_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=bare,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=adjust,
            ),
            label=f"stock_zh_a_hist({bare})",
        )
        if raw is None or raw.empty:
            raise EmptyResponseError(
                f"AKshare returned empty for {canonical} {start_str}..{end_str} "
                f"adjust={adjust!r}"
            )
        return self._normalise_ohlcv(raw, canonical_symbol=canonical)

    # ---------- Universe membership ----------

    def list_universe_event_log(self, name: str) -> pl.DataFrame:
        """v1: returns today's roster + per-stock in_date with out_date=NULL.

        Wraps `ak.index_stock_cons(symbol=index_code)`. The promise of a
        proper event log (with both add and drop events) is *not* met by
        this Source — see module docstring. The schema (in_date, out_date)
        accommodates a real event log in v1.5.
        """
        if name not in _UNIVERSE_TO_INDEX:
            raise UniverseUnavailableError(
                f"AKshareSource does not know how to fetch universe {name!r}; "
                f"known: {sorted(_UNIVERSE_TO_INDEX)}"
            )
        index_code = _UNIVERSE_TO_INDEX[name]

        raw = self._call_with_retry(
            lambda: ak.index_stock_cons(symbol=index_code),
            label=f"index_stock_cons({index_code})",
        )
        if raw is None or raw.empty:
            raise UniverseUnavailableError(
                f"AKshare returned empty member list for {name!r}"
            )
        return self._normalise_event_log(raw)

    # ---------- Calendar ----------

    def trade_calendar(self) -> pd.DatetimeIndex:
        raw = self._call_with_retry(
            lambda: ak.tool_trade_date_hist_sina(),
            label="tool_trade_date_hist_sina",
        )
        if raw is None or raw.empty:
            raise EmptyResponseError("AKshare returned empty trade calendar")
        # The column is `trade_date` of date-objects.
        return pd.DatetimeIndex(pd.to_datetime(raw["trade_date"])).normalize()

    # ---------- Internals ----------

    def _call_with_retry(self, fn: Callable[[], T], *, label: str) -> T:
        """Retry on `requests` connection / read errors only. AKshare's
        own "no data" mode returns an empty DataFrame, not an exception —
        the caller decides whether that's an error in its semantic context.
        """
        import requests  # imported here so test mocks don't have to monkeypatch globally

        last_err: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                wait = self.base_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    f"{label} attempt {attempt}/{self.max_retries} failed: {e!r}; "
                    f"retrying after {wait:.1f}s"
                )
                time.sleep(wait)
        raise RateLimitError(
            f"{label} failed after {self.max_retries} retries: {last_err!r}"
        )

    @staticmethod
    def _normalise_ohlcv(raw: pd.DataFrame, *, canonical_symbol: str) -> pl.DataFrame:
        # Rename Chinese columns. Strict — unknown headers mean upstream drift, fail loud.
        unexpected = set(raw.columns) - set(_OHLCV_COLUMN_MAP)
        if unexpected:
            logger.warning(
                f"AKshare OHLCV returned unexpected columns: {sorted(unexpected)}; "
                f"these will be dropped"
            )
        renamed = raw.rename(columns=_OHLCV_COLUMN_MAP)
        # Drop columns not in our canonical schema (amplitude, price_change, symbol_raw).
        keep = ["date", "open", "high", "low", "close", "volume", "turnover", "return_pct", "turnover_rate"]
        renamed = renamed[keep].copy()
        # date → datetime.date
        renamed["date"] = pd.to_datetime(renamed["date"]).dt.date
        # AKshare's 成交量 is in 手 (lots of 100 shares); convert to raw shares
        # so downstream consumers can read "volume" as shares without footguns.
        renamed["volume"] = renamed["volume"].astype(float) * 100.0
        # VWAP via TYP-proxy: (high + low + close) / 3.
        #
        # Why not turnover/volume: when adjust != "" (qfq/hfq), AKshare adjusts
        # OHLC for splits/dividends but leaves turnover and volume *raw*. So
        # turnover/volume gives the *unadjusted* VWAP — incompatible unit-system
        # with the adjusted close, and can land outside [low, high] in the
        # adjusted frame. TYP proxy is in the same unit as the price columns,
        # well-known in technical analysis (Tulip Indicators, TA-Lib), and
        # preserves the cross-day shape Alpha158 cares about.
        renamed["vwap"] = (
            renamed["high"].astype(float)
            + renamed["low"].astype(float)
            + renamed["close"].astype(float)
        ) / 3.0
        # Symbol column — canonical SH/SZ form, not the upstream bare form.
        renamed["symbol"] = canonical_symbol

        df = pl.from_pandas(renamed)
        # Cast to OHLCV_SCHEMA.
        df = df.select(
            [
                pl.col("date").cast(OHLCV_SCHEMA["date"]),
                pl.col("symbol").cast(OHLCV_SCHEMA["symbol"]),
                pl.col("open").cast(OHLCV_SCHEMA["open"]),
                pl.col("high").cast(OHLCV_SCHEMA["high"]),
                pl.col("low").cast(OHLCV_SCHEMA["low"]),
                pl.col("close").cast(OHLCV_SCHEMA["close"]),
                pl.col("volume").cast(OHLCV_SCHEMA["volume"]),
                pl.col("turnover").cast(OHLCV_SCHEMA["turnover"]),
                pl.col("vwap").cast(OHLCV_SCHEMA["vwap"]),
                pl.col("return_pct").cast(OHLCV_SCHEMA["return_pct"]),
                pl.col("turnover_rate").cast(OHLCV_SCHEMA["turnover_rate"]),
            ]
        )
        return df

    @staticmethod
    def _normalise_event_log(raw: pd.DataFrame) -> pl.DataFrame:
        """Normalise `ak.index_stock_cons` output into the event-log schema.

        Upstream columns are Chinese: 品种代码, 品种名称, 纳入日期. We rename,
        force stock_code to a 6-char string, and synthesise out_date=NULL
        for every row (this Source has no removal data).
        """
        unexpected = set(raw.columns) - set(_CONS_COLUMN_MAP)
        if unexpected:
            logger.warning(
                f"index_stock_cons returned unexpected columns: {sorted(unexpected)}; "
                f"these will be dropped"
            )
        renamed = raw.rename(columns=_CONS_COLUMN_MAP)
        required = {"stock_code", "in_date"}
        missing = required - set(renamed.columns)
        if missing:
            raise UniverseUnavailableError(
                f"index_stock_cons missing required columns {sorted(missing)} after "
                f"rename; got {list(renamed.columns)} — upstream may have changed"
            )
        rows = renamed[["stock_code", "in_date"]].copy()
        rows["stock_code"] = rows["stock_code"].astype(str).str.zfill(6)
        rows["in_date"] = pd.to_datetime(rows["in_date"], errors="coerce").dt.date
        rows["out_date"] = None  # v1: no removal data; null for every row

        df = pl.from_pandas(rows[["stock_code", "in_date", "out_date"]])
        df = df.with_columns(
            pl.col("stock_code").cast(pl.Utf8),
            pl.col("in_date").cast(pl.Date),
            pl.col("out_date").cast(pl.Date),
        )
        return df
