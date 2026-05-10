"""Tests for AKshareSource — wrapper around ak.stock_zh_a_hist + friends.

We mock the ak.* functions (rather than mocking HTTP) because:
  * AKshare's own HTTP layer is not stable (curl_cffi vs requests vs httpx).
  * The interface contract that matters is "what columns / dtypes does
    ak.stock_zh_a_hist return," which is upstream of HTTP.

Schema drift is the most likely break; we keep ONE network smoke test to
catch it the moment it happens.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import polars as pl
import pytest

from data_fetcher.akshare_source import AKshareSource
from data_fetcher.exceptions import (
    EmptyResponseError,
    RateLimitError,
    UniverseUnavailableError,
)
from data_fetcher.storage import OHLCV_SCHEMA


# ---------- Mock data ----------


def _canonical_akshare_ohlcv_response() -> pd.DataFrame:
    """A realistic 3-row sample of what ak.stock_zh_a_hist returns.

    Chinese column headers; date as datetime64[ns]; numeric as float64.
    """
    return pd.DataFrame(
        {
            "日期": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "股票代码": ["600519", "600519", "600519"],
            "开盘": [1700.0, 1710.0, 1720.0],
            "收盘": [1715.0, 1718.0, 1725.0],
            "最高": [1720.0, 1725.0, 1730.0],
            "最低": [1695.0, 1705.0, 1715.0],
            "成交量": [10000.0, 12000.0, 11000.0],
            "成交额": [17_150_000.0, 20_616_000.0, 18_975_000.0],
            "振幅": [1.5, 1.2, 0.9],
            "涨跌幅": [0.5, 0.17, 0.41],
            "涨跌额": [8.5, 3.0, 7.0],
            "换手率": [0.05, 0.06, 0.055],
        }
    )


def _canonical_akshare_index_cons_response() -> pd.DataFrame:
    """Realistic ak.index_stock_cons output: 3 today's-roster rows with Chinese
    headers. No out_date column upstream — out_date is synthesised as NULL by us.
    """
    return pd.DataFrame(
        {
            "品种代码": ["600519", "000001", "300750"],
            "品种名称": ["贵州茅台", "平安银行", "宁德时代"],
            "纳入日期": ["2005-04-08", "2005-04-08", "2018-12-14"],
        }
    )


def _canonical_akshare_calendar_response() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    return pd.DataFrame({"trade_date": [d.date() for d in dates]})


# ---------- Tests ----------


def test_pull_ohlcv_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_hist(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return _canonical_akshare_ohlcv_response()

    src = AKshareSource()
    monkeypatch.setattr("akshare.stock_zh_a_hist", fake_hist)
    df = src.pull_ohlcv(
        symbol="SH600519",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
        adjust="qfq",
    )
    # Right call args: bare 6-digit symbol, YYYYMMDD dates, period=daily
    assert captured["symbol"] == "600519"
    assert captured["period"] == "daily"
    assert captured["start_date"] == "20240101"
    assert captured["end_date"] == "20240131"
    assert captured["adjust"] == "qfq"

    # Schema match
    assert df.schema == OHLCV_SCHEMA
    assert df.height == 3
    # Symbol column gets the canonical (SH/SZ) form, not the bare upstream form.
    assert set(df["symbol"].unique().to_list()) == {"SH600519"}
    # AKshare returns volume in 手 (lots); we convert to shares (×100).
    assert df["volume"][0] == 10000.0 * 100.0
    # VWAP is the TYP proxy (high+low+close)/3, not turnover/volume — see
    # _normalise_ohlcv for why. Row 0: high=1720, low=1695, close=1715 → 1710.
    expected_vwap_0 = (1720.0 + 1695.0 + 1715.0) / 3.0
    assert abs(df["vwap"][0] - expected_vwap_0) < 1e-9
    # And it must lie between low and high.
    assert (df["low"] <= df["vwap"]).all() and (df["vwap"] <= df["high"]).all()


def test_pull_ohlcv_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "akshare.stock_zh_a_hist",
        lambda **kwargs: pd.DataFrame(),
    )
    src = AKshareSource()
    with pytest.raises(EmptyResponseError):
        src.pull_ohlcv(
            symbol="SH600519",
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31"),
        )


def test_pull_ohlcv_zero_volume_vwap_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A halt day can have zero volume; vwap (TYP proxy) is still well-defined
    because it depends only on high/low/close, not volume."""
    df_in = _canonical_akshare_ohlcv_response()
    df_in.iloc[1, df_in.columns.get_loc("成交量")] = 0.0
    df_in.iloc[1, df_in.columns.get_loc("成交额")] = 0.0
    monkeypatch.setattr("akshare.stock_zh_a_hist", lambda **kwargs: df_in)
    src = AKshareSource()
    df = src.pull_ohlcv(
        symbol="SH600519",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
    )
    # vwap = (high+low+close)/3 is finite even when volume is zero.
    assert df["vwap"][1] == (df["high"][1] + df["low"][1] + df["close"][1]) / 3.0
    assert df["volume"][1] == 0.0


def test_pull_ohlcv_retry_on_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """First two calls raise; third succeeds. We expect 3 calls total."""
    counter = {"n": 0}
    canonical = _canonical_akshare_ohlcv_response()

    def flaky(**kwargs):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        if counter["n"] < 3:
            import requests

            raise requests.exceptions.ConnectionError("simulated upstream blip")
        return canonical

    monkeypatch.setattr("akshare.stock_zh_a_hist", flaky)
    src = AKshareSource(max_retries=3, base_backoff_s=0.0)
    df = src.pull_ohlcv(
        symbol="SH600519",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
    )
    assert counter["n"] == 3
    assert df.height == 3


def test_pull_ohlcv_retry_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fails(**kwargs):  # type: ignore[no-untyped-def]
        import requests

        raise requests.exceptions.ConnectionError("simulated outage")

    monkeypatch.setattr("akshare.stock_zh_a_hist", always_fails)
    src = AKshareSource(max_retries=2, base_backoff_s=0.0)
    with pytest.raises(RateLimitError):
        src.pull_ohlcv(
            symbol="SH600519",
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31"),
        )


def test_list_universe_event_log_csi300(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_cons(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return _canonical_akshare_index_cons_response()

    monkeypatch.setattr("akshare.index_stock_cons", fake_cons)
    src = AKshareSource()
    df = src.list_universe_event_log("csi300")
    assert captured["symbol"] == "000300"
    # Expected schema
    assert df.schema["stock_code"] == pl.Utf8
    assert df.schema["in_date"] == pl.Date
    assert df.schema["out_date"] == pl.Date
    assert df.height == 3
    # v1 invariant: every row has out_date NULL (we have no removal data).
    assert df["out_date"].null_count() == df.height
    # Stock codes preserved as 6-digit strings.
    assert set(df["stock_code"].to_list()) == {"600519", "000001", "300750"}
    # in_date parsed correctly.
    catl = df.filter(pl.col("stock_code") == "300750")
    assert catl["in_date"][0] == dt.date(2018, 12, 14)


def test_list_universe_event_log_unknown_raises() -> None:
    src = AKshareSource()
    with pytest.raises(UniverseUnavailableError):
        src.list_universe_event_log("not_an_index")


def test_trade_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "akshare.tool_trade_date_hist_sina",
        lambda: _canonical_akshare_calendar_response(),
    )
    src = AKshareSource()
    cal = src.trade_calendar()
    assert isinstance(cal, pd.DatetimeIndex)
    assert len(cal) == len(pd.bdate_range("2024-01-01", "2024-01-31"))
    # All entries normalized to midnight.
    assert all(t == t.normalize() for t in cal)


# ---------- Network smoke (gated) ----------


@pytest.mark.network
def test_smoke_one_symbol_real_akshare() -> None:
    """Real AKshare call. Will fail loudly if their schema drifts.
    Run with `pytest --network`."""
    src = AKshareSource()
    df = src.pull_ohlcv(
        symbol="SH600519",
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-31"),
        adjust="qfq",
    )
    assert df.height > 10  # ~21 trading days in Jan 2024
    assert df.schema == OHLCV_SCHEMA
    assert (df["close"] > 0).all()
    # VWAP (TYP proxy = HLC/3) must lie between low and high on every row.
    assert (df["low"] <= df["vwap"]).all()
    assert (df["vwap"] <= df["high"]).all()


@pytest.mark.network
def test_smoke_csi300_event_log_real_akshare() -> None:
    src = AKshareSource()
    df = src.list_universe_event_log("csi300")
    # CSI 300 has exactly 300 members — drift signals upstream weirdness.
    assert 295 <= df.height <= 305
    assert df.schema["stock_code"] == pl.Utf8
    # Every row has 6-digit stock_code.
    assert all(len(c) == 6 and c.isdigit() for c in df["stock_code"].to_list())
    # v1 invariant: no removal data, so every out_date is NULL.
    assert df["out_date"].null_count() == df.height


@pytest.mark.network
def test_smoke_trade_calendar_real_akshare() -> None:
    src = AKshareSource()
    cal = src.trade_calendar()
    # CSI 300 launched 2005-04; the calendar should cover all the way to recent.
    assert cal[0] <= pd.Timestamp("2010-01-01")
    assert cal[-1] >= pd.Timestamp("2024-01-01")
    # Must include known trading day, exclude known weekend.
    assert pd.Timestamp("2024-01-02") in cal
    assert pd.Timestamp("2024-01-06") not in cal  # Saturday
    # Confirm no duplicate dates.
    assert len(cal) == len(set(cal))
    _ = dt.date.today()  # silence unused-import lint
