"""Shared pytest fixtures for data_fetcher.tests.

The strategy from the plan:
  * Offline tests run against tiny in-repo fixtures.
  * Network tests are gated behind `--network` and skipped by default.
  * Big multi-MB fixtures (the 5-symbol Phase-1 OHLCV cache) are produced
    by the network smoke step at the end of Phase 1, not committed eagerly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="Run tests that hit AKshare upstream (slow, flaky in CI).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="needs --network flag")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: test makes a real upstream call (AKshare); needs --network"
    )


# ---------- Fixture builders ----------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def trade_calendar_fixture(fixtures_dir: Path) -> pd.DatetimeIndex:
    """A small synthetic trading calendar covering 2023-01-01..2024-12-31 weekdays
    minus a hand-picked set of CN public-holiday dates.

    Uses a parquet cache so consecutive test sessions skip the build step.
    """
    path = fixtures_dir / "trade_calendar_2023_2024.parquet"
    if path.exists():
        df = pl.read_parquet(path)
        return pd.DatetimeIndex(df["trade_date"].to_pandas()).normalize()

    weekdays = pd.bdate_range("2023-01-01", "2024-12-31")
    # Hand-picked CN holidays (Chinese New Year, Qingming, Labor Day, etc.).
    holidays = pd.to_datetime(
        [
            "2023-01-02",  # New Year observed
            "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
            "2023-04-05",  # Qingming
            "2023-05-01", "2023-05-02", "2023-05-03",  # Labor
            "2023-06-22", "2023-06-23",  # Dragon Boat
            "2023-09-29",  # Mid-Autumn
            "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
            "2024-01-01",
            "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16",
            "2024-04-04", "2024-04-05",
            "2024-05-01", "2024-05-02", "2024-05-03",
            "2024-06-10",
            "2024-09-16", "2024-09-17",
            "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07",
        ]
    )
    cal = weekdays.difference(holidays)
    dates = [d.date() for d in cal]
    pl.DataFrame({"trade_date": dates}, schema={"trade_date": pl.Date}).write_parquet(
        path
    )
    return pd.DatetimeIndex(cal).normalize()


@pytest.fixture(scope="session")
def csi300_event_log_fixture(fixtures_dir: Path) -> pl.DataFrame:
    """A tiny synthetic CSI 300 add/drop event log — 8 rows covering edge cases:
    members from inception, members joined and never left, members joined and left,
    a same-day swap pair around a rebalance.
    """
    path = fixtures_dir / "csi300_event_log_synthetic.parquet"
    if path.exists():
        return pl.read_parquet(path)

    rows = [
        # stock_code, in_date, out_date (None = still in)
        ("600000", dt.date(2018, 1, 1), None),                  # joined pre-window, still in
        ("600519", dt.date(2018, 1, 1), None),                  # joined pre-window, still in
        ("000001", dt.date(2018, 1, 1), dt.date(2024, 6, 14)),  # joined pre-window, left mid-window
        ("300750", dt.date(2018, 12, 14), None),                # joined Dec 2018 rebalance
        ("688981", dt.date(2020, 7, 13), None),                 # joined post-2018 (STAR)
        ("000333", dt.date(2019, 6, 17), dt.date(2022, 12, 12)), # in then out
        ("002594", dt.date(2021, 6, 15), None),                  # CATL in
        ("601398", dt.date(2018, 1, 1), dt.date(2018, 1, 2)),    # same-day-ish drop
    ]
    df = pl.DataFrame(
        rows,
        schema=["stock_code", "in_date", "out_date"],
        orient="row",
    )
    df.write_parquet(path)
    return df


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """Per-test tmp dir that mimics ~/AutoLLM_data/ structure."""
    root = tmp_path / "AutoLLM_data"
    (root / "raw").mkdir(parents=True)
    (root / "universe").mkdir(parents=True)
    (root / "calendar").mkdir(parents=True)
    return root


@pytest.fixture(scope="session")
def synthetic_ohlcv_panel() -> pl.DataFrame:
    """A 30-day synthetic OHLCV panel for 2 symbols, used by storage tests."""
    dates = pd.bdate_range("2024-01-02", periods=30).date
    rows = []
    for sym in ("SH600519", "SZ000001"):
        for i, d in enumerate(dates):
            base = 100.0 + (i * 0.5 if sym == "SH600519" else i * 0.2)
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": base,
                    "high": base + 1.0,
                    "low": base - 1.0,
                    "close": base + 0.3,
                    "volume": 1_000_000.0 + i * 1_000,
                    "turnover": (base + 0.3) * (1_000_000.0 + i * 1_000),
                    "vwap": base + 0.2,
                    "return_pct": 0.3,
                    "turnover_rate": 0.5,
                }
            )
    return pl.DataFrame(rows)
