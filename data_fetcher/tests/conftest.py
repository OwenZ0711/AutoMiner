"""Shared fixtures: synthetic raw trees in BOTH layouts + marker gating.

`make_raw_tree` writes hand-built CSVs from a declarative bar spec so every
trap has a pinned, readable fixture. Real-data tests are gated behind the
`realdata` marker and the --realdata flag (they need Future_minute_data/).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

HEADER_12 = "exchange,symbol,open,close,high,low,amount,volume,position,bob,eob,type"
HEADER_13 = HEADER_12 + ",sequence"


@dataclass(slots=True)
class Bar:
    """One raw CSV row. Times are naive local (Asia/Shanghai)."""

    bob: dt.datetime
    open: float = 100.0
    close: float = 100.0
    high: float = 100.0
    low: float = 100.0
    volume: float = 10.0
    amount: float | None = None  # default: volume * close * 10 (multiplier 10)
    position: float = 1000.0
    nan_row: bool = False  # emit a literal-"nan" row (trap 5)

    def row(self, exchange: str, symbol: str, extra_sequence: bool) -> str:
        if self.nan_row:
            base = f"{exchange},{symbol}," + ",".join(["nan"] * 10)
            return base + ",nan" if extra_sequence else base
        eob = self.bob + dt.timedelta(minutes=1)
        amount = self.amount if self.amount is not None else self.volume * self.close * 10
        base = (
            f"{exchange},{symbol},{self.open},{self.close},{self.high},{self.low},"
            f"{amount},{self.volume},{self.position},"
            f"{self.bob:%Y-%m-%d %H:%M:%S}+08:00,{eob:%Y-%m-%d %H:%M:%S}+08:00,14"
        )
        return base + ",2" if extra_sequence else base


@dataclass(slots=True)
class ContractSpec:
    exchange: str  # "SHFE"
    symbol: str  # "RB2001" — case as it should appear IN the file
    bars: Sequence[Bar] = field(default_factory=list)


def write_historical_tree(root: Path, contracts: Sequence[ContractSpec]) -> Path:
    """`root/2005-202506/EXCH/PROD/CONTRACT.csv` — 12 cols, uppercase names."""
    tree = root / "2005-202506"
    for c in contracts:
        symbol_upper = c.symbol.upper()
        product = "".join(ch for ch in symbol_upper if ch.isalpha())
        path = tree / c.exchange / product / f"{symbol_upper}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [HEADER_12] + [b.row(c.exchange, symbol_upper, False) for b in c.bars]
        path.write_text("\n".join(lines) + "\n")
    return tree


def write_daily_tree(root: Path, contracts: Sequence[ContractSpec]) -> Path:
    """`root/2026/YYYYMM/YYYYMMDD/<symbol>.csv` — 13 cols, calendar-day keyed.

    Bars land in the folder of their bob CALENDAR date (that is the trap:
    post-midnight night bars sit in the next day's folder).
    """
    tree = root / "2026"
    by_day: dict[tuple[str, dt.date], list[str]] = {}
    meta: dict[str, str] = {}
    for c in contracts:
        meta[c.symbol] = c.exchange
        for b in c.bars:
            key = (c.symbol, b.bob.date())
            # symbol INSIDE 2026 files matches the filename case
            by_day.setdefault(key, []).append(b.row(c.exchange, c.symbol, True))
    for (symbol, day), rows in sorted(by_day.items()):
        path = tree / f"{day:%Y%m}" / f"{day:%Y%m%d}" / f"{symbol}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([HEADER_13, *rows]) + "\n")
    return tree


def day_session_bars(
    day: dt.date, *, n: int = 5, start_hm: tuple[int, int] = (9, 0), **kw: object
) -> list[Bar]:
    """n consecutive 1-minute day-session bars starting at start_hm."""
    t0 = dt.datetime.combine(day, dt.time(*start_hm))
    return [Bar(bob=t0 + dt.timedelta(minutes=i), **kw) for i in range(n)]  # type: ignore[arg-type]


@pytest.fixture()
def tmp_paths(tmp_path: Path):  # type: ignore[no-untyped-def]
    from futures_common.paths import FuturesPaths

    return FuturesPaths(data_root=tmp_path / "data", results_root=tmp_path / "results")
