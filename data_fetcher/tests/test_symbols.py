"""Tests for symbols.normalise — every row of the table in PHASE1_PLAN.

The normaliser is the single most-relied-on function in the module: every
storage path, every API call, every fixture key passes through it. 100%
table coverage + an explicit failure case for invalid input is the bar.
"""

from __future__ import annotations

import pytest

from data_fetcher.exceptions import InvalidSymbolError
from data_fetcher.symbols import normalise, to_akshare_bare, to_akshare_with_market


# (raw_input, expected_normalised) — one row per case from the plan.
TABLE = [
    # SH main board (60xxxx, 61xxxx)
    ("600519", "SH600519"),
    ("601318", "SH601318"),
    # SH STAR market (688xxx)
    ("688981", "SH688981"),
    # SZ main board (000xxx)
    ("000001", "SZ000001"),
    # SZ SME (002xxx)
    ("002594", "SZ002594"),
    # SZ ChiNext (300xxx)
    ("300750", "SZ300750"),
    # SZ B-share (200xxx)
    ("200012", "SZ200012"),
    # SH B-share (900xxx)
    ("900957", "SH900957"),
    # Already normalised — passthrough
    ("SH600519", "SH600519"),
    ("SZ000001", "SZ000001"),
    # Lower-case prefix
    ("sh600519", "SH600519"),
    ("sz000001", "SZ000001"),
    # Wind / Tushare dot-suffix
    ("600519.SH", "SH600519"),
    ("000001.SZ", "SZ000001"),
    # Bloomberg / XSHG-XSHE suffix
    ("600519.XSHG", "SH600519"),
    ("000001.XSHE", "SZ000001"),
    # Dot-suffix B-share
    ("900957.SH", "SH900957"),
    # Whitespace tolerance
    ("  600519  ", "SH600519"),
]

INVALID = [
    "",
    "12345",  # 5 digits
    "1234567",  # 7 digits
    "ABCDEF",
    "SH123",  # too short after prefix
    "XX600519",  # unknown prefix
    "600519.XX",  # unknown suffix
    "700519",  # 7-prefix not used by SH/SZ A/B shares
    None,  # type: ignore[list-item]
    123456,  # type: ignore[list-item]
]


@pytest.mark.parametrize(("raw", "expected"), TABLE)
def test_normalise_table(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


@pytest.mark.parametrize("bad", INVALID)
def test_normalise_invalid_raises(bad: object) -> None:
    with pytest.raises(InvalidSymbolError):
        normalise(bad)  # type: ignore[arg-type]


def test_normalise_is_idempotent() -> None:
    """normalise(normalise(x)) == normalise(x) for every valid input."""
    for raw, _ in TABLE:
        once = normalise(raw)
        twice = normalise(once)
        assert once == twice, f"non-idempotent on {raw!r}: {once} -> {twice}"


def test_to_akshare_bare() -> None:
    """AKshare wants 6-digit codes without market prefix for stock_zh_a_hist."""
    assert to_akshare_bare("SH600519") == "600519"
    assert to_akshare_bare("SZ000001") == "000001"
    assert to_akshare_bare("600519") == "600519"  # already bare


def test_to_akshare_with_market() -> None:
    """AKshare wants 'sh000300' / 'sz000001' lower-case with prefix for index_stock_hist."""
    assert to_akshare_with_market("SH000300") == "sh000300"
    assert to_akshare_with_market("SZ399001") == "sz399001"
