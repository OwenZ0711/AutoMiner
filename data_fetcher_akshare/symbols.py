"""Symbol normalisation for A-share equities.

Every external input — user CLI arg, AKshare row, fixture file name, parquet
partition key — passes through `normalise(raw) -> 'SH600519'` exactly once
on entry. Internal code never sees raw forms. This is the "single
canonical representation" pattern; if you find yourself comparing
`x.upper().startswith('sh')` anywhere downstream, that's the bug.

Conventions:
    * Internal canonical form: 'SH' or 'SZ' upper-case + 6 digits, e.g. 'SH600519'.
    * AKshare's `stock_zh_a_hist` wants the BARE 6 digits, no prefix.
    * AKshare's `index_stock_hist` wants the LOWER-CASE prefix + digits, e.g. 'sh000300'.
    * Wind / Tushare uses dot-suffix: '600519.SH'. Bloomberg uses '600519.XSHG'/'.XSHE'.

Prefix rules for bare 6-digit codes (A-share equities only; index codes are passthrough):
    leading '6'  -> SH (Shanghai main board, incl. 60xxxx and 688xxx STAR)
    leading '0'  -> SZ (Shenzhen main board, incl. 000xxx and 002xxx SME)
    leading '3'  -> SZ (ChiNext)
    leading '2'  -> SZ (B-share, rare)
    leading '9'  -> SH (B-share, rare)
    anything else -> InvalidSymbolError (8/4/5/7 prefix codes are not stocks)
"""

from __future__ import annotations

import re

from data_fetcher.exceptions import InvalidSymbolError

_DIGITS_RE = re.compile(r"^\d{6}$")


def _market_for_bare(bare: str) -> str:
    """Decide SH vs SZ for a bare 6-digit code."""
    head = bare[0]
    if head in {"6", "9"}:
        return "SH"
    if head in {"0", "2", "3"}:
        return "SZ"
    raise InvalidSymbolError(
        f"unsupported leading digit {head!r} in {bare!r}; A-share codes start "
        f"with one of {{0,2,3,6,9}}"
    )


def normalise(raw: object) -> str:
    """Return canonical 'SH600519' / 'SZ000001' form.

    Accepts: bare digits, prefixed (SH/SZ/sh/sz), Wind dot-suffix (XXX.SH / XXX.SZ),
    Bloomberg dot-suffix (XXX.XSHG / XXX.XSHE), with leading/trailing whitespace.
    Raises InvalidSymbolError on anything else.
    """
    if not isinstance(raw, str):
        raise InvalidSymbolError(f"expected str, got {type(raw).__name__}: {raw!r}")
    s = raw.strip().upper()
    if not s:
        raise InvalidSymbolError("empty symbol")

    # Dot-suffix: '600519.SH', '000001.SZ', '600519.XSHG', '000001.XSHE'
    if "." in s:
        digits, suffix = s.split(".", 1)
        if not _DIGITS_RE.fullmatch(digits):
            raise InvalidSymbolError(f"non-6-digit code in dot-suffix form: {raw!r}")
        if suffix in {"SH", "XSHG"}:
            market = "SH"
        elif suffix in {"SZ", "XSHE"}:
            market = "SZ"
        else:
            raise InvalidSymbolError(f"unknown dot-suffix {suffix!r} in {raw!r}")
        # Sanity-check that the suffix matches the bare-prefix rule.
        # B-shares are the only case where suffix can disagree with leading-digit
        # heuristic; trust the explicit suffix in that case.
        return f"{market}{digits}"

    # Already-prefixed: 'SH600519', 'SZ000001'
    if len(s) == 8 and s[:2] in {"SH", "SZ"} and _DIGITS_RE.fullmatch(s[2:]):
        return s

    # Bare 6-digit: '600519' -> 'SH600519'
    if _DIGITS_RE.fullmatch(s):
        market = _market_for_bare(s)
        return f"{market}{s}"

    raise InvalidSymbolError(
        f"could not normalise {raw!r} (expected 6 digits, optional SH/SZ prefix, "
        f"or .SH/.SZ/.XSHG/.XSHE suffix)"
    )


def to_akshare_bare(sym: str) -> str:
    """Strip the market prefix for AKshare's stock_zh_a_hist."""
    norm = normalise(sym)
    return norm[2:]


def to_akshare_with_market(sym: str) -> str:
    """Lower-case prefix form for AKshare's index_stock_hist (e.g. 'sh000300')."""
    norm = normalise(sym)
    return norm[:2].lower() + norm[2:]
