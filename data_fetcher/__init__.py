"""Futures data band: L0 ingest → L1 calendar/continuous/universe → L2 panel.

Reads ONLY the local raw trees under ``Future_minute_data/`` (read-only):

- ``2005-202506/EXCHANGE/PRODUCT/XXYYMM.csv`` — one file per real contract
- ``2026/YYYYMM/YYYYMMDD/*.csv``              — one folder per calendar day

and writes layered artifacts under ``~/AutoLLM_data/futures/`` (see
``futures_common.paths``). No network I/O anywhere in this package — the old
AKshare A-share fetcher lives on, untouched, in ``data_fetcher_akshare/``.
"""

__version__ = "0.1.0"
