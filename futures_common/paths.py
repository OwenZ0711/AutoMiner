"""Storage-path conventions — the ONLY source of on-disk layout knowledge.

Both bands construct every path through :class:`FuturesPaths`. Conventions:

- hive-style ``key=value`` partition directories;
- products/contracts always UPPERCASE in paths;
- parquet everywhere, except panel channels/features which are raw
  ``np.memmap`` files (``.f32`` / ``.u8``) with a sibling ``meta.json``;
- pipeline data lives under ``~/AutoLLM_data/futures`` (outside the repo),
  run reports under ``<repo>/results`` (git-ignored).
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_ROOT = Path("~/AutoLLM_data/futures").expanduser()
DEFAULT_RESULTS_ROOT = _REPO_ROOT / "results"
DEFAULT_RAW_HISTORICAL = _REPO_ROOT / "Future_minute_data" / "2005-202506"
DEFAULT_RAW_2026 = _REPO_ROOT / "Future_minute_data" / "2026"


@dataclass(frozen=True, slots=True)
class FuturesPaths:
    """All on-disk locations for the futures pipeline."""

    data_root: Path
    results_root: Path

    # ---------- L0 raw ingest ----------

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    def raw_product_dir(self, exchange: str, product: str) -> Path:
        return self.raw_root / f"exchange={exchange.upper()}" / f"product={product.upper()}"

    def raw_contract(self, exchange: str, product: str, contract: str) -> Path:
        return self.raw_product_dir(exchange, product) / f"{contract.upper()}.parquet"

    @property
    def raw_manifest(self) -> Path:
        return self.raw_root / "_manifest.parquet"

    # ---------- L1a calendar ----------

    @property
    def calendar_dir(self) -> Path:
        return self.data_root / "calendar"

    @property
    def session_templates(self) -> Path:
        return self.calendar_dir / "session_templates.parquet"

    @property
    def product_sessions(self) -> Path:
        return self.calendar_dir / "product_sessions.parquet"

    @property
    def trading_days(self) -> Path:
        return self.calendar_dir / "trading_days.parquet"

    @property
    def trading_days_bootstrap(self) -> Path:
        return self.calendar_dir / "trading_days_bootstrap.parquet"

    @property
    def night_exceptions(self) -> Path:
        return self.calendar_dir / "night_exceptions.parquet"

    # ---------- L1b continuous ----------

    @property
    def continuous_dir(self) -> Path:
        return self.data_root / "continuous"

    def daily_stats(self, product: str) -> Path:
        return self.continuous_dir / "daily_stats" / f"product={product.upper()}.parquet"

    def continuous_year(self, product: str, year: int) -> Path:
        return self.continuous_dir / f"product={product.upper()}" / f"year={year}.parquet"

    @property
    def roll_calendar(self) -> Path:
        return self.continuous_dir / "roll_calendar.parquet"

    # ---------- L1c universe ----------

    @property
    def universe_dir(self) -> Path:
        return self.data_root / "universe"

    @property
    def universe_products(self) -> Path:
        return self.universe_dir / "products.parquet"

    @property
    def era_flags(self) -> Path:
        return self.universe_dir / "era_flags.parquet"

    @property
    def liquidity(self) -> Path:
        return self.universe_dir / "liquidity.parquet"

    @property
    def volume_break_check(self) -> Path:
        return self.universe_dir / "volume_break_check.parquet"

    # ---------- L2 panel (memmaps + index) ----------

    def panel_dir(self, year: int) -> Path:
        return self.data_root / "panel" / f"year={year}"

    def panel_meta(self, year: int) -> Path:
        return self.panel_dir(year) / "meta.json"

    def panel_channel(self, year: int, name: str) -> Path:
        return self.panel_dir(year) / "channels" / f"{name}.f32"

    def panel_mask(self, year: int, name: str) -> Path:
        return self.panel_dir(year) / "masks" / f"{name}.u8"

    def panel_feature(self, year: int, name: str) -> Path:
        return self.panel_dir(year) / "features" / f"{name}.f32"

    def bob_index(self, year: int) -> Path:
        return self.panel_dir(year) / "bob_index.parquet"

    # ---------- L3/L4 lead-lag ----------

    @property
    def edges_dir(self) -> Path:
        return self.data_root / "edges"

    def edges_file(self, asof: dt.date, config_hash: str) -> Path:
        return self.edges_dir / f"asof={asof.isoformat()}_cfg={config_hash[:12]}.parquet"

    def leadlag_scan_dir(self, config_hash: str) -> Path:
        return self.data_root / "leadlag" / "scans" / config_hash[:12]

    # ---------- GP cache ----------

    def gp_cache_dir(self, year: int) -> Path:
        return self.data_root / "gp_cache" / f"year={year}"

    # ---------- results (in repo, git-ignored) ----------

    def run_dir(self, stage: str, run_id: str) -> Path:
        return self.results_root / "runs" / stage / run_id

    @property
    def latest_runs(self) -> Path:
        return self.results_root / "runs" / "latest.json"

    @property
    def trial_ledger_dir(self) -> Path:
        return self.results_root / "trial_ledger"

    @property
    def pool_dir(self) -> Path:
        return self.results_root / "pool"

    def holdout_marker(self, round_id: str) -> Path:
        return self.results_root / "holdout_scored" / f"{round_id}.marker"


def default_paths(
    data_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> FuturesPaths:
    """Build paths with env-var override ``AUTOLLM_DATA_ROOT`` for the data root."""
    if data_root is None:
        data_root = os.environ.get("AUTOLLM_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    if results_root is None:
        results_root = DEFAULT_RESULTS_ROOT
    return FuturesPaths(
        data_root=Path(data_root).expanduser(),
        results_root=Path(results_root).expanduser(),
    )


def repo_root() -> Path:
    return _REPO_ROOT
