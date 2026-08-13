"""Typed config loaders. Unknown keys are errors (typo protection).

- ``configs/defaults.toml`` → :class:`Defaults` (project, memory, slices, splits)
- ``configs/costs.yaml``    → :class:`CostsConfig` (per-product fees/ticks/multipliers)

Every stage's manifest ``config_snapshot`` is the resolved dataclass dict plus
that stage's CLI args; ``futures_common.manifest.config_hash`` hashes it.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from futures_common.paths import repo_root

DEFAULT_CONFIG_DIR = repo_root() / "configs"


class ConfigError(ValueError):
    """Malformed or unknown-key configuration."""


def _require_keys(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in {where}")


def _parse_date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise ConfigError(f"{where}: expected a date, got {value!r}")


# ---------------------------------------------------------------- defaults.toml


@dataclass(frozen=True, slots=True)
class ProjectCfg:
    seed: int = 42
    timezone: str = "Asia/Shanghai"


@dataclass(frozen=True, slots=True)
class MemoryCfg:
    default_budget_gb: float = 4.0
    kill_factor: float = 1.25
    stage_budget_gb: Mapping[str, float] = field(default_factory=dict)

    def budget_for(self, stage: str) -> float:
        return float(self.stage_budget_gb.get(stage, self.default_budget_gb))


@dataclass(frozen=True, slots=True)
class SliceCfg:
    products: tuple[str, ...]
    start: dt.date
    end: dt.date

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(range(self.start.year, self.end.year + 1))


@dataclass(frozen=True, slots=True)
class SplitsCfg:
    train: tuple[dt.date, dt.date]
    validation: tuple[dt.date, dt.date]
    holdout: tuple[dt.date, dt.date]


@dataclass(frozen=True, slots=True)
class Defaults:
    project: ProjectCfg
    memory: MemoryCfg
    slices: Mapping[str, SliceCfg]
    splits: SplitsCfg


def load_defaults(path: Path | None = None) -> Defaults:
    path = path or (DEFAULT_CONFIG_DIR / "defaults.toml")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    _require_keys(raw, {"project", "memory", "slices", "splits"}, str(path))

    proj_raw = raw.get("project", {})
    _require_keys(proj_raw, {"seed", "timezone"}, "project")
    project = ProjectCfg(
        seed=int(proj_raw.get("seed", 42)),
        timezone=str(proj_raw.get("timezone", "Asia/Shanghai")),
    )

    mem_raw = dict(raw.get("memory", {}))
    stage_budgets = {str(k): float(v) for k, v in mem_raw.pop("stage_budget_gb", {}).items()}
    _require_keys(mem_raw, {"default_budget_gb", "kill_factor"}, "memory")
    memory = MemoryCfg(
        default_budget_gb=float(mem_raw.get("default_budget_gb", 4.0)),
        kill_factor=float(mem_raw.get("kill_factor", 1.25)),
        stage_budget_gb=stage_budgets,
    )

    slices: dict[str, SliceCfg] = {}
    for name, section in raw.get("slices", {}).items():
        _require_keys(section, {"products", "start", "end"}, f"slices.{name}")
        slices[name] = SliceCfg(
            products=tuple(str(p).upper() for p in section["products"]),
            start=_parse_date(section["start"], f"slices.{name}.start"),
            end=_parse_date(section["end"], f"slices.{name}.end"),
        )

    splits_raw = raw.get("splits", {})
    _require_keys(splits_raw, {"train", "validation", "holdout"}, "splits")

    def _span(key: str) -> tuple[dt.date, dt.date]:
        pair = splits_raw[key]
        if len(pair) != 2:
            raise ConfigError(f"splits.{key}: expected [start, end]")
        lo = _parse_date(pair[0], f"splits.{key}[0]")
        hi = _parse_date(pair[1], f"splits.{key}[1]")
        if lo > hi:
            raise ConfigError(f"splits.{key}: start after end")
        return lo, hi

    splits = SplitsCfg(
        train=_span("train"), validation=_span("validation"), holdout=_span("holdout")
    )
    return Defaults(project=project, memory=memory, slices=slices, splits=splits)


# ---------------------------------------------------------------- costs.yaml


@dataclass(frozen=True, slots=True)
class ProductCosts:
    product: str
    exchange: str | None
    multiplier: float | None
    tick_size: float | None
    fee_bps_per_side: float
    slippage_ticks: int
    declared: bool  # False → fell back to defaults; backtests must refuse until verified


@dataclass(frozen=True, slots=True)
class InferenceCfg:
    multiplier_rel_tol: float = 0.05
    min_bars: int = 10_000


@dataclass(frozen=True, slots=True)
class CostsConfig:
    default_fee_bps_per_side: float
    default_slippage_ticks: int
    inference: InferenceCfg
    products: Mapping[str, ProductCosts]

    def for_product(self, code: str) -> ProductCosts:
        code = code.upper()
        if code in self.products:
            return self.products[code]
        return ProductCosts(
            product=code,
            exchange=None,
            multiplier=None,
            tick_size=None,
            fee_bps_per_side=self.default_fee_bps_per_side,
            slippage_ticks=self.default_slippage_ticks,
            declared=False,
        )


def load_costs(path: Path | None = None) -> CostsConfig:
    import yaml

    path = path or (DEFAULT_CONFIG_DIR / "costs.yaml")
    raw = yaml.safe_load(path.read_text())
    _require_keys(raw, {"version", "defaults", "inference", "products"}, str(path))

    defaults_raw = raw.get("defaults", {})
    _require_keys(defaults_raw, {"fee_bps_per_side", "slippage_ticks"}, "defaults")
    fee_default = float(defaults_raw.get("fee_bps_per_side", 1.5))
    slip_default = int(defaults_raw.get("slippage_ticks", 1))

    inf_raw = raw.get("inference", {})
    _require_keys(inf_raw, {"multiplier_rel_tol", "min_bars", "tick_method"}, "inference")
    inference = InferenceCfg(
        multiplier_rel_tol=float(inf_raw.get("multiplier_rel_tol", 0.05)),
        min_bars=int(inf_raw.get("min_bars", 10_000)),
    )

    products: dict[str, ProductCosts] = {}
    for code, section in (raw.get("products") or {}).items():
        code = str(code).upper()
        _require_keys(
            section,
            {"exchange", "multiplier", "tick_size", "fee_bps_per_side", "slippage_ticks"},
            f"products.{code}",
        )
        products[code] = ProductCosts(
            product=code,
            exchange=section.get("exchange"),
            multiplier=float(section["multiplier"]) if "multiplier" in section else None,
            tick_size=float(section["tick_size"]) if "tick_size" in section else None,
            fee_bps_per_side=float(section.get("fee_bps_per_side", fee_default)),
            slippage_ticks=int(section.get("slippage_ticks", slip_default)),
            declared=True,
        )

    return CostsConfig(
        default_fee_bps_per_side=fee_default,
        default_slippage_ticks=slip_default,
        inference=inference,
        products=products,
    )


# ---------------------------------------------------------------- generic yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a yaml config file as a plain dict (band-specific schemas validate downstream)."""
    import yaml

    out = yaml.safe_load(path.read_text())
    if not isinstance(out, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    return out
