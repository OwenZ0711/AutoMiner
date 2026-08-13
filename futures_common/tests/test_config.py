"""Config loader tests: strictness, hash stability, costs semantics."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from futures_common.config import ConfigError, load_costs, load_defaults
from futures_common.manifest import config_hash

MINIMAL_TOML = """
[project]
seed = 7

[memory]
default_budget_gb = 2.0
[memory.stage_budget_gb]
ingest = 1.0

[slices.tiny]
products = ["rb", "i"]
start = "2020-01-01"
end = "2020-12-31"

[splits]
train = ["2016-01-01", "2022-12-31"]
validation = ["2023-01-01", "2024-06-30"]
holdout = ["2024-07-01", "2026-02-28"]
"""


def test_load_defaults_minimal(tmp_path: Path) -> None:
    p = tmp_path / "defaults.toml"
    p.write_text(MINIMAL_TOML)
    d = load_defaults(p)
    assert d.project.seed == 7
    assert d.memory.budget_for("ingest") == 1.0
    assert d.memory.budget_for("unknown-stage") == 2.0
    assert d.slices["tiny"].products == ("RB", "I")  # upper-cased
    assert d.slices["tiny"].years == (2020,)
    assert d.splits.holdout == (dt.date(2024, 7, 1), dt.date(2026, 2, 28))


def test_load_defaults_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "defaults.toml"
    p.write_text(MINIMAL_TOML + "\n[project2]\nx = 1\n")
    with pytest.raises(ConfigError, match="project2"):
        load_defaults(p)


def test_load_defaults_rejects_typo_in_section(tmp_path: Path) -> None:
    p = tmp_path / "defaults.toml"
    p.write_text(MINIMAL_TOML.replace("seed = 7", "seed = 7\nsede = 8"))
    with pytest.raises(ConfigError, match="sede"):
        load_defaults(p)


def test_repo_defaults_toml_loads() -> None:
    d = load_defaults()
    assert "acceptance" in d.slices
    assert d.slices["acceptance"].products == ("RB", "I", "TA")
    assert d.memory.budget_for("panel") == 4.0


def test_repo_costs_yaml_loads_and_fallback() -> None:
    c = load_costs()
    rb = c.for_product("rb")
    assert rb.declared and rb.multiplier == 10.0 and rb.tick_size == 1.0
    unknown = c.for_product("ZZ")
    assert not unknown.declared
    assert unknown.fee_bps_per_side == c.default_fee_bps_per_side


def test_config_hash_stable_across_key_order() -> None:
    a = {"b": 1, "a": [2, 3], "d": {"y": dt.date(2020, 1, 1), "x": None}}
    b = {"d": {"x": None, "y": dt.date(2020, 1, 1)}, "a": [2, 3], "b": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_sensitive_to_values() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})
