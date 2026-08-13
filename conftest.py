"""Root pytest config: the --realdata gate shared by every test package."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--realdata",
        action="store_true",
        default=False,
        help="run tests that read Future_minute_data/ or the built acceptance slice",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--realdata"):
        return
    skip = pytest.mark.skip(reason="needs --realdata")
    for item in items:
        if "realdata" in item.keywords:
            item.add_marker(skip)
