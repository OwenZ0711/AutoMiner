"""Seeded named-stream RNG factory.

Bare ``np.random.*`` module-level calls are banned in pipeline code (ruff
NPY002). Every randomness path takes ``seed: int`` (default 42) and derives an
independent generator per named stream so GP mutation, permutation nulls, and
test fixtures never share state:

    make_rng(42, "gp.seed3")
    make_rng(42, "leadlag.null")
"""

from __future__ import annotations

from zlib import crc32

import numpy as np


def make_rng(seed: int = 42, stream: str = "") -> np.random.Generator:
    """Independent, reproducible generator for (seed, stream)."""
    return np.random.default_rng(np.random.SeedSequence([seed, crc32(stream.encode("utf-8"))]))
