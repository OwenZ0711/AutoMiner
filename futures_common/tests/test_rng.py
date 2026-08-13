"""Named-stream RNG: reproducible per (seed, stream), independent across streams."""

from __future__ import annotations

import numpy as np

from futures_common.rng import make_rng


def test_same_seed_same_stream_reproducible() -> None:
    a = make_rng(42, "gp.seed3").random(8)
    b = make_rng(42, "gp.seed3").random(8)
    np.testing.assert_array_equal(a, b)


def test_different_streams_independent() -> None:
    a = make_rng(42, "gp.seed3").random(8)
    b = make_rng(42, "leadlag.null").random(8)
    assert not np.array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = make_rng(42, "x").random(8)
    b = make_rng(43, "x").random(8)
    assert not np.array_equal(a, b)
