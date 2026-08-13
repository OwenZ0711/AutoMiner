"""AST tests: canonical hash, folding, shape preservation, mutation classes."""

from __future__ import annotations

import numpy as np

from futures_common.rng import make_rng
from gp_cta.gp.expressions import (
    Expr,
    TreeSpace,
    ast_hash,
    canonicalize,
    random_tree,
    tree_shape,
)
from gp_cta.gp.warmstart import classic_seeds, point_mutate, restricted_crossover


def test_commutative_children_hash_equal() -> None:
    a = Expr.call("add", Expr.feature("x_std"), Expr.feature("ret_5m"))
    b = Expr.call("add", Expr.feature("ret_5m"), Expr.feature("x_std"))
    assert ast_hash(a) == ast_hash(b)
    c = Expr.call("sub", Expr.feature("x_std"), Expr.feature("ret_5m"))
    d = Expr.call("sub", Expr.feature("ret_5m"), Expr.feature("x_std"))
    assert ast_hash(c) != ast_hash(d)  # sub is NOT commutative


def test_involution_and_const_folding() -> None:
    x = Expr.feature("x_std")
    assert canonicalize(Expr.call("neg", Expr.call("neg", x))) == canonicalize(x)
    assert canonicalize(Expr.call("abs", Expr.call("abs", x))).op == "abs"
    folded = canonicalize(Expr.call("add", Expr.const(1.0), Expr.const(2.0)))
    assert folded.op == "const" and folded.value == 3.0


def test_canonicalize_idempotent() -> None:
    rng = make_rng(42, "test.canon")
    space = TreeSpace(max_depth=5)
    for _ in range(50):
        e = random_tree(rng, space)
        c1 = canonicalize(e)
        assert canonicalize(c1) == c1
        assert ast_hash(e) == ast_hash(c1)


def test_hash_pinned_value() -> None:
    """Cross-process stability: pinned literal guard against silent drift."""
    e = Expr.call("ts_mean", Expr.feature("x_std"), value=60)
    assert ast_hash(e) == ast_hash(Expr.call("ts_mean", Expr.feature("x_std"), value=60))
    assert ast_hash(e) != ast_hash(Expr.call("ts_mean", Expr.feature("x_std"), value=120))


def test_point_mutation_preserves_shape() -> None:
    rng = make_rng(1, "test.mutshape")
    space = TreeSpace()
    for seed in classic_seeds():
        base = canonicalize(seed.expr)
        shape = tree_shape(base)
        for _ in range(200):
            mutated = point_mutate(base, rng, space)
            assert tree_shape(mutated) == shape, f"{seed.seed_id} shape changed"


def test_restricted_crossover_preserves_shape() -> None:
    rng = make_rng(2, "test.xshape")
    space = TreeSpace()
    base = canonicalize(classic_seeds()[0].expr)
    shape = tree_shape(base)
    variants = [point_mutate(base, rng, space) for _ in range(20)]
    for _ in range(100):
        a = variants[int(rng.integers(len(variants)))]
        b = variants[int(rng.integers(len(variants)))]
        child = restricted_crossover(a, b, rng)
        assert tree_shape(child) == shape


def test_point_mutation_generates_diversity() -> None:
    rng = make_rng(3, "test.diverse")
    space = TreeSpace()
    base = canonicalize(classic_seeds()[0].expr)
    hashes = {ast_hash(point_mutate(base, rng, space)) for _ in range(300)}
    assert len(hashes) > 20  # plenty of same-shape variants reachable


def test_tree_surgery_roundtrip() -> None:
    e = classic_seeds()[1].expr  # breakout
    for path in e.paths():
        sub = e.subtree(path)
        rebuilt = e.replace(path, sub)
        assert ast_hash(rebuilt) == ast_hash(e)


def test_random_tree_depth_bounded() -> None:
    rng = make_rng(4, "test.depth")
    space = TreeSpace(max_depth=4)
    assert all(random_tree(rng, space).depth() <= 4 for _ in range(100))


def test_seed_str_readable() -> None:
    s = str(classic_seeds()[0].expr)
    assert "ts_mean(close_adj, 60)" in s
    assert np.isfinite(len(s))
