"""Typed expression AST for GP formulas.

One value type flows through the tree — SERIES, a float32 (T, N) panel.
Non-series slots are typed literals drawn from closed domains; only point
mutation may touch them:

    WINDOW  {5, 15, 30, 60, 120, 240, 480, 1200} bars
    CONST   {-2, -1, -0.5, 0.5, 1, 2}
    RANK    {1, 2, 3}          (cross-asset terminal leader rank)
    REF_K   {1, 5, 15}         (ref lookback)
    FEATURE the panel feature names

MUTATION-CLASS RULE: a node may only be rewritten within its class
``(arity, slot_type)`` — ``ts_mean ↔ ema`` (arity-1 + WINDOW), ``add ↔ div``
(arity-2, no slot), ``feature ↔ feature``, ``lead_ret ↔ ll_strength``
(arity-0 + RANK). Tree SHAPE therefore never changes under point mutation,
which is exactly the warm-start paper's frozen-structure constraint.

Identity: ``ast_hash`` = blake2b of the canonical S-expression after cheap
rewrites (commutative child sort, involution folding, const folding). Stable
across processes (no Python ``hash()``); used for population dedup, the
fitness memo, the trial ledger, and the candidate pool.

Seed-authored literals may sit OUTSIDE the closed domains (e.g. window=1 in a
delta, const=15.49); mutation snaps them back into the domains.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

WINDOWS: Final = (5, 15, 30, 60, 120, 240, 480, 1200)
CONSTANTS: Final = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
RANKS: Final = (1, 2, 3)
REF_KS: Final = (1, 5, 15)

DEFAULT_FEATURES: Final = (
    "x_std",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "ret_1d",
    "vol_ewma",
    "z_60",
    "z_240",
    "z_1200",
    "oi_chg_1d",
    "vol_ratio",
    "range_1",
    "range_15",
    "range_60",
    "close_adj",
)

CROSS_TERMINALS: Final = ("lead_ret", "lead_oi_chg", "lead_stale", "ll_strength")


@dataclass(frozen=True, slots=True)
class OpSpec:
    arity: int
    slot: str | None  # None | "window" | "const" | "rank" | "feature" | "ref_k"
    commutative: bool = False


OPS: Final[dict[str, OpSpec]] = {
    # terminals
    "feature": OpSpec(0, "feature"),
    "const": OpSpec(0, "const"),
    "lead_ret": OpSpec(0, "rank"),
    "lead_oi_chg": OpSpec(0, "rank"),
    "lead_stale": OpSpec(0, "rank"),
    "ll_strength": OpSpec(0, "rank"),
    # unary
    "neg": OpSpec(1, None),
    "abs": OpSpec(1, None),
    "log1p": OpSpec(1, None),
    "sign": OpSpec(1, None),
    "ref": OpSpec(1, "ref_k"),
    # binary
    "add": OpSpec(2, None, commutative=True),
    "sub": OpSpec(2, None),
    "mul": OpSpec(2, None, commutative=True),
    "div": OpSpec(2, None),
    "min2": OpSpec(2, None, commutative=True),
    "max2": OpSpec(2, None, commutative=True),
    "gt": OpSpec(2, None),
    "lt": OpSpec(2, None),
    # ternary
    "if_then_else": OpSpec(3, None),
    # time-series (window slot)
    "ts_mean": OpSpec(1, "window"),
    "ts_std": OpSpec(1, "window"),
    "ts_min": OpSpec(1, "window"),
    "ts_max": OpSpec(1, "window"),
    "ts_delta": OpSpec(1, "window"),
    "ts_sum": OpSpec(1, "window"),
    "ts_rank": OpSpec(1, "window"),
    "ts_zscore": OpSpec(1, "window"),
    "ema": OpSpec(1, "window"),
    "decay_linear": OpSpec(1, "window"),
    "ts_corr": OpSpec(2, "window"),
}

SLOT_DOMAINS: Final[dict[str, tuple[float | int | str, ...]]] = {
    "window": WINDOWS,
    "const": CONSTANTS,
    "rank": RANKS,
    "ref_k": REF_KS,
    "feature": DEFAULT_FEATURES,
}


@dataclass(frozen=True, slots=True)
class Expr:
    """Immutable expression node (ported tree-surgery from the old baseline)."""

    op: str
    children: tuple[Expr, ...] = ()
    value: str | float | int | None = None  # the literal slot

    def __post_init__(self) -> None:
        spec = OPS.get(self.op)
        if spec is None:
            raise ValueError(f"unknown op {self.op!r}")
        if len(self.children) != spec.arity:
            raise ValueError(f"{self.op}: arity {spec.arity}, got {len(self.children)}")

    # ---------- constructors ----------

    @staticmethod
    def feature(name: str) -> Expr:
        return Expr("feature", value=name)

    @staticmethod
    def const(value: float) -> Expr:
        return Expr("const", value=float(value))

    @staticmethod
    def call(op: str, *children: Expr, value: float | int | str | None = None) -> Expr:
        return Expr(op, tuple(children), value)

    # ---------- structure ----------

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def paths(self) -> tuple[tuple[int, ...], ...]:
        out: list[tuple[int, ...]] = [()]
        for i, child in enumerate(self.children):
            out.extend((i, *p) for p in child.paths())
        return tuple(out)

    def subtree(self, path: tuple[int, ...]) -> Expr:
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def replace(self, path: tuple[int, ...], new_subtree: Expr) -> Expr:
        if not path:
            return new_subtree
        children = list(self.children)
        children[path[0]] = children[path[0]].replace(path[1:], new_subtree)
        return Expr(self.op, tuple(children), self.value)

    def mutation_class(self) -> tuple[int, str | None]:
        spec = OPS[self.op]
        return (spec.arity, spec.slot)

    # ---------- display ----------

    def __str__(self) -> str:
        if self.op == "feature":
            return str(self.value)
        if self.op == "const":
            return f"{float(self.value):.4g}"  # type: ignore[arg-type]
        if self.value is not None:
            inner = ", ".join(str(c) for c in self.children)
            sep = ", " if inner else ""
            return f"{self.op}({inner}{sep}{self.value})"
        return f"{self.op}({', '.join(str(c) for c in self.children)})"


def tree_shape(e: Expr) -> tuple[tuple[tuple[int, ...], int, str | None], ...]:
    """Positions + arities + slot types — the FROZEN structure of a seed."""
    return tuple((p, OPS[e.subtree(p).op].arity, OPS[e.subtree(p).op].slot) for p in e.paths())


# ---------------------------------------------------------- canonicalization


def canonicalize(e: Expr) -> Expr:
    children = tuple(canonicalize(c) for c in e.children)
    node = Expr(e.op, children, e.value)

    # const folding: any all-const subtree collapses to a literal
    if children and all(c.op == "const" for c in children) and e.op in _FOLDABLE:
        vals = [float(c.value) for c in children]  # type: ignore[arg-type]
        folded = _FOLDABLE[e.op](vals)
        if folded is not None and np.isfinite(folded):
            return Expr.const(float(folded))

    # involution / composition folding
    if e.op == "neg" and children[0].op == "neg":
        return children[0].children[0]
    if e.op in ("abs", "sign") and children[0].op == e.op:
        return children[0]
    if e.op == "abs" and children[0].op == "neg":
        return canonicalize(Expr("abs", children[0].children))

    # commutative child ordering by canonical key
    if OPS[e.op].commutative and len(children) == 2:
        a, b = children
        if canonical_key(a) > canonical_key(b):
            node = Expr(e.op, (b, a), e.value)
    return node


_FOLDABLE: dict[str, Callable[[list[float]], float | None]] = {
    "add": lambda v: v[0] + v[1],
    "sub": lambda v: v[0] - v[1],
    "mul": lambda v: v[0] * v[1],
    "div": lambda v: v[0] / v[1] if abs(v[1]) > 1e-12 else None,
    "neg": lambda v: -v[0],
    "abs": lambda v: abs(v[0]),
    "sign": lambda v: float(np.sign(v[0])),
    "log1p": lambda v: float(np.log1p(abs(v[0]))),
    "min2": lambda v: min(v),
    "max2": lambda v: max(v),
}


def canonical_key(e: Expr) -> str:
    if e.op == "feature":
        return f"(f {e.value})"
    if e.op == "const":
        return f"(c {float(e.value):.10g})"  # type: ignore[arg-type]
    parts = " ".join(canonical_key(c) for c in e.children)
    slot = "" if e.value is None else f" :{e.value}"
    return f"({e.op} {parts}{slot})"


def ast_hash(e: Expr) -> str:
    return hashlib.blake2b(canonical_key(canonicalize(e)).encode(), digest_size=16).hexdigest()


# --------------------------------------------------------------- random trees


@dataclass(frozen=True, slots=True)
class TreeSpace:
    """Sampling domains for random generation and point mutation."""

    features: tuple[str, ...] = DEFAULT_FEATURES
    windows: tuple[int, ...] = WINDOWS
    constants: tuple[float, ...] = CONSTANTS
    ranks: tuple[int, ...] = RANKS
    ref_ks: tuple[int, ...] = REF_KS
    cross_ok: bool = True
    max_depth: int = 6

    def domain(self, slot: str) -> tuple[float | int | str, ...]:
        out: dict[str, tuple[float | int | str, ...]] = {
            "window": self.windows,
            "const": self.constants,
            "rank": self.ranks,
            "ref_k": self.ref_ks,
            "feature": self.features,
        }
        return out[slot]

    def terminal_ops(self) -> tuple[str, ...]:
        return ("feature", "const", *(CROSS_TERMINALS if self.cross_ok else ()))

    def nonterminal_ops(self) -> tuple[str, ...]:
        return tuple(op for op, spec in OPS.items() if spec.arity > 0)

    def ops_in_class(self, arity: int, slot: str | None) -> tuple[str, ...]:
        out = tuple(op for op, spec in OPS.items() if spec.arity == arity and spec.slot == slot)
        if not self.cross_ok:
            out = tuple(op for op in out if op not in CROSS_TERMINALS)
        return out


def random_tree(rng: np.random.Generator, space: TreeSpace, max_depth: int | None = None) -> Expr:
    depth = space.max_depth if max_depth is None else max_depth
    return _random_node(rng, space, depth)


def _random_node(rng: np.random.Generator, space: TreeSpace, depth: int) -> Expr:
    if depth <= 1 or rng.random() < 0.3:
        op = str(rng.choice(space.terminal_ops()))
        return _with_random_slot(rng, space, op, ())
    op = str(rng.choice(space.nonterminal_ops()))
    spec = OPS[op]
    children = tuple(_random_node(rng, space, depth - 1) for _ in range(spec.arity))
    return _with_random_slot(rng, space, op, children)


def _with_random_slot(
    rng: np.random.Generator,
    space: TreeSpace,
    op: str,
    children: tuple[Expr, ...],
) -> Expr:
    slot = OPS[op].slot
    value = None if slot is None else _choice(rng, space.domain(slot))
    return Expr(op, children, value)


def _choice(rng: np.random.Generator, domain: Sequence[float | int | str]) -> float | int | str:
    return domain[int(rng.integers(len(domain)))]
