"""Expression trees for GP-generated trading signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl


EPS = 1e-12
MAX_ABS_VALUE = 1e6

BINARY_OPS = ("add", "sub", "mul", "div")
UNARY_OPS = ("abs", "log")
TS_OPS = ("ts_mean", "ts_std", "ts_min", "ts_max", "ts_delta")
ALL_OPS = BINARY_OPS + UNARY_OPS + TS_OPS
DEFAULT_WINDOWS = (3, 5, 10, 20)
DEFAULT_CONSTANTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Expr:
    """A small immutable expression tree."""

    op: str
    children: tuple["Expr", ...] = ()
    value: str | float | int | None = None

    @staticmethod
    def feature(name: str) -> "Expr":
        return Expr("feature", value=name)

    @staticmethod
    def const(value: float) -> "Expr":
        return Expr("const", value=float(value))

    @staticmethod
    def call(op: str, *children: "Expr", value: int | None = None) -> "Expr":
        return Expr(op=op, children=tuple(children), value=value)

    def evaluate(self, frame: pl.DataFrame) -> np.ndarray:
        """Evaluate the expression against one symbol's feature frame."""
        if self.op == "feature":
            if not isinstance(self.value, str):
                raise ValueError("feature expression missing feature name")
            return _clean(frame[self.value].to_numpy())
        if self.op == "const":
            return np.full(frame.height, float(self.value), dtype=float)

        args = [child.evaluate(frame) for child in self.children]
        if self.op == "add":
            return _clean(args[0] + args[1])
        if self.op == "sub":
            return _clean(args[0] - args[1])
        if self.op == "mul":
            return _clean(args[0] * args[1])
        if self.op == "div":
            denom = np.where(np.abs(args[1]) < EPS, np.nan, args[1])
            return _clean(args[0] / denom)
        if self.op == "abs":
            return _clean(np.abs(args[0]))
        if self.op == "log":
            return _clean(np.log(np.abs(args[0]) + EPS))
        if self.op in TS_OPS:
            window = _window(self.value)
            return _clean(_rolling(self.op, args[0], window))
        raise ValueError(f"unknown expression op: {self.op!r}")

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def paths(self) -> tuple[tuple[int, ...], ...]:
        out: list[tuple[int, ...]] = [()]
        for i, child in enumerate(self.children):
            out.extend((i, *path) for path in child.paths())
        return tuple(out)

    def subtree(self, path: tuple[int, ...]) -> "Expr":
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def replace(self, path: tuple[int, ...], new_subtree: "Expr") -> "Expr":
        if not path:
            return new_subtree
        idx = path[0]
        children = list(self.children)
        children[idx] = children[idx].replace(path[1:], new_subtree)
        return Expr(self.op, tuple(children), self.value)

    def __str__(self) -> str:
        if self.op == "feature":
            return str(self.value)
        if self.op == "const":
            return f"{float(self.value):.3g}"
        if self.op in TS_OPS:
            return f"{self.op}({self.children[0]}, {self.value})"
        return f"{self.op}({', '.join(str(child) for child in self.children)})"


def random_expr(
    rng: np.random.Generator,
    *,
    features: Iterable[str],
    max_depth: int,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    constants: Iterable[float] = DEFAULT_CONSTANTS,
    terminal_probability: float = 0.25,
) -> Expr:
    """Generate a random expression tree."""
    features = tuple(features)
    windows = tuple(windows)
    constants = tuple(constants)
    if max_depth <= 1 or rng.random() < terminal_probability:
        if rng.random() < 0.85:
            return Expr.feature(str(rng.choice(features)))
        return Expr.const(float(rng.choice(constants)))

    op = str(rng.choice(ALL_OPS))
    if op in BINARY_OPS:
        return Expr.call(
            op,
            random_expr(rng, features=features, max_depth=max_depth - 1),
            random_expr(rng, features=features, max_depth=max_depth - 1),
        )
    if op in UNARY_OPS:
        return Expr.call(
            op, random_expr(rng, features=features, max_depth=max_depth - 1)
        )
    return Expr.call(
        op,
        random_expr(rng, features=features, max_depth=max_depth - 1),
        value=int(rng.choice(windows)),
    )


def _window(value: str | float | int | None) -> int:
    window = int(value) if value is not None else 1
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    return window


def _rolling(op: str, values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if op == "ts_mean":
        return series.rolling(window=window, min_periods=window).mean().to_numpy()
    if op == "ts_std":
        return (
            series.rolling(window=window, min_periods=window)
            .std(ddof=0)
            .fillna(0.0)
            .to_numpy()
        )
    if op == "ts_min":
        return series.rolling(window=window, min_periods=window).min().to_numpy()
    if op == "ts_max":
        return series.rolling(window=window, min_periods=window).max().to_numpy()
    if op == "ts_delta":
        return (series - series.shift(window)).to_numpy()
    raise ValueError(f"unknown rolling op: {op}")


def _clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -MAX_ABS_VALUE, MAX_ABS_VALUE)
