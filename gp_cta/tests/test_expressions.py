from __future__ import annotations

import numpy as np
import polars as pl

from gp_cta.expressions import EPS, Expr


def test_protected_division_and_log_are_finite() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0], "zero": [0.0, 0.0, 0.0]})
    expr = Expr.call(
        "log",
        Expr.call("div", Expr.feature("x"), Expr.feature("zero")),
    )

    values = expr.evaluate(frame)

    assert np.isfinite(values).all()
    assert np.allclose(values, np.log(EPS))


def test_timeseries_delta_uses_past_values_only() -> None:
    frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 16.0]})
    expr = Expr.call("ts_delta", Expr.feature("x"), value=2)

    values = expr.evaluate(frame)

    assert values.tolist() == [0.0, 0.0, 3.0, 5.0]


def test_expression_string_is_stable() -> None:
    expr = Expr.call(
        "sub",
        Expr.call("ts_mean", Expr.feature("close"), value=5),
        Expr.feature("vwap"),
    )

    assert str(expr) == "sub(ts_mean(close, 5), vwap)"
