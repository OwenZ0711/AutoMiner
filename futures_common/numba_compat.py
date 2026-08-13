"""numba import shim: ``njit`` falls back to a no-op decorator without numba.

Every JIT kernel in the pipeline must behave identically under the fallback
(pure-NumPy/Python semantics) — tests run both ways.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _fallback_njit(*args: Any, **kwargs: Any) -> Any:
    if args and callable(args[0]):
        return args[0]

    def deco(f: F) -> F:
        return f

    return deco


try:  # pragma: no cover - environment-dependent
    from numba import njit as _numba_njit

    njit: Any = _numba_njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    njit = _fallback_njit
    HAVE_NUMBA = False


def njit_kernel(**kwargs: Any) -> Callable[[F], F]:
    """Signature-preserving njit wrapper (keeps mypy --strict happy)."""

    def deco(f: F) -> F:
        if HAVE_NUMBA:
            jitted: F = njit(**kwargs)(f)
            return jitted
        return f

    return deco


__all__ = ["HAVE_NUMBA", "njit", "njit_kernel"]
