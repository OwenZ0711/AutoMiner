"""Frozen-structure warm-start GP (Ren–Qin–Li, arXiv:2412.00896).

The paper's method, exactly: the population starts as ONE validated seed
formula whose tree SHAPE is frozen for the whole run.

- generation 1: POINT MUTATIONS ONLY — swap a node's content within its
  mutation class (same arity + slot type) or its literal within the closed
  domain; positions and arities never change;
- generations ≥ 2: 50% restricted crossover (same-shape parents swap subtrees
  at IDENTICAL tree positions), 40% point mutation, 10% copy;
- tournament size 4, elitism 1, canonical-AST-hash duplicate rejection,
  early stop after `stagnation` generations without improvement.

Each seed is an independent small search (the paper's decorrelation
mechanism); the engine runs them sequentially (memory rule 6).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from futures_common.rng import make_rng
from gp_cta.gp.expressions import (
    OPS,
    Expr,
    TreeSpace,
    ast_hash,
    canonicalize,
    tree_shape,
)
from gp_cta.gp.fitness import FitnessReport
from gp_cta.leadlag.edges import Edge


@dataclass(frozen=True, slots=True)
class SeedSpec:
    seed_id: str
    expr: Expr
    edge: Edge | None = None


def classic_seeds() -> tuple[SeedSpec, ...]:
    """The proposal's seed library (exact ASTs; authored literals allowed)."""
    close = Expr.feature("close_adj")

    ma_cross = Expr.call(
        "sign",
        Expr.call(
            "sub", Expr.call("ts_mean", close, value=60), Expr.call("ts_mean", close, value=480)
        ),
    )
    breakout = Expr.call(
        "sub",
        Expr.call("gt", close, Expr.call("ts_max", Expr.call("ref", close, value=1), value=240)),
        Expr.call("lt", close, Expr.call("ts_min", Expr.call("ref", close, value=1), value=240)),
    )
    delta1 = Expr.call("ts_delta", close, value=1)  # authored window=1
    rsi_style = Expr.call(
        "sub",
        Expr.call(
            "div",
            Expr.call("ts_mean", Expr.call("max2", delta1, Expr.const(0.0)), value=240),
            Expr.call("ts_mean", Expr.call("abs", delta1), value=240),
        ),
        Expr.const(0.5),
    )
    vol_mom = Expr.call(
        "div",
        Expr.call("ts_delta", close, value=240),
        Expr.call(
            "mul",
            Expr.call("ts_std", Expr.feature("x_std"), value=240),
            Expr.const(15.4919),  # √240, authored
        ),
    )
    return (
        SeedSpec("ma_cross", ma_cross),
        SeedSpec("breakout", breakout),
        SeedSpec("rsi_style", rsi_style),
        SeedSpec("vol_momentum", vol_mom),
    )


def seed_from_edge(edge: Edge) -> SeedSpec:
    """One seed per validated lead-lag edge: sign(LEAD_RET(1)) · gt(LL_STRENGTH(1), 0.02)."""
    expr = Expr.call(
        "mul",
        Expr.call("sign", Expr.call("lead_ret", value=1)),
        Expr.call("gt", Expr.call("ll_strength", value=1), Expr.const(0.02)),
    )
    return SeedSpec(f"edge_{edge.leader}_{edge.follower}", expr, edge=edge)


# --------------------------------------------------------------- GP operators


def point_mutate(e: Expr, rng: np.random.Generator, space: TreeSpace) -> Expr:
    """Rewrite ONE node within its mutation class; the tree shape is invariant."""
    paths = e.paths()
    path = paths[int(rng.integers(len(paths)))]
    node = e.subtree(path)
    spec = OPS[node.op]
    candidates_ops = [op for op in space.ops_in_class(spec.arity, spec.slot) if op != node.op]
    mutate_literal = spec.slot is not None and (not candidates_ops or rng.random() < 0.5)
    if mutate_literal:
        domain = [v for v in space.domain(spec.slot) if v != node.value]  # type: ignore[arg-type]
        if not domain:
            return e
        new_node = Expr(node.op, node.children, domain[int(rng.integers(len(domain)))])
    elif candidates_ops:
        new_op = candidates_ops[int(rng.integers(len(candidates_ops)))]
        new_node = Expr(new_op, node.children, node.value)
    else:
        return e
    return e.replace(path, new_node)


def restricted_crossover(left: Expr, right: Expr, rng: np.random.Generator) -> Expr:
    """Swap subtrees at an IDENTICAL position between same-shape parents."""
    paths = left.paths()
    path = paths[int(rng.integers(len(paths)))]
    return left.replace(path, right.subtree(path))


# ------------------------------------------------------------------- seed run


@dataclass(slots=True)
class SeedRunResult:
    seed_id: str
    best_expr: Expr
    best_report: FitnessReport
    generations_run: int
    evaluations: int
    trace: list[float] = field(default_factory=list)  # best fitness per generation


@dataclass(frozen=True, slots=True)
class EngineConfig:
    pop_size: int = 64
    generations: int = 10
    stagnation: int = 3
    tournament_k: int = 4
    p_crossover: float = 0.5
    p_point: float = 0.4
    elitism: int = 1
    seed: int = 42
    max_attempts_factor: int = 20


def run_seed(
    seed_spec: SeedSpec,
    fitness_fn: Callable[[Expr], FitnessReport],
    cfg: EngineConfig,
    space: TreeSpace,
    *,
    init: str = "warmstart",
) -> SeedRunResult:
    """One independent frozen-structure search from one seed formula."""
    stream = int(hashlib.blake2b(seed_spec.seed_id.encode(), digest_size=4).hexdigest(), 16)
    rng = make_rng(cfg.seed, f"gp.seed.{seed_spec.seed_id}.{stream}")

    memo: dict[str, FitnessReport] = {}
    evaluations = 0

    def score(e: Expr) -> FitnessReport:
        nonlocal evaluations
        h = ast_hash(e)
        if h not in memo:
            memo[h] = fitness_fn(e)
            evaluations += 1
        return memo[h]

    base = canonicalize(seed_spec.expr)
    shape0 = tree_shape(base)

    # ---------- generation 0/1
    if init == "warmstart":
        population = [base]
        seen = {ast_hash(base)}
        attempts = 0
        while len(population) < cfg.pop_size and attempts < cfg.max_attempts_factor * cfg.pop_size:
            cand = point_mutate(base, rng, space)
            attempts += 1
            h = ast_hash(cand)
            if h not in seen:
                seen.add(h)
                population.append(cand)
    else:  # random-init baseline (same shape budget not enforced: free trees)
        from gp_cta.gp.expressions import random_tree

        population = []
        seen = set()
        attempts = 0
        while len(population) < cfg.pop_size and attempts < cfg.max_attempts_factor * cfg.pop_size:
            cand = random_tree(rng, space)
            attempts += 1
            h = ast_hash(cand)
            if h not in seen:
                seen.add(h)
                population.append(cand)

    scored = sorted(population, key=lambda e: score(e).fitness, reverse=True)
    best = scored[0]
    best_fit = score(best).fitness
    trace = [best_fit]
    stagnant = 0
    gens = 1

    def tournament(pop: list[Expr]) -> Expr:
        k = min(cfg.tournament_k, len(pop))
        idx = rng.choice(len(pop), size=k, replace=False)
        return max((pop[int(i)] for i in idx), key=lambda e: score(e).fitness)

    for _gen in range(2, cfg.generations + 1):
        nxt = [best]  # elitism 1
        nxt_hashes = {ast_hash(best)}
        attempts = 0
        while len(nxt) < cfg.pop_size and attempts < cfg.max_attempts_factor * cfg.pop_size:
            attempts += 1
            roll = rng.random()
            if roll < cfg.p_crossover and init == "warmstart":
                cand = restricted_crossover(tournament(scored), tournament(scored), rng)
            elif roll < cfg.p_crossover + cfg.p_point:
                cand = point_mutate(tournament(scored), rng, space)
            else:
                cand = tournament(scored)
            h = ast_hash(cand)
            if h in nxt_hashes:
                continue
            if init == "warmstart" and tree_shape(cand) != shape0:
                continue  # frozen-structure invariant (defensive; ops preserve it)
            nxt_hashes.add(h)
            nxt.append(cand)
        scored = sorted(nxt, key=lambda e: score(e).fitness, reverse=True)
        gens = _gen
        new_best = scored[0]
        new_fit = score(new_best).fitness
        if new_fit > best_fit + 1e-12:
            best, best_fit, stagnant = new_best, new_fit, 0
        else:
            stagnant += 1
        trace.append(best_fit)
        if stagnant >= cfg.stagnation:
            break

    logger.info(
        "seed {}: best fitness {:.4f} after {} gens ({} evals)",
        seed_spec.seed_id,
        best_fit,
        gens,
        evaluations,
    )
    return SeedRunResult(
        seed_id=seed_spec.seed_id,
        best_expr=best,
        best_report=score(best),
        generations_run=gens,
        evaluations=evaluations,
        trace=trace,
    )
