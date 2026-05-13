"""Small genetic-programming engine for formula search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from gp_cta.evaluator import EvaluationResult, FormulaEvaluator
from gp_cta.expressions import Expr, random_expr
from gp_cta.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 100
    generations: int = 5
    max_depth: int = 4
    elite_size: int = 5
    tournament_size: int = 4
    mutation_rate: float = 0.35
    crossover_rate: float = 0.55
    seed: int = 42


@dataclass(frozen=True)
class Candidate:
    expr: Expr
    result: EvaluationResult


class GPEngine:
    def __init__(
        self,
        evaluator: FormulaEvaluator,
        config: GPConfig | None = None,
        features: tuple[str, ...] = FEATURE_COLUMNS,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or GPConfig()
        self.features = features
        self.rng = np.random.default_rng(self.config.seed)

    def evolve(self, train_panel: pl.DataFrame) -> list[Candidate]:
        population = [
            random_expr(self.rng, features=self.features, max_depth=self.config.max_depth)
            for _ in range(self.config.population_size)
        ]
        scored: list[Candidate] = []
        for _ in range(self.config.generations):
            scored = self._score(population, train_panel)
            elites = [candidate.expr for candidate in scored[: self.config.elite_size]]
            next_population = list(elites)
            while len(next_population) < self.config.population_size:
                roll = self.rng.random()
                if roll < self.config.crossover_rate and len(scored) >= 2:
                    parent_a = self._select(scored).expr
                    parent_b = self._select(scored).expr
                    child = self._crossover(parent_a, parent_b)
                elif roll < self.config.crossover_rate + self.config.mutation_rate:
                    child = self._mutate(self._select(scored).expr)
                else:
                    child = random_expr(
                        self.rng, features=self.features, max_depth=self.config.max_depth
                    )
                next_population.append(child)
            population = next_population
        return self._score(population, train_panel)

    def _score(self, population: list[Expr], frame: pl.DataFrame) -> list[Candidate]:
        scored = [
            Candidate(expr=expr, result=self.evaluator.evaluate(expr, frame))
            for expr in population
        ]
        return sorted(scored, key=lambda candidate: candidate.result.fitness, reverse=True)

    def _select(self, scored: list[Candidate]) -> Candidate:
        size = min(self.config.tournament_size, len(scored))
        idx = self.rng.choice(len(scored), size=size, replace=False)
        entrants = [scored[int(i)] for i in idx]
        return max(entrants, key=lambda candidate: candidate.result.fitness)

    def _crossover(self, left: Expr, right: Expr) -> Expr:
        left_paths = left.paths()
        right_paths = right.paths()
        left_path = left_paths[int(self.rng.integers(len(left_paths)))]
        right_path = right_paths[int(self.rng.integers(len(right_paths)))]
        child = left.replace(left_path, right.subtree(right_path))
        if child.depth() > self.config.max_depth + 1:
            return left
        return child

    def _mutate(self, expr: Expr) -> Expr:
        paths = expr.paths()
        path = paths[int(self.rng.integers(len(paths)))]
        replacement = random_expr(
            self.rng, features=self.features, max_depth=max(1, self.config.max_depth // 2)
        )
        child = expr.replace(path, replacement)
        if child.depth() > self.config.max_depth + 1:
            return expr
        return child
