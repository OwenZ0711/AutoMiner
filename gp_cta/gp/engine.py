"""Mining engine: sequential warm-start seed runs → correlation-gated pool.

Seeds run SEQUENTIALLY (memory rule 6 — parallelism is capped by memory, and
one seed run's eval buffers are the working set). Each seed's best formula is
scored once on the VALIDATION split; the cross-seed CandidatePool then
rejects any candidate whose decimated train-signal correlation against an
already-pooled candidate exceeds ``max_abs_corr`` (AlphaGen's pool gate).
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from futures_common.paths import FuturesPaths
from gp_cta.gp.backtest import BacktestConfig, ProductCostArrays
from gp_cta.gp.expressions import Expr, TreeSpace, ast_hash
from gp_cta.gp.fitness import FitnessReport, evaluate_fitness
from gp_cta.gp.terminals import EvalContext, evaluate
from gp_cta.gp.warmstart import EngineConfig, SeedRunResult, SeedSpec, run_seed
from gp_cta.ledger import TrialLedger
from gp_cta.panel_io import PanelReader


@dataclass(slots=True)
class Candidate:
    seed_id: str
    expr: Expr
    hash: str
    train: FitnessReport
    validation: FitnessReport
    accepted: bool
    reject_reason: str = ""


@dataclass(slots=True)
class MiningReport:
    round_id: str
    init: str
    candidates: list[Candidate] = field(default_factory=list)
    best_validation_fitness: float = float("-inf")
    killswitch_shifted_fitness: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "init": self.init,
            "n_candidates": len(self.candidates),
            "n_accepted": sum(c.accepted for c in self.candidates),
            "best_validation_fitness": self.best_validation_fitness,
            "killswitch_shifted_fitness": self.killswitch_shifted_fitness,
            "candidates": [
                {
                    "seed_id": c.seed_id,
                    "formula": str(c.expr),
                    "hash": c.hash,
                    "train_fitness": c.train.fitness,
                    "validation_fitness": c.validation.fitness,
                    "accepted": c.accepted,
                    "reject_reason": c.reject_reason,
                }
                for c in self.candidates
            ],
        }


class CandidatePool:
    """Mutual-correlation gate over decimated train signals."""

    def __init__(self, max_abs_corr: float = 0.7, decimation: int = 15) -> None:
        self.max_abs_corr = max_abs_corr
        self.decimation = decimation
        self._signals: dict[str, np.ndarray] = {}

    def _decimate(self, signal: np.ndarray) -> np.ndarray:
        return signal[:: self.decimation].ravel().astype(np.float64)

    def offer(self, h: str, signal: np.ndarray) -> tuple[bool, str]:
        dec = self._decimate(signal)
        std = np.std(dec)
        if std <= 1e-12:
            return False, "degenerate signal (constant)"
        for other_hash, other in self._signals.items():
            n = min(dec.size, other.size)
            corr = np.corrcoef(dec[:n], other[:n])[0, 1]
            if np.isfinite(corr) and abs(corr) > self.max_abs_corr:
                return False, f"|corr|={abs(corr):.2f} vs pooled {other_hash[:8]}"
        self._signals[h] = dec
        return True, ""


def run_mining_round(
    panel: PanelReader,
    ctx: EvalContext,
    ctx_shifted: EvalContext,
    costs: Mapping[str, ProductCostArrays],
    seeds: Sequence[SeedSpec],
    *,
    paths: FuturesPaths,
    ledger: TrialLedger,
    engine_cfg: EngineConfig,
    bt_cfg: BacktestConfig,
    space: TreeSpace,
    train: tuple[dt.date, dt.date],
    validation: tuple[dt.date, dt.date],
    pool_max_corr: float = 0.7,
    pool_decimation: int = 15,
    init: str = "warmstart",
    round_id: str = "round",
) -> MiningReport:
    report = MiningReport(round_id=round_id, init=init)
    pool = CandidatePool(pool_max_corr, pool_decimation)

    for seed_spec in seeds:

        def train_fitness(e: Expr, _seed_id: str = seed_spec.seed_id) -> FitnessReport:
            return evaluate_fitness(
                e,
                ctx,
                panel,
                costs,
                bt_cfg,
                train,
                ledger=ledger,
                seed_id=_seed_id,
                split_name="train",
            )

        result: SeedRunResult = run_seed(seed_spec, train_fitness, engine_cfg, space, init=init)
        val_report = evaluate_fitness(
            result.best_expr,
            ctx,
            panel,
            costs,
            bt_cfg,
            validation,
            ledger=ledger,
            seed_id=seed_spec.seed_id,
            split_name="validation",
            allow_flip=False,
        )
        h = ast_hash(result.best_expr)
        signal = evaluate(result.best_expr, ctx)
        accepted, reason = pool.offer(h, signal)
        report.candidates.append(
            Candidate(
                seed_id=seed_spec.seed_id,
                expr=result.best_expr,
                hash=h,
                train=result.best_report,
                validation=val_report,
                accepted=accepted,
                reject_reason=reason,
            )
        )
        report.best_validation_fitness = max(report.best_validation_fitness, val_report.fitness)

    # +1-bar kill-switch (look-ahead gate (b)) on the best candidate
    if report.candidates:
        best = max(report.candidates, key=lambda c: c.train.fitness)
        shifted = evaluate_fitness(
            best.expr,
            ctx_shifted,
            panel,
            costs,
            bt_cfg,
            train,
            ledger=None,
            allow_flip=False,
        )
        report.killswitch_shifted_fitness = shifted.fitness
        logger.info(
            "kill-switch: best train fitness {:.4f} → shifted {:.4f}",
            best.train.fitness,
            shifted.fitness,
        )

    _persist_pool(report, paths)
    ledger.flush()
    return report


def _persist_pool(report: MiningReport, paths: FuturesPaths) -> None:
    paths.pool_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "round_id": report.round_id,
            "init": report.init,
            "seed_id": c.seed_id,
            "hash": c.hash,
            "formula": str(c.expr),
            "train_fitness": c.train.fitness,
            "validation_fitness": c.validation.fitness,
            "train_sharpe": c.train.mean_sharpe,
            "validation_sharpe": c.validation.mean_sharpe,
            "accepted": c.accepted,
            "reject_reason": c.reject_reason,
        }
        for c in report.candidates
    ]
    if not rows:
        return
    out = paths.pool_dir / f"{report.round_id}_{report.init}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    pl.DataFrame(rows).write_parquet(tmp)
    os.replace(tmp, out)
