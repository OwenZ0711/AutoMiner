# `STL pipelines/` — Pipeline A: STL CTA signal mining

Implementation of the per-asset Signal Temporal Logic miner described in `../proposal_version1.md` §4 (Pipeline A), §5.2 + §5.6 (the F_A fitness and pre/post-fitness gates), §11 (full algorithmic spec), and §11.6 (data-processing pipeline).

**Status:** to be built. Proposal v1 specifies 19 user-selected upgrades on top of the base Mohammadinejad-style MCTS-over-STL-AST. This file points at the proposal sections; the proposal is the source of truth.

## v1 design (one paragraph)

**Mining mode:** triple-barrier three-class meta-labelling (L1+L3+L5). Each (asset, t) sample on which a deterministic primary TSMOM-with-vol-filter says "long" carries a three-class label {long, flat, short} from vol-scaled triple-barrier. Magnitude-bucket weights (F6) on long-class samples; sample-uniqueness weights (L2) on every metric. **Search:** template-free Monte Carlo Tree Search over the STL formula AST (Mohammadinejad et al. 2023 lineage), with hierarchical features (M3, raw OHLCV ∪ Alpha158), multi-horizon mining at H ∈ {3, 5, 10, 20} (S1), multi-fidelity rollouts (S2), and transfer-learning warm-start from sector + vol-regime neighbours (S3). UCB1, action-mask discipline, random rollout, frequent-subtree avoidance. **Reward:** three-class margin separation `[m_long + α·m_short] / σ_ρ` minus complexity and FPR penalties plus a small Bayesian-shrunk realised-Sharpe tiebreak (F1). **Gates:** pre-fitness (F2 hit-count ≥ 30, E1 leakage assertion, complexity ≤ 32, no class collapse); post-fitness (F4 PR-AUC ≥ 0.40, F5 CVaR-5% ≥ −1%, F1 Sharpe-shrunk ≥ 0.6) on top of the base CTA gate (§6.2). **Pool:** admitted individually, blended at deployment via BMA over per-asset top-K (M2), nightly Sharpe re-eval with auto-evict (E4). **Hardware target:** M1 Mac, 16 GB RAM. **No deep learning in v1.**

## Reading list before coding

Papers in `../NLP and LLM automatic signal Miner/relative papers/`:
- **#2 Neural-STL** — STL-semantics background.
- **#4 STL Decision Trees / MILP** — 5-asset sanity baseline (proposal §11.5).
- **#3 TLINet, #1 RAMTL** — v2 references; *not* needed for v1.

External (not in `relative papers/`):
- **Mohammadinejad et al. 2023** — *Temporal logic explanations for dynamic decision systems using anchors and Monte Carlo Tree Search.* Artificial Intelligence Journal, DOI 10.1016/j.artint.2023.103905; AAAI 2024 reprint. Direct precedent for the v1 MCTS engine.
- **López de Prado** *Advances in Financial Machine Learning* — ch. 3 (triple-barrier, meta-labelling), ch. 4 (sample-uniqueness weights). Source for L1, L2, L3 implementations.

## Acceptance criteria for v1 (across Phase 3 / 3.5 / 3.6)

### Phase 3 — Base MCTS pipeline
- `Formula` AST with full STL grammar including hierarchical features.
- Vectorised robustness `ρ` in pure Polars across (asset × time) panel.
- `TripleBarrierLabelComputer` producing three-class labels + magnitude buckets, parquet cache keyed by rule hash.
- `SampleUniquenessWeighter` per López de Prado ch. 4, parquet cache.
- Deterministic `PrimaryModel` (TSMOM-with-vol-filter).
- F_A fitness with all components as independently testable pure functions.
- Pre-fitness gates (F2, E1, complexity, class-collapse) and post-fitness gates (F4, F5, F1).
- Base MCTS with action-mask, UCB1, random rollout, frequent-subtree avoidance.
- 5-asset MILP sanity check: MCTS top-1 ≥ 80 % of MILP optimum's separation margin.

### Phase 3.5 — Multi-horizon, multi-fidelity, warm-start
- 4 parallel populations per asset at H ∈ {3, 5, 10, 20}, admitted independently with horizon tags.
- Multi-fidelity rollouts (cheap 30 % subsample for first 70 % of iterations, full panel for last 30 %).
- Sector + vol-regime nearest-neighbour warm-start, with vanilla cold-start for first 30 % of universe.

### Phase 3.6 — BMA + DecayMonitor + full CSI 300 end-to-end
- BMA combiner: per-asset top-K (K=5 default) blended signal at deployment.
- DecayMonitor: nightly Sharpe re-eval with auto-evict triggers per §11.6.8.
- End-to-end CSI 300 mine across all 4 horizons with full admission stack.

## Out of v1 scope (deferred to v2)

- Differentiable refinement (TLINet) — Colab.
- Genetic Programming — superseded by MCTS for v1; could re-add in v2 as alt-search.
- Retrieval warm-start (RAMTL) — needs an embedding model.
- Continuous-position-mode fitness — defer; meta-labelling three-class is the v1 starting point.
- Negative-only / regime-conditional / drawdown-tolerant labelling modes — architecture supports them via `LabelComputer` swap; build only triple-barrier three-class in v1.
- LLM-policy prior in MCTS — v2 if compute allows.
