# Auto-Mining Alpha & Signal Factory — Proposal v1

**Author:** Owen
**Status:** Draft, May 2026 — *revised after (a) the Qlib runtime-dep pivot and (b) the Claude chat Research-mode review of evaluation metrics*
**Scope:** four parallel auto-miners feeding a single shared strategy pool, evaluated by two parallel admission gates with explicit multiple-testing correction, with a self-owned parquet/polars stack as the data foundation, Qlib's Alpha158 borrowed as a reference feature set, and a clear extension path for additional sources.

> **Convention.** Numerical thresholds that the research recommended but I (Owen) want to revisit on real CSI 300 backtest distributions are tagged `«TUNABLE»`. Greppable; default values are operative until I change them.

---

## 0. Decision log

| Date    | Decision                                                                                         | Why                                                                                                                                          |
|---------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-05 | **Drop Qlib as a runtime dependency.** Keep the github clone as a local read-only reference.      | Heavyweight abstractions for our scale; install footprint fragile (hard `gym` dep, pandas-3.x friction).                                     |
| 2026-05 | **Borrow Alpha158** from Qlib as a *feature-list*, ported into our own evaluator.                 | Well-understood baseline; cheap to port; useful as a Pipeline-C/D bootstrap feature set.                                                      |
| 2026-05 | **Stack: parquet + polars + DuckDB + custom DSL.**                                                 | Right-sized for ~500K rows total; 30-second debug loop; total operator-vocabulary control (which we need for STL).                            |
| 2026-05 | **Build the four pipelines as sibling folders under `AutoLLM/<pipeline>/`** instead of one mega-package. | Different runtime/dependency profiles per pipeline.                                                                                          |
| 2026-05 | **Drop IC/ICIR for Pipelines A and B; use CTA-style metrics.** Keep IC/ICIR for C and D.          | IC/ICIR is a cross-sectional concept; per-asset trend signals don't have one. CTA literature has its own metric stack and we adopt it.        |
| 2026-05 | **Decouple search reward from gate criterion.** Each pipeline has a smooth scalar fitness; each candidate also passes a multi-objective hard-threshold gate. | Standard Goodhart pattern: optimising a logical-AND of thresholds saturates gradients (RL/MCTS) or destroys GP fitness signal. The research-mode review made the case explicit; we adopt it. |
| 2026-05 | **Bake multiple-testing correction into the gate from v1:** Deflated Sharpe Ratio (DSR) for CTA, Harvey-Liu-Zhu BHY haircut on IC t-statistic for alpha, Combinatorial Purged k-Fold (CPCV) for all in-sample/out-of-sample splits with 10-day embargo, Combinatorially-Symmetric Cross-Validation Probability of Backtest Overfitting (CSCV-PBO) computed monthly on the joint pool. | With thousands of candidates per generation, raw thresholds give false-discovery rates in the high tens of percent (Bailey-López de Prado). Without this we're admitting noise. |
| 2026-05 | **Two parallel admission gates** (one for CTA candidates A∪B, one for alpha candidates C∪D) instead of one shared gate with branchy thresholds. | The two families share Stage-2 (correlation), Stage-3 (decay/regime), and Stage-4 (pool-incremental) shapes but want different thresholds and tiebreak scalars. Two gates is cleaner code than one branchy one. |
| 2026-05 | **Pipeline A v1 search algorithm = template-free MCTS, not GP.** Defer GP, TLINet differentiable refinement, retrieval warm-start to v2. | Hardware constraint (M1 Mac, 16 GB RAM, 512 GB SSD): no DL training comfortably runs locally. MCTS over the STL formula AST is pure-Python + Polars and has zero DL dependency. The Mohammadinejad et al. 2023 paper (`Artificial Intelligence` j., DOI 10.1016/j.artint.2023.103905; AAAI 2024 reprint) demonstrates exactly this approach for STL anchor-explanation mining and is the v1 reference. |
| 2026-05 | **Compute environment policy: keep v1 strictly on the M1 laptop.** Pipelines requiring meaningful DL training (B's PPO, C's PPO with cross-sectional ops, D's LLM agents) move to Colab/cloud or get sized down to MPS-compatible small models. See new §15. | M1 + 16 GB is plenty for tree search + Polars + Alpha158 + AKshare backfill. It is *not* enough for two simultaneous PyTorch workloads or for fp16 inference on a 7B+ LLM with reasonable throughput. Acknowledge the constraint up front; design v1 around it; run v2 components on Colab where they need to. |
| 2026-05 | **Pipeline A v1 mines with triple-barrier three-class labels.** Replace the simple-window positive-only rule with López de Prado's triple-barrier method (profit-take, stop-loss, time-out, all vol-scaled), producing **three classes** {long, flat, short}. Long-side only at deployment (Chinese A-share short-sale restrictions); short-class labels still inform the formula. Subject to change. | Triple-barrier respects how a real trade actually closes; vol-scaled barriers handle regime-dependent return scales without re-tuning; three classes are more informative than binary for what an STL formula can express. Standard in modern quant ML (López de Prado *AFML* ch. 3). |
| 2026-05 | **Sample-uniqueness weights everywhere.** Every metric (margin, IC, Sharpe, DSR, F1, PR-AUC) is computed with López de Prado's sample-uniqueness weights to handle overlapping forward-return windows. | Without this, daily samples with overlapping forward windows are double-counted and DSR's i.i.d. assumption is silently wrong. Effective sample size is ~ N / forward-window-length, not N. (AFML ch. 4.) |
| 2026-05 | **Meta-labelling architecture.** Pipeline A learns **meta** labels — given a *primary* model emits a "long" signal, predict whether to take it. The v1 primary is a deterministic TSMOM filter (`20d momentum > 0 AND realised vol < 60d median`). | Decouples direction from entry-quality, collapses MCTS search space ~5×, isolates STL's value-add to the entry-timing question. AFML ch. 3, also the implicit shape of AlphaAgent's three-agent split. |
| 2026-05 | **Multi-horizon mining.** Mine four parallel MCTS populations per asset at horizons {3, 5, 10, 20} days. Admit each independently into the pool. | Different formulas are best at different horizons; mining at a single horizon throws this away. Cheap because MCTS is already per-asset embarrassingly parallel. |
| 2026-05 | **Hierarchical features.** Atomic predicates can use Alpha158 features (`alpha158_ROC_5 > 0.02`) as well as raw OHLCV. | Lets MCTS reason over derived features without rediscovering them. The whole reason we ported Alpha158 in §8.2 was to use it as a substrate. |
| 2026-05 | **Bayesian model averaging at deployment, not at admission.** The pool admits individual formulas; at deployment time, the per-asset signal is a BMA-weighted blend of the asset's top-K admitted formulas. | Single-best-formula-per-asset is fragile; BMA reduces variance with no admission-logic complexity cost. Standard in HF allocators. |

---

## 1. Executive summary

Build a **research factory with two stages** — a *Mining Factory* and an *Estimation Factory* — connected by a shared, version-controlled strategy pool. Inside the Mining Factory run **four independent miners in parallel**, each searching a different representation × algorithm cell of the design space:

|       | Per-asset (CTA-like)                          | Cross-sectional (alpha-factor)                 |
|-------|-----------------------------------------------|------------------------------------------------|
| **Symbolic / STL** | **Pipeline A** — STL miner (template-free MCTS in v1; differentiable / GP later) | — (deferred)                                    |
| **Reinforcement learning** | **Pipeline B** — RL CTA miner | **Pipeline C** — RL alpha miner |
| **LLM-driven** | — (deferred) | **Pipeline D** — LLM alpha miner |

Each pipeline produces candidates which flow through the **Estimation Factory**. The Estimation Factory contains **two parallel admission gates** — one for the CTA family (A, B), one for the alpha family (C, D) — with appropriate multi-objective threshold tests, multiple-testing correction (DSR for CTA, Harvey-Liu-Zhu haircut for alpha), pool-incremental contribution checks, and a deflated-tiebreak scalar for ranking inside the admitted set.

**Day-one data foundation:** a custom `data_fetcher` module (AKshare → parquet, polars in memory, DuckDB for ad-hoc queries) plus a thin `DataAdapter` protocol consumed by every pipeline. Alpha158 ported from Qlib gives a reference feature set. Other sources slot in via the same adapter.

**Borrow-but-don't-clone principle:** for each pipeline, this proposal lists which papers' *ideas* we adopt, which *interfaces* we copy verbatim because they save weeks, and which choices we deliberately deviate from because they are over-fit to a regime we don't share.

---

## 2. Goals and non-goals

### Goals
- A reproducible, modular, version-controlled pool of trading signals — every entry has a parent miner, a hash, a backtest report, an admission record (including which multiple-testing test it passed), and a decay history.
- Four miners that share **only the data layer, the evaluator, and the pool** — not the search algorithm, not the representation. Heterogeneity in the pool is the explicit research bet.
- An admission process that operationalises the user's stated criteria — IC, IR, Sharpe, turnover, absolute return, correlation with existing pool — into a deterministic, reviewable pipeline with explicit FDR control. No hand-curation in production.
- A research-to-production path: the same code that mines also evaluates incumbents nightly and flags decay.

### Non-goals (v1)
- Live trading / order execution.
- A single "best" model. We never pick a winner.
- Beating SOTA on any specific paper benchmark.
- A web UI. Use Jupyter / standalone MLflow / a small Streamlit dashboard at most.

---

## 3. System architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER (DataAdapter)                          │
│  data_fetcher (AKshare → parquet) + Alpha158 derived features cache         │
│  ── future: tushare, ricequant, news/event APIs, alt data ───              │
└──────────────────┬─────────────────────────────────────────────────────────┘
                   │ DataAdapter.panel(universe, fields, start, end) -> Polars LazyFrame
                   │
       ┌───────────┴───────────────────────────────────────────────────┐
       │                       MINING FACTORY                          │
       │                                                               │
       │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
       │  │ PIPELINE A │ │ PIPELINE B │ │ PIPELINE C │ │ PIPELINE D │ │
       │  │  STL-CTA   │ │  RL-CTA    │ │ RL-Alpha   │ │ LLM-Alpha  │ │
       │  │ fitness F_A│ │ fitness F_B│ │ fitness F_C│ │ fitness F_D│ │
       │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ │
       │        └──────────────┴──────┬───────┴──────────────┘         │
       │              CandidateMessage (uniform shape)                 │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
       ┌──────────────────────────────┴───────────────────────────────┐
       │                     ESTIMATION FACTORY                        │
       │                                                               │
       │   ┌────────────────┐         ┌────────────────────┐           │
       │   │   CTA gate     │         │   Alpha gate       │           │
       │   │   (A ∪ B)      │         │   (C ∪ D)          │           │
       │   │                │         │                    │           │
       │   │ Stage-1: Sharpe│         │ Stage-1: IC/ICIR    │           │
       │   │ Stage-2: corr  │         │ Stage-2: corr/AST   │           │
       │   │ Stage-3: decay │         │ Stage-3: decay      │           │
       │   │ Stage-4: ΔSharpe│         │ Stage-4: ΔICIR     │           │
       │   │ MTC: DSR        │         │ MTC: HLZ haircut   │           │
       │   └────────────────┘         └────────────────────┘           │
       │   System-wide CSCV-PBO computed monthly on the joint pool     │
       │   ─→ admit + record   |   reject + reason → Quarantine        │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                          ┌───────────┴────────────┐
                          │      STRATEGY POOL     │
                          │ parquet + sqlite       │
                          │ CTA sub-pool / alpha   │
                          │ sub-pool / joint pool  │
                          └────────────────────────┘
```

Two cross-cutting services:
- **EvaluatorService** — pure function `(formula | callable, universe, train/val/test windows, metric_family) -> MetricsReport`. `metric_family ∈ {"cta", "alpha"}` selects the report shape. Single source of truth for every metric.
- **PoolService** — wraps the on-disk pool. Read API: list members, fetch their daily PnL panels, compute pairwise return correlation, expose AST forest. Write API: append a candidate atomically with full audit row.

---

## 4. The four pipelines (high-level)

Full fitness formulas live in §5. Full Estimation-Factory thresholds in §6. Pipeline A's full implementation spec is in §11.

### Pipeline A — STL-CTA miner
- **Representation.** STL formulas with **hierarchical features**: atomic predicates can be `(raw_feature CMP threshold)` over `{$close, $open, $high, $low, $volume, ret_1d, ret_5d, vol_20d}` *or* over **any Alpha158 feature** (`alpha158_RSI_14 > 0.7`, `alpha158_ROC_20 < 0`). Bounded depth (≤4) and bounded interval lengths.
- **Mining mode (v1).** **Triple-barrier three-class labels** + **meta-labelling**. A primary deterministic TSMOM filter emits "long" candidate days; for each such day, the triple-barrier method assigns one of {long, flat, short} based on whether profit-take / stop-loss / time-out fires first; vol-scaled barriers (default ±2σ_t profit-take, −1σ_t stop-loss, 10d time-out). The MCTS miner predicts the label conditional on the primary. **Sample-uniqueness weights** applied across all metrics. Long-only at deployment (Chinese A-share short-sale restrictions; short-class labels still inform the formula).
- **Search (v1).** **Template-free Monte Carlo Tree Search** over the STL formula AST. Mohammadinejad et al. 2023 (AIJ / AAAI 2024 reprint) is the direct precedent. **Three search additions** beyond their baseline: (S1) **multi-horizon** — four parallel populations per asset at horizons {3, 5, 10, 20} days; (S2) **multi-fidelity** — sub-sampled days for early rollouts, full panel for late refinement; (S3) **transfer-learning warm-start** — for asset B, seed MCTS from the top-K formulas mined on B's industry-sector neighbour A. **No deep learning in v1.**
- **Search (v2 only).** GP, TLINet differentiable refinement, RAMTL retrieval warm-start, all deferred. Each is a Colab workload by the time we add it.
- **Fitness shape (full formula §5.2).** Multi-class separation across {long, flat, short} as the primary smooth term, plus a Bayesian-shrunk realised-Sharpe tiebreak. **Magnitude-aware** sample weights (long-class samples weighted by their realised return-bucket).
- **Pre-fitness gates (cheap, fail-fast).** Min-hit-count ≥ 30 trades on train (F2). Feature/label-window non-overlap assertion (E1).
- **Pool admission gates beyond §6.2 base CTA gate.** PR-AUC for long-class ≥ 0.40 (F4). 5%-CVaR of per-trade returns ≥ −1% (F5).
- **Pool combination at deployment.** **Bayesian model averaging** over the asset's top-K admitted formulas (M2), weights from posterior under a no-skill prior. Each asset gets a blended signal, not a single-best.
- **Decay tracking** (E4). Nightly re-evaluation of every admitted formula's rolling 60d Sharpe; auto-evict on threshold crossing.
- **Borrow.** Mohammadinejad et al. 2023 for MCTS-over-STL-AST. López de Prado *AFML* for triple-barrier (ch. 3), sample-uniqueness (ch. 4), meta-labelling (ch. 3). STL Decision Trees MILP (#4) for tiny-universe sanity baselines. TLINet (#3), AGM robustness, RAMTL — v2 deferred.

### Pipeline B — RL-CTA miner
- **Representation.** Reverse-Polish-notation token sequences over per-asset operators (no cross-sectional ops).
- **Search.** REINFORCE with the QFR (#7) variance-bounded greedy baseline. (Not PPO — for the deterministic factor MDP, QFR Proposition 1 says trajectory-level REINFORCE has lower variance than PPO with a critic.)
- **Fitness shape (§5.3).** `Sharpe_net + λ₄·Sortino − λ₅·MaxDD − λ₆·Turnover`. **Lazy switch:** once the CTA sub-pool has ≥ 5 admitted strategies, swap to `F_B^pool = Sharpe(pool ∪ {φ}) − Sharpe(pool)` — the CTA analog of AlphaGen's pool-incremental ICIR.
- **Borrow.** AlphaGen's RPN encoding + action mask (algorithm only; our adapter underneath). QFR's baseline math (~30 lines of PyTorch). Hurst-Ooi-Pedersen "Century of Trend Following" — Sortino > Sharpe across nearly all decades is why Sortino is a fitness term.

### Pipeline C — RL alpha-factor miner
- **Representation.** RPN with cross-sectional operators (`CSRank`, `CSZScore`, `CSMean`) added.
- **Search.** REINFORCE + QFR baseline + pool-incremental ICIR reward (the AlphaGen reward, restored).
- **Fitness shape (§5.4).** `ΔICIR_pool + λ₇·IR_IC − λ₈·maxCorr − λ_{8b}·ASTSim − λ₉·Complexity`. The AST-similarity term is added for symmetry with Pipeline D — return-correlation and AST-similarity catch *different* duplicate-detection failures (AlphaAgent paper makes this case empirically).
- **Borrow.** AlphaGen's training loop + tokeniser + 6-feature input set (`open, high, low, close, volume, vwap`); replace its `AlphaCalculator` with our `EvaluatorService`. QFR's IR-shaped reward.

### Pipeline D — LLM alpha-factor miner
- **Representation.** Two-tier: NL hypothesis → DSL expression (same operator set as C).
- **Search.** Three-agent loop wrapped in MCTS, LLM as policy/value prior in PUCT.
- **Fitness shape (§5.5, used as MCTS leaf-backup value).** `ICIR + λ₁₀·RankIC − λ₁₁·ASTSim − λ_{11b}·maxCorr − λ₁₂·ParamCount − λ₁₃·𝟙[FrequentSubtree]`. The `maxCorr` term is added for symmetry with Pipeline C.
- **Borrow.** AlphaAgent (#9) three-agent split + AST-similarity novelty + complexity regularisers — the three regularisers their paper empirically validates against alpha decay. Alpha-Jungle (#10) MCTS pattern + frequent-subtree avoidance. Alpha-GPT (#8) hypothesis-first prompt structure for the Idea Agent.

**Shared design principle.** A and B share the term family `Sharpe_net + drawdown/turnover penalties`. C and D share `ICIR + Rank-IC + diversity penalties`. This is the fitness-family split the literature actually does — ICIR is a cross-sectional quantity; per-asset CTA signals don't have one.

---

## 5. Per-pipeline fitness functions (the search reward)

The fitness is the **inner search reward** — a single smooth scalar each pipeline's optimiser climbs. It is not the gate (§6); the gate is multi-objective and threshold-based. The two are designed to be monotone-aligned but deliberately different shapes.

### 5.1 Why per-pipeline (not unified)

A unified fitness is mathematically incoherent:

- ICIR is cross-sectional — averages of rank correlations *across assets* per day. For a per-asset CTA strategy the cross-section has size one, so ICIR is undefined.
- Sharpe / Sortino are time-series — moments of a single asset's PnL series. For a cross-sectional alpha that produces a *rank score* rather than a position, Sharpe is computed on the long-short decile portfolio, not the alpha itself.

So A/B share a CTA-style fitness family and C/D share an IC-style alpha fitness family. This is non-negotiable.

The four formulas below all have the same structural shape: `core_metric + λ_diversity_or_stability_term − λ_complexity_term − λ_cost_term`. The differences are which metric is the core and which terms are smoothing additions.

### 5.2 F_A — STL-CTA fitness (Pipeline A, v1: triple-barrier three-class with meta-labelling)

In v1 Pipeline A is a **three-class meta-labelling classifier**. Each (asset, t) sample on which the *primary* TSMOM filter said "long" carries a triple-barrier label $y_t \in \{\text{long}, \text{flat}, \text{short}\}$ (§11.6.1). The MCTS miner predicts the label from STL robustness $\rho(\phi, x, t)$. **All means and correlations below use sample-uniqueness weights $w_t$** (§11.6.3) to down-weight overlapping forward-return windows.

The fitness is a **smooth scalar for MCTS rollouts**; the gate (§6.2) and the per-candidate quality bar use additional hard-thresholded metrics that are *not* in the fitness (PR-AUC, CVaR, hit-count) — see §5.6.

$$
F_A(\phi) = \underbrace{\frac{m_{\text{long}}(\phi) + \alpha_{\text{short}} \cdot m_{\text{short}}(\phi)}{\sigma_{\rho}(\phi)}}_{\text{three-class normalised margin}} - \lambda_{\text{cplx}} \cdot \mathrm{Complexity}(\phi) - \lambda_{\text{FP}} \cdot \mathrm{FPR}(\phi) + \lambda_{S} \cdot \widetilde{\mathrm{Sharpe}}_{\text{net}}^{\text{realised}}(\phi)
$$

Components:

- $m_{\text{long}}(\phi) = \widehat{\mathbb{E}}_{w}\bigl[\rho \mid y = \text{long}\bigr] - \widehat{\mathbb{E}}_{w}\bigl[\rho \mid y \ne \text{long}\bigr]$ — weighted mean robustness on long-class samples *minus* the rest. **Magnitude-weighted (F6):** within the long class, samples are further weighted by their realised return-bucket — a +5% entry contributes more than a +2.01% entry. Weight schedule: bucket `(2%, 4%]` → ×1.0, `(4%, 6%]` → ×1.5, `> 6%` → ×2.0. `«TUNABLE»`.
- $m_{\text{short}}(\phi) = \widehat{\mathbb{E}}_{w}\bigl[-\rho \mid y = \text{short}\bigr] - \widehat{\mathbb{E}}_{w}\bigl[-\rho \mid y \ne \text{short}\bigr]$ — same shape, sign-flipped (we want $\rho < 0$ on short days). Even though we don't trade short in v1, the formula learns better discrimination when it has to recognise *both* tails.
- $\alpha_{\text{short}} = 0.3$ default `«TUNABLE»` — long is the deployment class; short is informational, weighted lower.
- $\sigma_\rho(\phi)$ — pooled $\rho$-std (sample-weighted) for scale normalisation.
- $\mathrm{Complexity}(\phi)$ — AST node count / `max_formula_size` (default 32).
- $\mathrm{FPR}(\phi) = \widehat{\mathbb{E}}_{w}\bigl[\mathbb{1}[\rho > 0] \mid y \ne \text{long}\bigr]$ — false-positive rate (firing on flat-or-short days).
- $\widetilde{\mathrm{Sharpe}}_{\text{net}}^{\text{realised}}(\phi)$ — **Bayesian-shrunk** realised Sharpe of the long-only position panel `position_t = 1 if ρ > 0 else 0`, post 30 bp round-trip cost. James-Stein-style shrinkage toward zero with shrinkage factor `k / (t² + k)` (§11.4.6). Defaults to a near-zero contribution for low-firing formulas.

**Default λ's:** $\lambda_{\text{cplx}} = 0.001$, $\lambda_{\text{FP}} = 0.5$, $\lambda_{S} = 0.1$. `«TUNABLE»`.

**Why these terms and not others.** The fitness is deliberately small (4 weighted scalar terms). Other quality measures the user asked for — PR-AUC (F4), per-trade CVaR-5% (F5), min-hit-count (F2) — are **gate criteria, not fitness terms** (§5.6). Putting them in the fitness saturates gradients (Goodhart) and makes MCTS unstable. They live one level up.

**MCTS reward.** UCB1 with $c_{\mathrm{UCB}} = \sqrt{2}$ default `«TUNABLE»`. Multi-fidelity (S2): cheap rollouts on a 30 % stratified sub-sample of days for the first 70 % of MCTS iterations; full panel for the last 30 %. Transfer-learning warm-start (S3): when starting on a new asset, seed the tree's first 5 000 visits with the top-K formulas mined on the asset's industry-sector neighbour, biased toward their root expansions.

**Position rule for downstream backtest:** `position_t = 1` if $\rho(\phi, t) > 0$ AND primary said long, else `0`. Long-only.

**v2 evolution.** Continuous-position fitness (Sharpe-driven) becomes available as `mode="continuous"` alongside `mode="meta_label_3class"`. Both share the same MCTS / formula / DataAdapter machinery.

### 5.3 F_B — RL-CTA fitness (Pipeline B)

$$
F_B(\phi) = \widehat{Sharpe}_{net}(r^{\phi}) + \lambda_4 \cdot \widehat{Sortino}(r^{\phi}) - \lambda_5 \cdot \mathrm{MaxDD}(r^{\phi}) - \lambda_6 \cdot \mathrm{Turnover}(\phi)
$$

After the CTA sub-pool has $\geq 5$ admitted strategies, **lazy-switch** to:

$$
F_B^{pool}(\phi) = \mathrm{Sharpe}(\mathrm{pool}_B \cup \{\phi\}) - \mathrm{Sharpe}(\mathrm{pool}_B)
$$

with the pool weighted by Baltas-Kosowski 2017 correlation-adjusted volatility-targeting. (Don't switch back if pool drops below 5 — switch is one-way to avoid reward-shape oscillation.)

**Default λ's:** $\lambda_4 = 0.3$, $\lambda_5 = 0.2$, $\lambda_6 = 0.5$. `«TUNABLE»`.

**Trajectory-level rewards** per QFR Proposition 1. Use the QFR greedy-policy baseline; do not use PPO with a critic.

**Open question** (worth its own ablation in Phase 5): does $F_B^{pool}$ beat $F_B$ as a search reward in CTA space? The AlphaGen experiment showed pool-incremental beats single-instance for cross-sectional alphas, but the analog for CTA is not in the public literature.

### 5.4 F_C — RL alpha-factor fitness (Pipeline C)

$$
F_C(\phi) = \underbrace{\mathrm{ICIR}(\mathrm{pool}_C \cup \{\phi\}) - \mathrm{ICIR}(\mathrm{pool}_C)}_{\text{pool-incremental ICIR (AlphaGen)}} + \lambda_7 \cdot \widehat{IR}_{IC}(\phi) - \lambda_8 \cdot \mathrm{maxCorr}(\phi, \mathrm{pool}_C) - \lambda_{8b} \cdot \mathrm{ASTSim}(\phi, \mathrm{pool}_C) - \lambda_9 \cdot \mathrm{Complexity}(\phi)
$$

**Default λ's:** $\lambda_7 = 0.5$, $\lambda_8 = 1.0$, $\lambda_{8b} = 0.5$, $\lambda_9 = 0.001$. `«TUNABLE»`.

**Rationale.** AlphaGen's central claim (Yu et al. KDD 2023) is that mining single-alpha IC produces redundant alphas; pool-incremental contribution rewards synergy directly. The QFR IR-shaping ($\widehat{IR}_{IC}$) pushes for *stable* IC across windows, not just headline ICIR. The maxCorr term penalises behavioural redundancy; the ASTSim term penalises structural redundancy (the AlphaAgent paper shows these catch different failures).

**Universe.** Replicate AlphaGen exactly: CSI 300, six raw features `{open, high, low, close, volume, vwap}`, 20-day forward return label.

### 5.5 F_D — LLM alpha-factor fitness (Pipeline D, MCTS leaf-backup)

$$
F_D(\phi) = \mathrm{ICIR}(\phi) + \lambda_{10} \cdot \mathrm{RankIC}(\phi) - \lambda_{11} \cdot \mathrm{ASTSim}(\phi, \mathrm{pool}_D) - \lambda_{11b} \cdot \mathrm{maxCorr}(\phi, \mathrm{pool}_D) - \lambda_{12} \cdot \mathrm{ParamCount}(\phi) - \lambda_{13} \cdot \mathbb{1}[\mathrm{FrequentSubtree}(\phi)]
$$

**Default λ's:** $\lambda_{10} = 0.5$, $\lambda_{11} = 0.2$, $\lambda_{11b} = 0.2$, $\lambda_{12} = 0.005$, $\lambda_{13} = 0.2$. `«TUNABLE»`.

**Hypothesis-factor alignment** is implemented as an LLM-judge **gate** at the Idea-Agent → Factor-Agent step (binary: pass/fail), not a continuous reward term, to keep $F_D$ smooth.

**MCTS integration.** LLM's value estimate is the prior $P(s, a)$ in PUCT. Backed-up leaf value is $F_D$ from the Eval-Agent's backtest. Frequent-subtree avoidance is an **additional** prompt-time prior on top of the $\lambda_{13}$ reward term — Alpha-Jungle's actual implementation puts it at prompt-time; we keep both belt-and-braces in v1.

### 5.6 Pipeline A pre-fitness gates and post-fitness admission gates

For Pipeline A specifically, four cheap **pre-fitness gates** run before $F_A$ is even computed (any failure → reject this candidate, MCTS treats as $F_A = -\infty$):

1. **Min-hit-count (F2).** Reject if the formula fires on < 30 long-class days in the train window. Below this threshold no statistical claim is meaningful.
2. **Feature/label leakage assertion (E1).** Assert no overlap between the maximum feature look-back and the label forward-window. Should be impossible by construction; failing means a formula uses a feature whose computation peeks past `t`.
3. **Complexity bound.** Reject if AST size > `max_formula_size = 32` (the smooth complexity penalty in $F_A$ is a soft prior; this is the hard ceiling).
4. **Trivial-class collapse.** Reject if the formula's predicted-class distribution is degenerate (e.g. always-long, always-flat).

Three additional **post-fitness admission gates** layered on top of the §6.2 base CTA gate, applied only to candidates the MCTS has selected as top-K and which passed the base CTA gate:

5. **PR-AUC long-class (F4).** AUC of precision-recall curve when treating $\rho > \tau$ as the long-class predictor, swept over $\tau$. Threshold `«TUNABLE»` ≥ **0.40** at admission. PR-AUC is more honest than ROC-AUC under class imbalance.
6. **Per-trade CVaR-5% (F5).** $\mathrm{CVaR}_{5\%}$ of realised per-trade returns — the average return on the worst 5 % of trades. Threshold ≥ **−1 %** `«TUNABLE»`. A formula that occasionally produces −10 % trades fails even if its mean is good.
7. **Bayesian-shrunk Sharpe ≥ 0.6** (cheaper threshold than the un-shrunk Sharpe ≥ 0.8 in §6.2 because shrinkage already penalises noise).

### 5.7 Search-reward vs gate-criterion split

These are *not* the same object. The gate is in §6. The fitness is what the optimiser climbs.

**Search reward must be:**
- single scalar (RL/MCTS/GP all need this)
- smooth / not saturating (the textbook Goodhart pattern)
- cheap on every rollout
- monotone-aligned with the gate but not identical

**Gate criterion must be:**
- multi-objective (Sharpe AND turnover AND DD AND ΔICIR_pool, etc.)
- hard-thresholded (pool composition stays interpretable)
- computed on a held-out window with multiple-testing correction
- has a tiebreak scalar (DSR for CTA, deflated IC t-stat for alpha)

**Pseudocode:**

```python
# inside each generation, for each pipeline P in {A, B, C, D}
for episode in range(N_episodes):
    candidate = miner_P.sample()
    metrics = backtest(candidate, train_window)
    reward = F_P(candidate, metrics, current_pool_P)        # smooth scalar
    miner_P.update(reward)

# every K generations
for candidate in top_M_by_reward:
    if gate_pass(candidate, holdout_window):                # multi-threshold
        if multiple_testing_pass(candidate, K * N_episodes):# DSR or HLZ
            if pool_incremental_pass(candidate, pool):       # ΔICIR or ΔSharpe
                pool.admit(candidate, score=tiebreak(candidate))
```

---

## 6. Estimation Factory — two parallel admission gates

### 6.1 Why two gates

The CTA family (A, B) and the alpha family (C, D) share gate *shapes* but want different *thresholds* and different *tiebreak scalars*. Implementing this as one branchy gate with `if metric_family == "cta": ...` is ugly; two parallel gates with a small shared base class is clean.

The Stage-1 metrics differ (Sharpe-dominated vs IC-dominated). Stage-2 (pool correlation), Stage-3 (decay/regime), and Stage-4 (pool-incremental contribution) are conceptually shared; the formulas differ.

### 6.2 CTA gate (admits A∪B candidates into the CTA sub-pool)

**Stage 1 — stand-alone post-cost performance** (3-year rolling, walk-forward via CPCV §6.4):

| Metric                              | Threshold                                                          | Notes |
|-------------------------------------|--------------------------------------------------------------------|-------|
| Annualised Sharpe (post-cost)       | ≥ **0.8** `«TUNABLE»` *(research suggested 1.0; we use 0.8 because DSR ≥ 0.95 below already does the FDR work)* |  |
| Annualised Sortino (post-cost)      | ≥ **1.4** `«TUNABLE»`                                               | Trend-following naturally has positive skew (Hurst-Ooi-Pedersen) |
| Calmar (CAGR / |MaxDD|)             | ≥ **0.5** `«TUNABLE»`                                               |  |
| Max drawdown                        | ≤ **20%** post-cost on train+val `«TUNABLE»`                        | Tighten to 15% if leveraged via index futures |
| Daily one-sided turnover            | ≤ **15%** for A, ≤ **25%** for B `«TUNABLE»`                        | RL has known overtrading; STL is naturally regime-switching |
| Time-in-market                      | ≥ **30%**                                                            | Filters "rarely trades" Sharpe-via-non-participation signals |
| Annual absolute return (post-cost)   | ≥ **8%** `«TUNABLE»`                                                |  |
| Profit factor                       | ≥ **1.3**                                                            |  |
| Win rate                            | **35% – 65%**, **reject if > 75%** `«TUNABLE»`                      | High win-rate is a CTA red flag (mean-reversion fit-to-noise). The research said reject at 70%; we use 75% until calibrated on CSI 300 distributions. |
| Average holding period              | **3 – 60** trading days                                              | 1–2 days: cost economics break (post-T+1 settlement still pays the round-trip). > 60 days: untestable on 5–7y data. |
| Equity-curve linearity (R² of cum-log-PnL on time) | ≥ **0.6**                                              | Penalises lumpy P&L |
| **Pipeline A only — Min-hit-count (F2)**            | ≥ **30** long-class trades on train                                  | Statistical-power floor; pre-fitness reject |
| **Pipeline A only — PR-AUC for long class (F4)**    | ≥ **0.40** `«TUNABLE»`                                               | Imbalance-honest replacement for accuracy/F1 |
| **Pipeline A only — Per-trade CVaR-5% (F5)**         | ≥ **−1 %** `«TUNABLE»`                                                | Tail-aware; rejects formulas with occasional −10% trades |
| **Pipeline A only — Bayesian-shrunk Sharpe (F1)**    | ≥ **0.6** `«TUNABLE»`                                                | Lower than the un-shrunk threshold above because shrinkage already penalises noise |

**Stage 2 — return + structural correlation** (vs every existing CTA-sub-pool member):

- Pearson correlation of strategy returns ≤ **0.5** (Baltas-Kosowski's pairwise-corr point: pool diversification depends on this).
- AST/structural similarity: TF-IDF cosine over operator-token frequencies ≤ **0.7**, and longest common subtree ≤ **5** nodes.

**Stage 3 — decay & regime stability:**

- Rolling 6-month Sharpe sign-flip rate ≤ **30%** of windows.
- Worst rolling 12-month Sharpe ≥ **−0.3**.
- Regime-conditional Sharpe must be **non-negative** in at least 3 of 4 regimes: `{bull-CSI300, bear-CSI300, high-realised-vol, low-realised-vol}`. Regime labels computed from CSI 300 itself: bull/bear by 60d return sign, high/low vol by trailing-20d realised vol vs its 60d median. (Don't use a "Chinese VIX" — China doesn't have a deeply-liquid one; iVIX exists for SSE 50 ETF but is too thin for regime-conditioning.)

**Stage 4 — pool-incremental contribution:**

- ΔPool-Sharpe ≥ **0.10** when the candidate is added to the CTA sub-pool under Baltas-Kosowski correlation-adjusted volatility-targeted weighting (not equal-weight).

**Tiebreak / ranking inside admitted:** Deflated Sharpe Ratio ≥ **0.95** (see §6.4) at the asserted significance level, with $N$ = total trials by miner A∪B in the current generation. Candidates are ranked by DSR within the admitted set.

### 6.3 Alpha-factor gate (admits C∪D candidates into the alpha sub-pool)

**Universe** (replicate AlphaGen exactly): CSI 300 + AlphaGen's six raw features. Splits: 2018–2022 train / 2023 val / 2024–2025 test, advancing one year per quarter.

**Stage 1 — stand-alone metrics:**

| Metric                                        | Threshold                                  |
|-----------------------------------------------|--------------------------------------------|
| Pearson IC (mean over days, 20-day forward return) | ≥ **0.025** `«TUNABLE»`           |
| Rank-IC (Spearman, mean over days)            | ≥ **0.03** `«TUNABLE»`                     |
| ICIR                                          | ≥ **0.30**                                  |
| Rank-ICIR                                     | ≥ **0.40**                                  |
| Long-short top-decile minus bottom-decile annualised return | ≥ **8%**             |
| Daily turnover of long-only top-50 portfolio  | ≤ **40%** `«TUNABLE»`                       |

> **Calibration anchor (verify):** Qlib's published LightGBM-on-Alpha158 baseline on CSI 300 reportedly lands at IC ≈ 0.040, ICIR ≈ 0.41, Rank-ICIR ≈ 0.51. **Spot-check this against `qlib/examples/benchmarks/LightGBM/README.md`** before relying on it as a calibration anchor — these specific numbers came from the research review and I haven't independently verified.

**Stage 2 — return + structural correlation:**

- Long-short return correlation with existing alpha-pool members ≤ **0.6**.
- AST-similarity (cosine on operator TF-IDF) ≤ **0.7**, frequent-subtree share ≤ **30%** (Alpha-Jungle's metric).

**Stage 3 — decay & regime stability:**

- IC half-life ≥ **8** trading days (estimated by exponential fit to lagged-IC profile).
- IC sign-flip rate per 60-day rolling window ≤ **30%**.
- Worst rolling 6-month IC ≥ **0.005** (the alpha is at least non-anti-predictive in the worst regime).
- ICIR per regime: at least 3 of 4 ≥ **0.10**.

**Stage 4 — pool-incremental:**

- ΔICIR of the ridge-combiner ≥ **0.05**.
- Marginal contribution to combination Rank-IC ≥ **0.005**.

**Tiebreak / ranking inside admitted:** Deflated IC t-statistic above the Harvey-Liu-Zhu BHY-haircut threshold (see §6.4). Candidates ranked by deflated IC t-stat within the admitted set.

### 6.4 Multiple-testing correction

With thousands of candidates per generation, raw thresholds give false-discovery rates in the high tens of percent. Without explicit correction, the gate is a placebo. Three layers, all from v1:

**Layer 1 — Within the CTA gate: Deflated Sharpe Ratio (Bailey & López de Prado 2014).** For each candidate $i$ that passes Stages 1–3:

$$
\mathrm{DSR}_i = \Phi\!\left(\frac{(\widehat{SR}_i - SR^*) \sqrt{T-1}}{\sqrt{1 - \gamma_3 \widehat{SR}_i + \frac{\gamma_4 - 1}{4} \widehat{SR}_i^2}}\right)
$$

where $\widehat{SR}_i$ is the un-annualised Sharpe over $T$ daily observations, $\gamma_3, \gamma_4$ are the third and fourth standardised moments of daily returns, and the deflated benchmark is:

$$
SR^* = \sqrt{V[\widehat{SR}_n]}\, \cdot \, \big[(1 - \gamma) \, \Phi^{-1}(1 - 1/N) + \gamma \, \Phi^{-1}(1 - 1/(Ne))\big]
$$

with $V[\widehat{SR}_n]$ = variance of Sharpes across all $N$ trials in the generation, $\gamma$ = Euler-Mascheroni $\approx 0.5772$. **Reject if $\mathrm{DSR} < 0.95$.**

**Effective N.** When trials are correlated (and they are: most candidates differ by one operator), use López de Prado's ONC clustering or PCA on the trial-return matrix to estimate $N_{\text{eff}}$, and substitute $N_{\text{eff}}$ for $N$ above. **v1 fallback: $N_{\text{eff}} = N$** (most conservative; adds a wider haircut). v2 work: estimate $N_{\text{eff}}$ from the actual trial-return matrix and replace.

**Reference implementation.** CRAN `pbo` package (Bailey-Borwein-López de Prado) and López de Prado's `mlfinlab`. Do *not* implement from this proposal's formula alone — pull the reference and compare your output to its examples.

**Layer 2 — Within the alpha gate: Harvey-Liu-Zhu BHY haircut on IC t-statistic.** For each candidate alpha:

1. Compute IC t-stat: $t_{IC} = \widehat{IC} / \sigma(IC) \cdot \sqrt{n_{\text{periods}}}$.
2. Convert to single-test p-value $p_s$.
3. Apply BHY (Benjamini-Hochberg-Yekutieli) adjustment with the current generation's $M$ trials: $p_{BHY} = p_s \cdot (M \cdot c(M)) / k$ where $k$ is the candidate's rank in ascending p-value order and $c(M) = \sum_{i=1}^M 1/i$. BHY is HLZ's recommended adjustment because it permits arbitrary dependence across tests (Bonferroni is too aggressive; Holm assumes independence).
4. Convert $p_{BHY}$ back to a haircut Sharpe / IC. Harvey & Liu (2015) "Backtesting" gives the formula. Expected haircut magnitude on CSI 300 with thousands of candidates: **substantial**, possibly 70–80% of raw IC. (Verify the exact magnitude against Harvey-Liu Table II/III before quoting it.)
5. **Threshold:** deflated $|t_{IC}| \geq 3.0$ (HLZ's "new factor needs t > 3" cutoff).

**Layer 3 — System-wide: Probability of Backtest Overfitting (PBO) via CSCV** (Bailey-Borwein-López de Prado-Zhu 2017), computed monthly on the joint pool:

1. Concatenate daily PnL of every admitted strategy and every rejected-but-top-quintile candidate into an $T \times N$ matrix.
2. Split rows into $S = 16$ contiguous blocks (long enough to preserve auto-correlation; CSCV's recommendation).
3. For each of the $\binom{S}{S/2}$ symmetric IS/OOS splits, identify the IS-best strategy and compute its OOS rank.
4. PBO = fraction of splits in which the IS-best is below the OOS median.
5. **Target PBO ≤ 0.3.** At PBO ≥ 0.5 the pool is no better than random; tighten the gate.

**Splits everywhere — Combinatorial Purged k-Fold (CPCV) with embargo.** All in-sample/out-of-sample splits inside the gates use CPCV with **embargo = 10 trading days**. The embargo purges label leakage: with a 5-day forward-return label, train and test must be ≥ 5 days apart; 10 is a 2× safety margin, *not* tied to T+1 settlement (T+1 means 1-day stock settlement, not 10-day; the embargo is about label leakage, not settlement). Arian-Norouzi-Seco 2024 show CPCV produces strictly lower PBO and strictly higher DSR than walk-forward on synthetic equity data.

**Implementation cost flag.** DSR + BHY + CPCV + monthly CSCV-PBO is **roughly 1 week of focused work** to implement and validate against published examples. The DSR's $V[\widehat{SR}_n]$ in particular has tripped up many implementers (it's variance across trials in the generation, *not* across folds). Do not skip the validation step.

---

## 7. Pool-level diagnostics

ΔICIR / ΔPool-Sharpe is necessary but not sufficient. Pool-level diagnostics give us the dashboard view that catches failure modes the per-candidate gate misses.

### 7.1 CTA sub-pool diagnostics

- **Pool Sharpe under correlation-adjusted volatility-targeted weighting** (Baltas-Kosowski 2017 dynamic-leverage). *Primary aggregate.*
- **Pairwise signed-return correlation distribution:** median ≤ 0.3, 95th percentile ≤ 0.6.
- **Holding-period diversity:** the pool should contain at least one strategy in each of `{short ~3–10d, medium ~10–30d, long ~30+d}`. SG CTA practitioner finding (CFA Institute 2026 — verify date): horizon-balanced pools have better Calmar.
- **Worst-decile drawdown overlap:** the fraction of days on which more than half the pool is in drawdown ≥ 5% should be ≤ 20%.

### 7.2 Alpha sub-pool diagnostics

- **Combination-model ICIR** (the AlphaGen objective). *Primary aggregate.*
- **AST-similarity diversity:** mean pairwise AST cosine ≤ 0.5; entropy of operator-frequency distribution > $\log K - 1$ (where $K$ is the operator-vocabulary size).
- **Pipeline-source entropy:** shares from Pipeline C vs Pipeline D should not collapse to one pipeline. **Monitor only; do not gate** until the alpha sub-pool reaches ≥ 30 strategies. (At 30+ the 80/20 floor is meaningful; below it, one pipeline genuinely underperforming the other is an *information signal*, not noise to fight.)
- **PCA explained-variance on the pool's IC time-series matrix:** top-3 PCs should not explain > 80% (otherwise the pool is rank-deficient and ΔICIR estimates are noise).
- **Tail mutual information / Kendall's tau in the bottom decile of forward returns:** catches copula-tail dependence that linear correlation misses.
- **Capacity:** average market-cap-weighted ADV consumed by an $X$M deployment of the long-short portfolio. Flag any factor that needs > 1% of CSI 300 daily ADV. *(v2: requires intraday volume data which our v1 daily-bar Qlib install does not have.)*

### 7.3 Joint pool diagnostics (CTA + alpha as P&L streams)

- **Joint-pool annualised Sharpe under equal-risk-contribution weighting.** *Primary aggregate.*
- **Joint CVaR-5%** on daily returns ≤ 2× the worst single-strategy CVaR (no single strategy dominates the joint tail).
- **Drawdown-overlap matrix:** count of strategies simultaneously in 1-month drawdown ≥ 10% must stay ≤ 30% of pool.
- **Regime-coverage matrix:** in each of the 4 regimes the joint pool must have positive Sharpe.
- **Aggregate turnover budget:** sum of post-netting position turnover, with cost = $C \cdot \sum_i |\Delta w_i|$ ≤ 50 bps annualised vs benchmark `«TUNABLE»`.

### 7.4 Monitor-only vs gate-binding (v1 → v2 boundary)

| Diagnostic                                   | v1 status | v2 status |
|----------------------------------------------|-----------|-----------|
| Sub-pool Sharpe / ICIR primary aggregate       | gate-binding via ΔPool-* in §6 | unchanged |
| Pairwise return correlation distribution      | gate-binding (§6 Stage 2)       | unchanged |
| Holding-period diversity                       | monitor                         | gate-binding |
| Worst-decile DD overlap                        | monitor                         | gate-binding |
| AST-similarity diversity                       | gate-binding (§6 Stage 2)       | unchanged |
| Pipeline-source entropy                        | monitor                         | gate-binding above 30 strategies |
| PCA explained-variance                         | monitor                         | gate-binding above 30 strategies |
| Tail mutual info / Kendall's tau              | monitor                         | maybe gate-binding for CVaR cluster |
| Capacity                                       | not computed (no intraday data) | gate-binding |
| Joint-pool ERC Sharpe                          | monitor                         | monitor |
| Joint CVaR-5%, DD overlap matrix              | monitor                         | gate-binding |
| Regime coverage                                | monitor                         | gate-binding |
| Aggregate turnover budget                      | monitor                         | gate-binding |
| CSCV-PBO monthly                               | gate-binding (system-wide)      | unchanged |

The principle: **in v1, every gate-binding metric must be cheap to compute and have a defensible threshold from published practitioner work. v2 metrics are the ones where threshold calibration needs in-house simulation.**

---

## 8. Data foundation

### 8.1 The stack we own

| Layer                    | Tool                              | Why                                                               |
|--------------------------|-----------------------------------|-------------------------------------------------------------------|
| Raw ingest                | AKshare via `data_fetcher`         | Same source we already have wired up                               |
| Storage                   | parquet, partitioned by year/symbol | Portable, columnar, plays well with everything                    |
| Ad-hoc query              | DuckDB                            | Full SQL over parquet without spinning up a DB                     |
| In-memory compute         | Polars (LazyFrame) primarily; pandas where libraries demand it | 5–20× faster than pandas on rolling-window ops |
| Expression engine         | Custom DSL (~300 LoC) + Alpha158 ported on top | We need STL operators that no off-the-shelf engine has |
| Experiment tracking        | MLflow standalone                  | Same MLflow Qlib uses, no Qlib wrapper                              |
| Universe / PIT             | Custom (snapshot CSI 300 history per date from AKshare) | PIT correctness is the #1 quant-research bug source |

### 8.2 Borrowing Alpha158 from Qlib

Qlib's Alpha158 (defined in `qlib/contrib/data/handler.py`) is a list of 158 formulas over `$open / $high / $low / $close / $volume`. Most are rolling stats over windows of {5, 10, 20, 30, 60} days. **Action:** port the formula list into `AutoLLM/alpha158/alpha158.py` and re-implement operators in our DSL. ~half a day of work; gives every pipeline a 158-feature baseline.

### 8.3 The `DataAdapter` protocol

Every pipeline consumes a `DataAdapter`, never `data_fetcher` directly:

```python
class DataAdapter(Protocol):
    def panel(self, universe: list[str], fields: list[str],
              start: pd.Timestamp, end: pd.Timestamp,
              freq: str = "day") -> pl.LazyFrame: ...
    def calendar(self) -> pd.DatetimeIndex: ...
    def universe(self, name: str, on_date: pd.Timestamp) -> list[str]:
        """Point-in-time membership; never returns a future-leaking snapshot."""
    def windows(self, asset: str, start: pd.Timestamp, end: pd.Timestamp,
                length: int) -> Iterator[np.ndarray]:
        """Per-asset windowed slices for the STL miner."""
    def context_bundle(self, on_date: pd.Timestamp) -> dict:
        """Retrieval-style bundle for the LLM Idea Agent."""
```

### 8.4 Folder structure across `AutoLLM/`

```
AutoLLM/
├── proposal_version1.md           ← this file
├── research_prompt_for_claude_chat.md
├── CLAUDE_CODE_BUILD_PROMPT.md
├── CLAUDE.md                       ← project conventions; written by CC on first run
├── data_fetcher/
├── alpha158/
├── STL pipelines/                  ← Pipeline A
├── signal_miner/                   ← shared services (Evaluator, Pool, Gate, MTC)
├── qlib/                           ← read-only github clone, kept for Alpha158 reference
└── NLP and LLM automatic signal Miner/
    └── relative papers/
```

Pipelines B/C/D will be added as sibling folders later (`RL_CTA_pipeline/`, `RL_alpha_pipeline/`, `LLM_alpha_pipeline/`).

---

## 9. Modules to build (v1)

### 9.1 `data_fetcher/`
See `data_fetcher/README.md`. AKshare client, parquet writer (idempotent, partitioned), PIT universe, `AKshareParquetAdapter` implementing `DataAdapter`.

### 9.2 `alpha158/`
Port the 158 formulas. Cross-validate against Qlib reference output (run Qlib in a *throwaway* venv we delete after the validation; we never depend on it at runtime).

### 9.3 `STL pipelines/`
Pipeline A. See `STL pipelines/README.md` and §11 for the full algorithm spec.

### 9.4 `signal_miner/` (shared services)

- `eval/` — EvaluatorService: `(formula | callable, adapter, windows, metric_family) -> MetricsReport`. **CTA-family report (Pipeline A/B):** `Sharpe, Sharpe_shrunk, Sortino, MaxDD, Calmar, CAGR, turnover, time_in_market, equity_R², PR_AUC_long, CVaR_5pct_per_trade, hit_count, IC_TS_per_asset (mean ± std across universe — IC1), ICIR_rolling (rolling 120d window — IC3), conditional_mean_return_per_trade (point-biserial — IC4)`. **Alpha-family report (Pipeline C/D):** `IC_pearson, ICIR, IC_rank, ICIR_rank, long_short_Sharpe, turnover, abs_return, IC_half_life, rolling_IC_per_regime`. Same evaluator, two report shapes; the family selector dispatches to the right metric stack.
- `pool/` — PoolService: parquet store of admitted strategies + sqlite metadata + AST forest.
- `gate/` — Gate-1/2/3/4 as a pipeline of stages, parameterised by metric family. Two concrete instances: `CTAGate` and `AlphaGate`.
- `mtc/` (multiple-testing correction) — DSR, BHY haircut, CPCV-with-embargo, CSCV-PBO. Validate every function against published examples (CRAN `pbo`, López de Prado's `mlfinlab`).

---

## 10. Phased delivery plan

### Phase 0 — Orientation (~half a day)
Set up `~/venvs/myenv` with the lean stack. Read proposal + papers. Write `AutoLLM/CLAUDE.md`.

### Phase 1 — `data_fetcher/` (~3 days)
Build per §9.1. Smoke-pull CSI 300 backfill 2018-01 → today. Write `AKshareParquetAdapter`.

### Phase 2 — `alpha158/` (~half a day)
Port 158 formulas. Cross-validate against Qlib reference (throwaway-venv). 158/158 columns must match within 1e-6.

### Phase 3 — `STL pipelines/` baseline (~2 weeks)
Implement **template-free MCTS** baseline per §11 with the **triple-barrier three-class meta-labelling** pipeline from §11.6 and the $F_A$ fitness from §5.2. Includes:
- §11.6.1 Triple-barrier `LabelComputer` (L1, L5)
- §11.6.2 Magnitude buckets (F6)
- §11.6.3 Sample-uniqueness weights (L2)
- §11.6.4 Primary TSMOM filter (L3)
- §11.4.6 Bayesian-shrunk Sharpe (F1)
- §11.4.3 Pre-fitness gates: hit-count (F2), leakage assertion (E1)
- §11.4.4 PR-AUC (F4) and §11.4.5 CVaR-5% (F5) gate computations
- M3 hierarchical features in §11.3 grammar
- §11.3 base MCTS only (S1 multi-horizon and S2 multi-fidelity follow in Phase 3.5)

Mine 200 candidates per asset on a 50-symbol slice at horizon $H = 5$ first, then validate against the 5-asset MILP baseline (§11.5).

### Phase 3.5 — Multi-horizon, multi-fidelity, transfer-learning warm-start (~1 week)
Add S1 / S2 / S3 per §11.3.2–§11.3.4. Re-mine the same 50-symbol slice across all 4 horizons with multi-fidelity and warm-start enabled. Confirm (a) wall-clock per asset doesn't blow up, (b) different horizons admit different formulas, (c) S3 reduces effective MCTS iterations needed for late assets by ≥ 30 %.

### Phase 3.6 — Pool combination & decay tracking (~3 days)
- M2 BMA combiner: per-asset top-K blended signal at deployment time.
- E4 DecayMonitor: nightly Sharpe re-eval, auto-evict triggers per §11.6.8.
- First end-to-end CSI 300 mine, 4 horizons, full pool admission with all gates (§6.2 base + Pipeline-A-only F2/F4/F5/F1 thresholds).

**Demo:** pool dashboard with first ~150 admitted formulas tagged by (asset, horizon), per-asset BMA-blended equity curves vs buy-and-hold, DecayMonitor's first weekly report.

### Phase 4 — `signal_miner/` shared services (~1 week + 3 days for MTC)
Implement EvaluatorService, PoolService, both gates (`CTAGate`, `AlphaGate`), and the MTC layer (DSR, BHY, CPCV, monthly CSCV-PBO). Validate MTC functions against CRAN `pbo` reference. **Don't skip the validation.**

### Phase 5 — Wire Pipeline A to the CTA gate (~3 days)
Phase 3.6 already produced the first end-to-end pool. Phase 5 layers in the **shared services that Pipelines B/C/D will need**: the generic `EvaluatorService` family-dispatch (§9.4), the alpha-family gate (§6.3), MTC validators (§6.4 — DSR / BHY / CPCV / CSCV-PBO). Compute pool-level diagnostics (§7) for the existing CTA sub-pool to make sure the new shared services agree with what Phase 3.6's bespoke code produced.

### Phase 6+ — Pipelines B, C, D
- B (RL-CTA): fork AlphaGen RL machinery, swap data layer, ~2 weeks. After CTA sub-pool ≥ 5, switch to $F_B^{pool}$ (§5.3).
- C (RL-alpha): same fork base, cross-sectional ops, AlphaGen reward restored, ~1 week.
- D (LLM-alpha): three-agent + MCTS, ~3 weeks.

### Phase 7 — Multi-pipeline orchestration
Scheduled mining jobs, decay scheduling, dashboard, monthly CSCV-PBO recomputation.

---

## 11. Pipeline A baseline algorithm — full spec (Phase 3)

### 11.1 Grammar

```
Formula  := Atomic | Not Formula | Formula And Formula | Formula Or Formula
          | G[a,b] Formula | F[a,b] Formula | Formula U[a,b] Formula

Atomic   := Feature CMP Threshold
CMP      := > | <
Feature  := RawFeature | DerivedFeature                # M3 hierarchical features
RawFeature     := $close | $open | $high | $low | $volume
                | ret_1d | ret_5d | vol_20d
DerivedFeature := <any of the 158 named features in alpha158/alpha158.py>
Threshold:= float (initialised at the rolling-window quantile of the feature)
[a,b]    := temporal interval, integer days, 0 ≤ a ≤ b ≤ 60
```

Bounded depth (`max_depth ≤ 4`) and bounded interval lengths to keep search tractable. **Feature-sampling prior in MCTS**: 70 % `RawFeature` / 30 % `DerivedFeature` early in tree exploration; flip to 50/50 in the second half (§11.3.3 multi-fidelity transition is a natural inflection point).

### 11.2 Quantitative semantics (robustness)

Standard signed-distance interpretation:

```
ρ(x > c, t)             = x(t) − c
ρ(x < c, t)             = c − x(t)
ρ(¬φ, t)                = −ρ(φ, t)
ρ(φ ∧ ψ, t)             = min(ρ(φ, t), ρ(ψ, t))
ρ(φ ∨ ψ, t)             = max(ρ(φ, t), ρ(ψ, t))
ρ(G[a,b] φ, t)          = min_{τ ∈ [t+a, t+b]} ρ(φ, τ)
ρ(F[a,b] φ, t)          = max_{τ ∈ [t+a, t+b]} ρ(φ, τ)
ρ(φ U[a,b] ψ, t)        = max_{τ ∈ [t+a, t+b]} min(ρ(ψ, τ), min_{τ' ∈ [t, τ]} ρ(φ, τ'))
```

All implemented as Polars rolling reductions; no per-row Python loops.

**Pre-processing:** Winsorize features at the 1st/99th percentile per asset per rolling 252-day window before computing robustness — otherwise `min` / `max` over a 60-day window get dominated by a single outlier print.

**v2 upgrade.** Replace standard min/max with **AGM (arithmetic-geometric-mean) robustness** from Mehdipour-Vasile-Belta 2019 when adding TLINet differentiable refinement — AGM gives smooth gradients through the temporal operators.

### 11.3 Search — Template-free Monte Carlo Tree Search with multi-horizon, multi-fidelity, transfer-learning warm-start

v1 search is **MCTS over partial STL formulas**, in the lineage of Mohammadinejad et al. 2023 (AIJ; AAAI 2024 reprint), with three additions: multi-horizon mining (S1), multi-fidelity rollouts (S2), transfer-learning warm-start (S3).

#### 11.3.1 Base MCTS engine

**State.** A partial STL formula AST with one or more open holes (unfilled subtrees). The root state is the single hole `?`.

**Action.** Replace one open hole with one of:
- A primitive atomic predicate `Feature CMP Threshold` (CMP ∈ {>, <}; threshold initialised at the rolling-window quantile of the feature; refinable via small-mutation actions later). **Hierarchical features (M3): the `Feature` slot is sampled from `{raw_OHLCV, ret_1d, ret_5d, vol_20d}` ∪ `{alpha158_*}` — the full Alpha158 feature set is in the action space, with a sampling prior that biases toward simpler raw features early and Alpha158 derived features later.**
- A logical operator `Not ?`, `? And ?`, `? Or ?` introducing one or two new holes.
- A temporal operator `G[a,b] ?`, `F[a,b] ?`, `? U[a,b] ?` with sampled `(a,b)` introducing one or two new holes.
- The terminal action `seal` — refuse to expand further.

**Action-mask discipline.** A hole's typing constrains which actions are legal (predicates take a real-valued feature, logical ops take Boolean subformulas, etc.). Borrowed from Alpha² (#6); ~3× branching reduction.

**Selection.** UCB1: $a^* = \arg\max_a [W/N + c_{\mathrm{UCB}} \sqrt{\ln N(s)/N(s,a)}]$ with $c_{\mathrm{UCB}} = \sqrt{2}$ default `«TUNABLE»`.

**Expansion.** Sample a legal action; uniform-over-legal default for v1; LLM-policy prior (à la Alpha-Jungle) is a v2 upgrade.

**Rollout.** Random completion to termination, compute $F_A$ (per §5.2 and §11.3.3 multi-fidelity below).

**Backup.** Standard MCTS — propagate $F_A$ up, increment visit counts.

**Frequent-subtree avoidance.** Track empirical frequency of subtrees of size ≥ 3; additive penalty `−λ_subtree` when frequency > 25 % of visited nodes. Stops MCTS collapsing onto one motif.

**Reproducibility.** Seedable RNG, deterministic scoring, deterministic action ordering at ties.

#### 11.3.2 Multi-horizon mining (S1)

**Four parallel populations per asset, mined independently** at horizons $H \in \{3, 5, 10, 20\}$ trading days. Each horizon produces its own triple-barrier label panel (the barrier-clock in §11.6 uses $H$ as the time-out leg, and barrier widths scale as $\sigma_t \cdot \sqrt{H/5}$ relative to the 5-day baseline). Each population runs its own MCTS to completion. Each horizon's top-K candidates pass through the gate independently and enter the pool independently, **tagged with their horizon** in the audit row.

Why we do this: different STL formulas have different natural horizons (a `G[0,2] (mom_5d > 0)` is a 3-day game; a `F[0,15] (vol < 0.4 × vol_60)` is a 20-day game). Forcing a single horizon throws this away. Cost is essentially nothing because MCTS is already per-asset embarrassingly parallel — we just spawn 4× more processes (sharded across the universe in batches to fit memory).

#### 11.3.3 Multi-fidelity rollouts (S2)

Within a single MCTS run, fitness evaluation is the bottleneck (~70 % of wall-clock). Two-tier evaluation:

| Tier      | Sample fraction                              | When used                                             |
|-----------|----------------------------------------------|-------------------------------------------------------|
| **Cheap** | 30 % of train days, stratified by class      | first 70 % of MCTS iterations (`< 35 000` of 50 000)   |
| **Full**  | 100 % of train days                          | last 30 % of MCTS iterations + final top-K re-scoring  |

Stratification is by triple-barrier label *and* sample-uniqueness weight bucket so the cheap evaluator gets a balanced view. The transition between tiers happens at iteration 35 000 with a re-evaluation of the current top-100 candidates on full data; the pruned set is what the late-stage MCTS UCB acts on.

**Bias risk.** Cheap-tier rollouts have ~3× the noise of full-tier; the late-stage re-evaluation is critical so we don't lock in over-fitting to the cheap subsample. The 70/30 split is `«TUNABLE»` against the actual variance we observe in Phase 3.

#### 11.3.4 Transfer-learning warm-start (S3)

Pre-cluster the universe by industry sector (CSI 300 sector tags from AKshare) into ~30 sector groups. For each asset $b$, identify its **3 nearest neighbours** by sector and by realised-vol regime over the last 252 trading days. When MCTS starts on $b$, **seed the tree's first 5 000 visits** with the top-K admitted formulas mined on $\{a_1, a_2, a_3\}$, biased toward their root-level expansions:

```
seed_tree(asset_b):
    neighbours = nearest_neighbours_by_sector_and_vol(b, k=3)
    seed_formulas = {top_K admitted on each neighbour}
    for f in seed_formulas:
        play_out(f, n_visits=⌈5000 / |seed_formulas|⌉)
    # MCTS continues from this state with full visits/value statistics
```

This is *informed* warm-start, not constraint — the MCTS is free to discard seeded paths via UCB if their actual fitness on $b$ is poor. The seeds just lower the exploration budget needed.

**Negative-transfer risk.** A neighbour's top formula may be over-fit to that neighbour's idiosyncrasies. Mitigation: only seed from neighbours whose admitted formulas pass *their own* §6.2 gate; never seed from quarantined candidates.

**Cold-start.** First 30 % of the universe (or any asset whose nearest neighbours have no admitted formulas yet) skips warm-start and runs vanilla. After 30 % of the universe is mined, warm-start kicks in for the rest.

#### 11.3.5 Iterations, parallelism, output

**Iterations.** 50 000 per asset per horizon (4 horizons × 50 000 = 200 000 effective per asset). With multi-fidelity at 70/30, this is roughly equivalent to 35 000 full-fidelity rollouts wall-clock-wise. Estimate: ~5–10 minutes per asset per horizon on M1 with 8 worker processes.

**Wall-clock budget.** 300 assets × 4 horizons × ~7 min average ≈ 140 hours single-threaded, ≈ **18 hours wall-clock with 8 workers** for a full CSI 300 mine. Doable on the M1 over a weekend or via incremental nightly runs.

**Output.** Top-K = 200 per (asset, horizon), deduplicated by AST hash and structural similarity ≥ 0.7.

**Parallelism.** Per-(asset, horizon) process pool of size ≤ 8. Do not compose Polars threading with `multiprocessing.Pool` of size > 8 (Polars saturates the M1's 8 perf cores by itself).

### 11.4 Fitness implementation

Each component of $F_A$ is its own pure function. The top-level orchestrator is not allowed to recompute components inline.

#### 11.4.1 Robustness panel

- `robustness_panel(formula, feature_panel) -> ρ_panel` — vectorised Polars, no per-row Python.

#### 11.4.2 Three-class margin (with sample-uniqueness + magnitude weights)

- `class_margin(rho_panel, label_panel, sample_weights, magnitude_weights, target_class) -> float` — returns `weighted_mean(ρ | y = target) − weighted_mean(ρ | y ≠ target)`. `target_class ∈ {long, short}`. Magnitude-weighted (F6) for the long class.
- `pooled_rho_std(rho_panel, sample_weights) -> float`.
- `false_positive_rate(rho_panel, label_panel, sample_weights) -> float` — `weighted_mean(1[ρ > 0] | y ≠ long)`.

#### 11.4.3 Pre-fitness gates (§5.6 1–4)

- `hit_count_long(rho_panel, label_panel) -> int` — number of long-class days with ρ > 0. F2 gate: `>= 30` or fitness = `-inf`.
- `assert_no_label_leakage(formula, label_rule_max_window) -> None | raise` — E1 gate; runs once at formula compile time, not per-rollout.
- `complexity(formula) -> int`.
- `class_distribution(rho_panel, label_panel) -> dict[class, fraction]` — used to detect trivial-class collapse.

#### 11.4.4 PR-AUC for long class (F4)

- `pr_auc_long(rho_panel, label_panel, sample_weights) -> float` — sklearn's `precision_recall_curve` plus `auc`, with `sample_weight` propagated. Threshold sweep over ρ. **Gate threshold ≥ 0.40, applied after MCTS top-K selection — not in fitness.**

#### 11.4.5 Per-trade CVaR-5 % (F5)

- `realised_trade_returns(rho_panel, return_panel, cost_bps) -> trade_returns_array` — returns per-trade post-cost returns where each "trade" is a contiguous run of `position = 1`. Position rule: `position_t = 1 if ρ > 0 AND primary_signal_t == long else 0`.
- `cvar_5pct(trade_returns) -> float` — mean of the worst 5 % of trades. **Gate threshold ≥ −1 %, not in fitness.**

#### 11.4.6 Bayesian-shrunk Sharpe (F1) — used in fitness as small tiebreak, also as gate

- `realised_sharpe(rho_panel, return_panel, cost_bps) -> float` — annualised Sharpe of the realised long-only PnL, sample-uniqueness-weighted.
- `sharpe_t_stat(sharpe, n_observations) -> float` — un-annualised Sharpe × √n.
- `bayesian_shrink_sharpe(sharpe_raw, t_stat, prior_variance) -> float` — James-Stein shrinkage:
  ```
  shrinkage_factor = k / (t_stat^2 + k)
  sharpe_shrunk    = sharpe_raw * (1 - shrinkage_factor)
  ```
  where `k` is calibrated from the cross-sectional distribution of empirical Sharpes within the current MCTS generation (use the prior_variance estimate). Default `k = mean_t_stat^2` if prior is unknown.

#### 11.4.7 Top-level orchestrator

- `pipeline_a_fitness(formula, panel, label_panel, sample_weights, magnitude_weights, lambdas, gate_thresholds) -> float | -inf` — runs pre-fitness gates first; if any fail, returns `-inf`. Otherwise computes:
  ```
  F = (m_long + α_short * m_short) / σ_ρ - λ_cplx * complexity - λ_FP * FPR + λ_S * Sharpe_shrunk
  ```
  with all means using `sample_weights × magnitude_weights`.

Every function independently unit-tested with at least one hand-computed example.

### 11.5 Validation against MILP baseline

For sanity, Phase 3 also runs a **5-asset MILP** (paper #4 — Optimal STL Decision Trees) on a 1-year window with the *same positive-only labels*, and confirms the MCTS top-1 reaches ≥ 80% of the MILP optimum's separation margin on the same window. (We compare on the margin term of $F_A$, not on Sharpe — the MILP optimises classification-style separation directly, so margin is the apples-to-apples comparison.)

### 11.6 Data processing pipeline — labels, weights, splits, primary model

The v1 mining mode is **triple-barrier three-class meta-labelling**. The labelling rule, the primary model, the sample-uniqueness weights, the magnitude weights, and the train/val/test discipline are all part of the data-processing pipeline; they are not the optimiser's problem.

#### 11.6.1 Triple-barrier three-class labels (L1, L5)

For each (asset, $t$) sample on which the *primary model* (§11.6.4) emitted a "long" candidate signal, define three barriers:

```
σ_t            = realised vol of (log_return) over trailing 20 days   # vol-scaled barriers
upper_barrier  = +p_upper × σ_t × √H        # profit-take, e.g. p_upper = 2.0
lower_barrier  = -p_lower × σ_t × √H        # stop-loss,    e.g. p_lower = 1.0
time_out       = H trading days             # horizon-dependent (S1)

label y_t ∈ {long, flat, short}:
    walk forward day by day from t+1 to t+H
    first barrier hit:
        upper  →  y_t = long
        lower  →  y_t = short
    if neither hit by t+H:
        y_t = flat
```

For the multi-horizon mining (S1), $H \in \{3, 5, 10, 20\}$ produces four label panels per universe; barriers scale as $\sigma_t \cdot \sqrt{H/5}$ relative to the 5-day baseline so the in-sample positive rate stays roughly comparable across horizons.

**Defaults `«TUNABLE»`:** $p_{\text{upper}} = 2.0$, $p_{\text{lower}} = 1.0$ (a 2:1 reward-to-risk profile; trend-following has positive skew, so this is reasonable). All three parameters live in config.

**Expected class distribution on CSI 300** (rough, regime-dependent): ~25 % long, ~25 % short, ~50 % flat (within the meta-label subset, i.e. days the primary said long). Well-balanced enough that we don't need aggressive class-balancing.

#### 11.6.2 Magnitude buckets within the long class (F6)

Each long-class sample also carries a **magnitude bucket** based on the realised return at the upper barrier hit:

```
return_at_barrier = price at upper_barrier_day / price at t - 1
bucket =  1.0   if return_at_barrier ∈ (p_upper×σ_t,  2×p_upper×σ_t]
       =  1.5   if return_at_barrier ∈ (2×p_upper×σ_t, 3×p_upper×σ_t]
       =  2.0   if return_at_barrier  >   3×p_upper×σ_t
```

These buckets become **magnitude weights** in the fitness's long-class margin (§5.2). Short-class and flat-class samples carry magnitude weight 1.0.

#### 11.6.3 Sample-uniqueness weights (L2)

Per López de Prado *AFML* ch. 4. For each sample $(asset, t)$:

```
overlap_count_t = number of other samples in the same asset whose
                   forward-window [s, s+H_max] contains t
sample_weight_t = 1 / (1 + overlap_count_t)
```

Where $H_{\max} = 20$ (the longest horizon we mine at). Weights are computed once per (asset, label-rule) combination and cached to `~/AutoLLM_data/sample_weights/{rule_hash}/{universe}/{asset}.parquet`.

**Every** weighted mean / correlation / Sharpe / IC / DSR / PR-AUC / CVaR in the system is computed with these weights. Without them, daily samples with overlapping forward-windows are double-counted and the DSR's i.i.d. assumption silently fails.

**Effective sample size estimate:** with $H_{\max} = 20$, the average overlap count is ~ 19, so $N_{\text{eff}} \approx N / 20$. This single observation is what drives the DSR deflation in §6.4 from "mild" to "real".

#### 11.6.4 Meta-labelling primary model (L3)

The MCTS miner is trained as a **meta-labeller**: given a binary "long" candidate signal from a deterministic primary, predict whether to take it.

**v1 primary model (deterministic):**

```python
primary_signal_t = (
    momentum_20d(close, t)  > 0
    AND
    realised_vol_20d(log_returns, t) < median(realised_vol_20d, last 60 days)
)
```

This is a vanilla TSMOM-with-low-vol-filter primary. ~10–25 % of days fire, depending on regime. **Only those days are eligible for meta-labelling**; on the rest, the system trades nothing regardless of what STL says.

**v2 alternative primaries** (deferred; same plug-point):
- Cross-sectional: long if asset is in CSI 300 top-quintile by 20d momentum
- Regime-aware: TSMOM filter conditional on a regime classifier
- Stitched: union of multiple primaries, MCTS picks among them

**Why meta-labelling.** The MCTS search space is collapsed by ~5× (only ~20 % of days are eligible), the labels are cleaner (the primary already filtered out mostly non-actionable days), and Pipeline A's contribution is isolated to the entry-quality question — it learns *when not to take* a primary signal.

#### 11.6.5 Label panel materialisation and caching

Compute labels and sample-weights once per (universe, label_rule, primary_rule) combination, cache to:

```
~/AutoLLM_data/labels/{label_rule_hash}/{primary_rule_hash}/{universe}/H={H}/{asset}.parquet
```

Hash of every config knob is part of the path so changing any rule produces a new cache, never silently overwrites. `LabelComputer` takes `label_rule`, `primary_rule`, and `H`, hits the cache.

#### 11.6.6 Train / validation / test splits

Walk-forward CPCV with embargo, per the multiple-testing-correction stack in §6.4:

- **Training window:** 3 years.
- **Validation window:** 6 months (used for $F_A$ during MCTS).
- **Test window:** 6 months (held back; used only at gate-time for Sharpe / Sortino / PR-AUC / CVaR on the realised position panel).
- **Embargo:** 10 trading days (covers the 20d max forward-window plus safety margin via the sample-uniqueness weighting).
- **Walk-forward cadence:** advance by 6 months per fold; 5 folds total over the 5-year backfill.

#### 11.6.7 The pipeline graph

```
data_fetcher  →  DataAdapter
                     │
        ┌────────────┴───────────────┐
        │                            │
        ▼                            ▼
   PrimaryModel               FeaturePanel
   (TSMOM filter)             (raw + Alpha158)
        │                            │
        ▼                            │
  primary_signal_t                   │
        │                            │
        ▼                            │
   TripleBarrierLabelComputer(H)     │
        │                            │
   y_t ∈ {long,flat,short},         │
   magnitude_weights,               │
   sample_weights                   │
        │                            │
        └─────────────┬──────────────┘
                      ▼
              MetaLabelEvaluator
              (computes F_A per §5.2)
                      │
                      ▼
              MCTS engine (§11.3)
              4 horizons × N assets
                      │
                      │ Top-K per (asset, horizon)
                      ▼
              Pre-fitness gates: F2, E1
                      │
                      ▼
              CTA gate (§6.2): Sharpe/Sortino/etc.
              + F4 PR-AUC + F5 CVaR + F1 shrunk Sharpe
                      │
                      │ admitted candidates, tagged by horizon
                      ▼
              PoolService (CTA sub-pool)
                      │
                      ▼
              BMA combiner (M2) — per-asset top-K blend
                      │
                      ▼
              DecayMonitor (E4) — nightly Sharpe re-eval
```

This is the v1 contract. Pipeline B will share everything from `MetaLabelEvaluator` downward (different miner, same evaluation/admission/pool stack).

#### 11.6.8 Pool combination and decay tracking

**M2 — Bayesian model averaging at deployment.** The pool admits individual formulas; at deployment time, the per-asset signal is a BMA-weighted blend of the asset's top-K admitted formulas:

```
weight_i ∝ exp( BIC_i / 2 ) × prior_i
signal_t(asset) = Σ_i  weight_i  ×  ρ_i(asset, t)
position_t(asset) = 1  if signal_t(asset) > 0  AND primary_t(asset) = long  else 0
```

where BIC = Bayesian Information Criterion estimate from the formula's in-sample log-likelihood under a no-skill noise prior, and `prior_i` is uniform unless we have reasons to bias toward simpler formulas. K = 5 default `«TUNABLE»`.

**E4 — DecayMonitor.** Every admitted formula has its rolling 60-day realised Sharpe (sample-weighted) recomputed nightly on whatever market data has accrued since admission. Auto-evict triggers:

| Trigger                                                 | Action                                |
|---------------------------------------------------------|---------------------------------------|
| Rolling 60d Sharpe < `0.3 × admission_Sharpe`            | demote to Watch list (re-evaluate weekly) |
| Rolling 60d Sharpe < 0 for 3 consecutive weeks           | auto-evict from pool                  |
| MaxDD > `1.5 × admission_MaxDD`                          | demote to Watch list                  |
| Sample-uniqueness-adjusted t-stat of recent PnL < 1.0    | demote to Watch list                  |

Evicted formulas are not deleted — they go to a `decay_history` table for post-hoc analysis.

#### 11.6.9 Other label modes for v2+

The architecture (`LabelComputer` as a swappable module, `F_A` as a function of `(ρ_panel, label_panel, sample_weights, magnitude_weights)`) accommodates the following without rework:

| Mode                                      | When to add | Notes                                                          |
|-------------------------------------------|-------------|----------------------------------------------------------------|
| Triple-barrier three-class meta-label (v1)| now         | this section                                                    |
| Negative-only (mirror)                    | v2          | for short-side rules; needs short-sale availability            |
| Continuous-return regression              | v2          | replaces three-class with continuous label; switches $F_A$ to Sharpe-driven |
| Regime-conditional                        | v3          | label rule conditional on a regime classifier                  |
| Drawdown-tolerant                         | v2          | relax the lower barrier; for momentum-only environments         |
| Cross-sectional rank label                | v2          | for Pipeline C-flavoured cross-sectional STL (currently deferred — cross-sectional STL is unusual) |

---

## 12. Borrow / adapt / skip — quick reference

|  Source                                | Borrow                                                  | Adapt                                                   | Skip                                                   |
|----------------------------------------|---------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------|
| #1 RAMTL                               | template-retrieval *idea*                                | small custom embedding instead of theirs                | Bayesian-opt heavy machinery                           |
| #2 Neural-STL                          | conceptual grounding                                     | —                                                       | toy CDC examples                                       |
| #3 TLINet                              | smooth-min/max formulas                                  | as Polars / torch modules; v2                            | their structure-learning (we use MCTS in v1, differentiable refinement only in v2) |
| #4 Optimal STL Decision Trees (MILP)   | 5-asset baseline for sanity-check                        | restrict to ≤10 assets                                  | as the main miner                                      |
| #5 AlphaGen                            | RPN, action mask, 6-feature input, 20-day label          | swap Qlib data layer for our adapter                    | their qlib-coupled data path                           |
| #6 Alpha²                              | legal-action-mask discipline                              | —                                                       | full AlphaZero stack                                   |
| #7 QFR                                 | variance-bounded REINFORCE baseline + IR-shaped reward    | re-implement (~30 lines)                                | full benchmark suite                                   |
| #8 Alpha-GPT                           | hypothesis-first prompt structure                         | autonomy (no human-in-loop)                             | their UI                                               |
| #9 AlphaAgent                          | Idea/Factor/Eval split + AST-similarity + complexity reg  | EvaluatorService swap                                   | their hard-coded backtest module                       |
| #10 Alpha-Jungle (LLM-MCTS)            | LLM-as-prior + frequent-subtree avoidance                | re-implement from paper                                 | —                                                      |
| **Qlib (no paper)**                    | Alpha158 formula list                                    | port to our DSL                                         | Workflow/Handler/Dataset/Recorder/Backtest/Strategy    |
| **Lim-Zohren-Roberts 2019** (Deep Momentum Networks)| Sharpe-loss training; turnover regularisation idea | —                                                       | the LSTM architecture                                  |
| **Hurst-Ooi-Pedersen "Century of Trend"** | Sortino > Sharpe motivation                          | Sortino term in F_B                                     | —                                                      |
| **Baltas-Kosowski 2017**               | correlation-adjusted vol-targeted weighting for CTA pool | as the pool-Sharpe weighting                            | —                                                      |
| **Bailey-López de Prado 2014**         | Deflated Sharpe Ratio formula                            | use CRAN `pbo` as reference impl                        | —                                                      |
| **Harvey-Liu-Zhu 2016**                | t > 3 cutoff + BHY-haircut on IC t-stat                  | applied per-generation                                  | —                                                      |
| **Bailey-Borwein-LdP-Zhu 2017** (PBO/CSCV) | monthly system-wide PBO                              | $S = 16$ blocks                                          | —                                                      |
| **Arian-Norouzi-Seco 2024**            | CPCV with 10-day embargo for all splits                  | embargo length sized to label window                    | —                                                      |
| **Mehdipour-Vasile-Belta 2019** (AGM robustness) | smooth STL robustness for v2                  | replace standard min/max in TLINet refinement           | —                                                      |
| **Mohammadinejad et al. 2023** (Anchors + MCTS for STL, AIJ / AAAI 2024 reprint, DOI 10.1016/j.artint.2023.103905) | **Pipeline A v1's MCTS-over-STL-AST machinery** — node = partial formula, action = expand a hole with a primitive, reward = margin separation in our re-targeting | Re-target their anchor-precision reward to our $F_A$ margin separation; keep their typing-mask, action set, expansion priors | their explanation-of-decision motivation (we're mining trade entries, not explanations); their hybrid-control / ACAS-Xu domains (we're on CSI 300) |

---

## 13. Risk register

| Risk                                                        | Likelihood | Impact | Mitigation                                                 |
|-------------------------------------------------------------|-----------|--------|-------------------------------------------------------------|
| EvaluatorService numbers diverge from any external benchmark   | M         | H      | Property-based tests in Phase 0; Phase 2 cross-validation against Qlib's Alpha158 to 1e-6 |
| MTC implementation subtly wrong (DSR / BHY / CPCV)            | H         | H      | Validate against CRAN `pbo` and López de Prado `mlfinlab` on every function before relying on it |
| Custom DSL operator bug → silently bad signals                 | M         | H      | Hand-test each operator on a 10-day toy series              |
| AKshare endpoint changes / rate-limits                         | M         | M      | Bridge handles both Chinese/English column aliases; configurable backoff |
| LLM pipeline cost blows up (Pipeline D)                        | H         | M      | Local Qwen for Factor Agent; rate-limit Idea Agent          |
| Pool fills with near-duplicates                                 | M         | H      | Both AST-similarity and return-correlation gates required at Stage 2 |
| Universe survivorship bias                                       | H         | H      | PIT membership built into `data_fetcher` from day one         |
| Look-ahead in label                                              | M         | H      | EvaluatorService asserts no overlap of label window and feature window; CPCV embargo = 10 days |
| Sharpe ≥ 0.8 + DSR ≥ 0.95 turns out too restrictive (no admits)  | M         | M      | Calibration: monitor admit rate after 1 month; drop DSR threshold to 0.90 first, then drop Sharpe to 0.7 only if still empty |
| $N_{\text{eff}} = N$ fallback in DSR is over-conservative       | M         | M      | v2 work: implement ONC clustering on trial-return matrix |
| Win-rate range / turnover thresholds wrong for CSI 300 vs US futures | M    | M      | All marked `«TUNABLE»`; recalibrate against the real Phase 3 backtest distributions |
| **M1 16 GB memory pressure** when running MCTS on CSI 300 in parallel | M | M | 8 worker processes × ~1 GB resident = 8 GB; leave headroom by sharding the universe (e.g. 50 symbols at a time × 6 batches) rather than spawning 300 workers. Polars' default thread pool already saturates at the M1's 8 perf cores; do NOT compose Polars threading with `multiprocessing.Pool` of size > 8. |
| **Positive-class label too sparse** if the default `forward_5d > 2% AND max_DD < 1%` rule yields < 3 % of days | M | M | LabelComputer reports the empirical positive rate at materialisation time; if < 3 %, log a warning and either (a) loosen the rule via config, or (b) extend the forward window to 10 days. |
| **MCTS collapses onto one motif** | M | M | Frequent-subtree avoidance (§11.3) is on by default; if it still happens, raise $\lambda_{\text{subtree}}$ or lower the visit threshold for triggering the penalty. |
| **Triple-barrier parameter calibration** ($p_{\text{upper}}, p_{\text{lower}}, H$) wrong for CSI 300 | M | M | All three are `«TUNABLE»`. Phase 3 produces empirical class-distribution diagnostics; if the long class is < 15 % or > 40 %, retune. |
| **Sample-uniqueness weights propagation bugs** (a metric uses raw N instead of $N_{\text{eff}}$) | H | H | Single source of truth: every metric flows through `signal_miner/eval/weighted_stats.py`, which always takes a `sample_weights` argument. Property-based test: synthetic data with known overlap → weighted Sharpe matches analytical formula. |
| **Meta-labelling primary model wrong** | M | M | The TSMOM-with-vol-filter primary is the *least* surprising choice; if it's biased, the meta-label is biased the same way. Sanity check in Phase 3: report meta-label class distribution per regime; if dramatically imbalanced, change the primary. |
| **Multi-fidelity bias** (cheap subsample over-fits to its 30 % stratification) | M | M | Late-stage re-evaluation on full panel at iter 35k re-orders top-100 candidates. If the order changes drastically (Spearman < 0.6), bump cheap-tier sample fraction up. |
| **Negative transfer** in S3 warm-start | L | M | Only seed from neighbours whose admitted formulas pass *their own* §6.2 gate. Cold-start first 30 % of universe vanilla. |
| **Bayesian shrinkage too aggressive on real data** | M | L | The shrinkage `k` is calibrated to the cross-sectional Sharpe variance in the current generation; if Phase 3 shows it's killing low-firing-but-real edges, lower `k` or report `Sharpe_raw` next to `Sharpe_shrunk` so we can compare. |
| **Hierarchical features (M3) inflate the action space and slow MCTS** | M | M | Sampling prior weights raw OHLCV features higher than Alpha158 derivative features early in tree exploration, lower later. If MCTS still doesn't find good Alpha158-using formulas, bump the late-stage prior weight. |
| **BMA combiner over-blends** (averages away the best formula's signal) | L | M | Default K = 5; lower to K = 3 if the BMA-blended Sharpe consistently underperforms the asset's top-1 formula's Sharpe by > 15 %. |
| **DecayMonitor false-positive evictions** in regime change | M | M | Eviction trigger requires 3 consecutive weeks of negative Sharpe — gives ~2 months of regime-shift tolerance. If still too sensitive, lengthen window. |

---

## 14. What to do this week

1. Open `AutoLLM/CLAUDE_CODE_BUILD_PROMPT.md` and follow it on the Mac. It is designed to be pasted into Claude Code at the `AutoLLM/` working directory and run as a single conversation.
2. **Spot-check three claims before they harden into v1 calibration anchors:**
    1. The Qlib LightGBM-on-Alpha158 baseline IC numbers (≈ 0.040 / 0.41 / 0.51) — verify against `qlib/examples/benchmarks/LightGBM/README.md`.
    2. The Harvey-Liu-Zhu Backtesting paper's Table II/III haircut magnitudes — confirm the "70–80%" expectation for CSI 300 generation sizes.
    3. The CFA Institute "Decoding CTA Allocations by Trend Horizon" piece allegedly dated 28 Jan 2026 — confirm it exists.
3. After Phase 1 (`data_fetcher`) lands, smoke-test it manually with a 5-symbol pull and a 5-symbol panel read.
4. After Phase 3 (`STL pipelines/` baseline) lands, run the demo notebook and write three sentences on what surprised you. Append them to §0 decision log as a new row.
5. After Phase 4 (signal_miner with MTC), run the validation suite for DSR, BHY, CPCV, and CSCV-PBO against the published reference examples. **Do not start admitting strategies from Phase 5 onwards until Phase 4 validation passes.**

---

## 15. Compute environment — what runs locally on the M1 vs what runs on Colab

The user's hardware is a **MacBook with M1 chip, 16 GB unified memory, 512 GB SSD**. We design v1 to fit; we plan v2 to overflow honestly.

### 15.1 Runs comfortably on the M1 (v1)

| Workload                                                           | Memory peak | Wall-clock (CSI 300, 5y backfill) | Notes |
|--------------------------------------------------------------------|-------------|-----------------------------------|-------|
| AKshare data backfill                                              | ~ 0.5 GB    | ~ 30 min one-time, < 5 min daily incremental | I/O-bound |
| Parquet / Polars / DuckDB stack                                    | 1–2 GB working set | Sub-second on cached features | Polars is the right call |
| Alpha158 port + cross-validation against Qlib                       | ~ 1 GB       | ~ 5 min                           | Qlib runs in throwaway venv, then deleted |
| **Pipeline A — MCTS, 50 k iters per asset, 8 workers in parallel** | ~ **8 GB**   | ~ **24–48 hr full universe**      | The headline laptop workload. Sharded across 6 universe-batches of 50 symbols. |
| Estimation Factory: gates 1–4 + DSR + BHY + CPCV                   | ~ 1 GB      | seconds per candidate, minutes per generation | Pure Polars + scipy |
| Monthly CSCV-PBO on the joint pool                                 | ~ 2 GB      | ~ 30 min                          | Once a month |
| MLflow standalone for experiment tracking                          | ~ 0.5 GB    | continuous, idle                   | |

**Grand total when everything runs:** ~10–12 GB peak. Leaves 4–6 GB for OS + browser + Cursor + the rest of the user's actual computing life. This is the budget v1 must live within.

### 15.2 Runs with care on the M1 (smallish DL — v2 candidates)

| Workload                                  | Why it's borderline | What to do |
|-------------------------------------------|---------------------|------------|
| Pipeline B — RL CTA (PPO + QFR) with small MLP policy | PyTorch on MPS works; fp32, batch 256, MLP policy with ≤ 3 hidden layers at 256 units fits in ~3 GB | Run on M1 if no other heavy workloads; otherwise → Colab |
| Pipeline C — RL alpha (AlphaGen reproduction) | Their default network is small enough; their default training run is ~ 12 hr on a single GPU | M1 will be ~ 3–5× slower → realistically Colab |
| Pipeline D — LLM Factor Agent (local Qwen-7B int4) | int4 inference is ~ 5 GB resident, ~ 10 tok/sec on M1 | Fine for low-volume use; don't run alongside Pipeline A |
| TLINet differentiable refinement of STL (v2 of Pipeline A) | Small differentiable model; gradients through robustness | M1 fine; sub-15-min for typical refinement runs |

### 15.3 Needs Colab / cloud (v2+)

| Workload                                  | Why                                                 |
|-------------------------------------------|-----------------------------------------------------|
| Pipeline B + C trained simultaneously      | Two PyTorch processes blow the memory budget         |
| LLM Idea Agent (Pipeline D) at frontier-model scale | Frontier-model inference doesn't fit locally; use Claude / GPT API |
| Hyperparameter sweeps over MCTS / GP / PPO  | Embarrassingly parallel; cheap on free Colab tier   |
| Long backtests on > CSI 500 universe       | RAM footprint of the (asset × time × feature) panel scales linearly; > 1000 names exceeds 16 GB |
| Replicating AlphaAgent on S&P 500 + CSI 500 simultaneously | Their published experiment; replication is v3 work |

### 15.4 The Colab handshake pattern

When a v2 pipeline goes to Colab, the contract with the laptop is:

1. Laptop pushes the **canonical input bundle** (DataAdapter snapshot for the relevant universe + train/val/test windows + the relevant `LabelComputer` output) to Google Drive as a single tarball.
2. Colab pulls the bundle, runs the workload, writes outputs back to Drive: trained model artifacts + per-candidate score parquets.
3. Laptop pulls the outputs and feeds them through the **same gates and pool** as the local pipelines. The Estimation Factory does not care which machine produced a candidate.

This keeps the Pool as the single source of truth; no part of the gate or admission logic ever runs in Colab. That's deliberate — Colab is for *training*, the laptop is for *judging*.

---

## Appendix A — open questions the research review surfaced

These are deliberately *not* in the proposal body because they are calibration questions to be settled empirically, not architectural ones. They each belong on the Phase 5+ ablation schedule.

1. **Does $F_B^{pool}$ (marginal-pool-Sharpe) beat $F_B$ as a search reward in CTA space?** AlphaGen showed pool-incremental beats single-instance for cross-sectional alphas, but the analog for CTA pools is not in the public literature. Replicate AlphaGen's Q3 ablation in CTA space.
2. **Do AST-similarity novelty regularisers transfer from cross-sectional alphas to per-asset CTA formulae?** AlphaAgent's evidence is on cross-sectional factors. Probably yes, but the ablation belongs on Pipeline B's experimental schedule.
3. **What is the effective number of independent trials $N_{\text{eff}}$ inside one Mining Factory generation?** Run López de Prado's ONC clustering on the actual trial-return matrix once Phase 5 is producing real candidates. Replace the conservative $N_{\text{eff}} = N$ fallback if $N_{\text{eff}} / N$ turns out to be small.
4. **Should the joint pool gate be hard-binding (rejects candidates that push joint CVaR above threshold) or stay monitor-only?** v2 question; depends on appetite for crowded-factor risk after a 1-year shadow period.
5. **Where does Alpha-Jungle's frequent-subtree avoidance belong — reward term or prompt-time prior?** v1 has both; v2 should ablate to one.
6. **Should RAMTL's interval-trajectory robustness replace the standard $\rho$ in $F_A$?** Worth a directed test on Pipeline A given small CSI 300 sample.

---

## Appendix B — what changed in this revision

This revision integrates **19 user-selected upgrades** beyond the previous revision: L1 (triple-barrier), L2 (sample-uniqueness), L3 (meta-labelling), L5 (three-class), F1 (Bayesian-shrunk Sharpe), F2 (min-hit-count), F4 (PR-AUC), F5 (per-trade CVaR), F6 (magnitude buckets), S1 (multi-horizon), S2 (multi-fidelity MCTS), S3 (transfer warm-start), M2 (BMA combiner), M3 (hierarchical features), E1 (leakage assertion), E4 (DecayMonitor), IC1 (time-series IC), IC3 (rolling ICIR), IC4 (point-biserial / conditional mean per trade).

| Section | Change                                                                                       |
|---------|----------------------------------------------------------------------------------------------|
| §0      | Six new decision-log rows: triple-barrier three-class labels (L1+L5), sample-uniqueness everywhere (L2), meta-labelling (L3), multi-horizon mining (S1), hierarchical features (M3), BMA at deployment (M2) |
| §4      | Pipeline A description rewritten end-to-end to reflect all 19 items                          |
| §5.2    | **F_A rewritten as three-class margin** with magnitude weighting (F6), sample-uniqueness everywhere (L2), Bayesian-shrunk Sharpe tiebreak (F1) |
| §5.6    | **New section** — Pipeline A pre-fitness gates (F2, E1, complexity, trivial-class collapse) and post-fitness admission gates (F4 PR-AUC ≥ 0.40, F5 CVaR-5 % ≥ −1 %, F1 shrunk Sharpe ≥ 0.6) |
| §6.2    | Added Pipeline A-only rows: hit-count, PR-AUC, CVaR-5%, shrunk-Sharpe                         |
| §9.4    | EvaluatorService report shape extended: CTA family now includes Sharpe_shrunk, PR_AUC_long, CVaR_5pct_per_trade, hit_count, IC_TS_per_asset (IC1), ICIR_rolling (IC3), conditional_mean_return_per_trade (IC4) |
| §10     | Phase 3 expanded to ~2 weeks for the v1 stack; **new Phase 3.5** for multi-horizon / multi-fidelity / warm-start; **new Phase 3.6** for BMA combiner + DecayMonitor + first end-to-end CSI 300 mine |
| §11.1   | Grammar formally splits Feature into RawFeature ∪ DerivedFeature (M3); MCTS feature-sampling prior 70/30 → 50/50 |
| §11.3   | **§11.3.1 base MCTS, §11.3.2 multi-horizon (S1), §11.3.3 multi-fidelity (S2), §11.3.4 transfer-learning warm-start (S3), §11.3.5 budget summary** |
| §11.4   | **Fitness implementation rewritten** as 7 sub-modules covering robustness panel, three-class margin, pre-fitness gates, PR-AUC, CVaR, Bayesian shrinkage, and the orchestrator |
| §11.6   | **Full rewrite**: triple-barrier three-class labels with vol-scaled barriers (§11.6.1), magnitude buckets (§11.6.2), sample-uniqueness weights (§11.6.3), meta-labelling primary model (§11.6.4), label cache (§11.6.5), CPCV splits (§11.6.6), pipeline graph (§11.6.7), **§11.6.8 BMA combiner + DecayMonitor**, §11.6.9 v2 label-mode roster |
| §13     | Eight new risk-register rows: triple-barrier calibration, sample-uniqueness propagation, primary-model bias, multi-fidelity bias, negative transfer, shrinkage too aggressive, hierarchical-feature action-space inflation, BMA over-blend, DecayMonitor false-positives |
