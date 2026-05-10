# Claude Code build prompt — `AutoLLM/` v1

This file is the orchestration document. Open Claude Code at `/Users/zhangziyao/Desktop/AutoLLM/`, paste the **Prompt to Claude Code** block below into the first message, and run as a single conversation. The harness rules are deliberate — every clause earned its place by failing without it.

---

## How this prompt works (orientation for the human, not Claude Code)

The prompt below has three structural pieces:

1. **Bootstrap** — Claude Code reads the proposal + paper summaries, then writes a `CLAUDE.md` that becomes the durable shared context for the rest of the session.
2. **Phases** — each phase is independently demo-able and ends with a *gate* (tests pass + human-readable artifact). Phases run sequentially in the main thread, but **per-phase research and per-phase implementation may be split across subagents** to keep the main context lean.
3. **Harness rules** — invariants that hold across all phases (Plan mode before non-trivial work, TodoWrite for state, subagent dispatch policy, when to stop and ask).

The phases as written cover **Phase 0 through Phase 3** of the proposal — i.e. up through "STL pipelines baseline working end-to-end on a tiny universe". Phases 4+ (signal_miner shared services, Pipelines B/C/D) get a separate prompt later.

---

## Prompt to Claude Code

> You are about to build the v1 of a quantitative-research factory in `/Users/zhangziyao/Desktop/AutoLLM/`. Your spec is `proposal_version1.md`. Your reading list is `NLP and LLM automatic signal Miner/relative papers/`. Your output is three working modules — `data_fetcher/`, `alpha158/`, and `STL pipelines/` — plus a `CLAUDE.md` and updated `proposal_version1.md` decision log.
>
> Read these harness rules carefully **before** doing anything. They are not optional. If you violate one, stop and ask.
>
> ---
>
> ### Harness rules (apply to every phase)
>
> **R1 — Plan mode before any non-trivial change.** Before you start coding any of Phase 1 / 2 / 3, drop into Plan mode and propose: file layout, public API shape, test strategy. Show me the plan. Wait for approval. The exception is Phase 0 (orientation) which is just reading + writing one markdown file.
>
> **R2 — Use TodoWrite as the state machine.** Maintain a single live todo list throughout the conversation. The first thing you do (before reading any file) is create a todo list with **the four phases below as top-level items**, then add subtasks as you go. Mark tasks `in_progress` when starting, `completed` when done, never both at once. Don't expand subtasks for a phase you haven't started yet — keep the future fuzzy until you get there.
>
> **R3 — Subagent dispatch policy.** Use the `Task` tool with a fresh subagent any time:
>  - You need to **read a paper** (PDF in `relative papers/`). Subagent reads + extracts the algorithmic detail you need + returns ≤300 words. Never load a full PDF into your main context.
>  - You need to **read a large existing module** for inspiration (e.g. Qlib's `qlib/contrib/data/handler.py` for the Alpha158 formulas; AlphaGen's training loop). Subagent reads, summarises the relevant interface or algorithm, and returns code skeletons. Never read >500 lines into your main context yourself.
>  - You need to **run a long verification** (e.g. cross-checking 158 Alpha158 columns against Qlib's output, or a slow integration test against AKshare). Subagent runs it, reports pass/fail with the failing-row hash if any.
>  - **Never** use a subagent for the actual implementation of the modules. The implementation is the durable artifact and belongs in the main thread where I can see every edit.
>
> **R4 — Test-first within each phase.** For each module:
>  1. Write the public-API test file first (the failing tests).
>  2. Implement.
>  3. Run the tests.
>  4. **If a test fails twice in a row after edits, stop and report.** Don't loop "fix then rerun" more than twice without a human in the loop — that's the failure mode where you start hallucinating the bug.
>
> **R5 — Conventions live in `CLAUDE.md`, not in your head.** First task of Phase 0 is to write `AutoLLM/CLAUDE.md`. Anything that you'd otherwise carry as session-state (Python version, formatter choice, DSL syntax, file naming, where venvs live, where data goes on disk) goes into `CLAUDE.md` instead. If a future phase needs a convention you haven't recorded, *add it* to `CLAUDE.md` before you use it.
>
> **R6 — Don't add stuff outside `AutoLLM/`.** Everything you write goes into `AutoLLM/`. Never modify the cloned `qlib/` folder. Never modify files inside `NLP and LLM automatic signal Miner/relative papers/`. If you need a venv, put it at `~/venvs/<purpose>` (paths under the home directory are fine, just not inside the project tree).
>
> **R7 — Asking is free; guessing is expensive.** If a spec is ambiguous, write the ambiguity into your todo list as a question, finish what you can without it, and ask before starting anything that depends on the answer. Do not invent thresholds, file paths, or operator semantics that aren't in the proposal.
>
> **R8 — Network discipline.** Network calls (AKshare, package installs) only happen in:
>  - Phase 0 setup: package installs.
>  - Phase 1 integration tests: AKshare smoke pulls, gated behind a `--network` pytest marker.
>  - Phase 2 cross-validation against Qlib: a *one-time* small AKshare pull then Qlib local computation in a throwaway venv.
>
>  All other tests run on fixture data baked into the repo. `data_fetcher/tests/fixtures/` should hold a 5-symbol × 100-day parquet sample after Phase 1 so Phases 2–3 are network-free.
>
> **R9 — Stop conditions.** Stop and report to me at:
>  - End of Phase 0 (after `CLAUDE.md` is written, before Phase 1 plan).
>  - After each phase's plan (R1) before you implement.
>  - After each phase's tests pass (or after R4's twice-failed condition).
>  - Any time R7 fires.
>
> ---
>
> ### Phase 0 — Orient (no code yet)
>
> Goal: build the durable shared context for the rest of the session.
>
> 1. Read `AutoLLM/proposal_version1.md` end-to-end.
> 2. Read `AutoLLM/NLP and LLM automatic signal Miner/relative papers/SUMMARY.md` end-to-end. (Do **not** open the individual paper PDFs yet — defer that to per-phase subagent dispatch when a specific algorithm is needed.)
> 3. Read `AutoLLM/data_fetcher/README.md` and `AutoLLM/STL pipelines/README.md`.
> 4. Skim `AutoLLM/qlib/qlib/contrib/data/handler.py` (this is where Qlib's Alpha158 lives). Don't load the file — dispatch a subagent: "summarise the Alpha158 feature definitions in `qlib/contrib/data/handler.py`. Return: the list of feature names, the categories they fall into (rolling stats / cross-asset / etc.), and a representative example formula in Python pseudocode for each category. ≤400 words."
> 5. Write `AutoLLM/CLAUDE.md`. It must include, at minimum:
>    - One paragraph: what is this project.
>    - The folder map (copy from proposal §6.4 and keep updated).
>    - Conventions: Python version (3.11+), venv location (`~/venvs/autollm311` for shared), formatter (`ruff format`), test runner (`pytest`), parquet writer (`polars` for new writes, pandas only when interfacing with libraries that demand it).
>    - DSL syntax (carry over from proposal §9.1 plus whatever Alpha158 needs).
>    - Where data lives on disk (`~/AutoLLM_data/` — keep it outside the repo).
>    - One section: "open questions" — anything you flagged via R7.
> 6. **Stop and report** with the contents of CLAUDE.md and your phase-0 todo list.
>
> ---
>
> ### Phase 1 — `data_fetcher/` (~3 days of work)
>
> Spec: `AutoLLM/proposal_version1.md` §7.1, plus `AutoLLM/data_fetcher/README.md`.
>
> Plan-mode design must include:
> - The directory structure under `data_fetcher/`.
> - Public API: typed signatures for `Fetcher.pull`, `Fetcher.panel`, `Fetcher.universe`, `AKshareParquetAdapter`.
> - Storage layout on disk: exact path patterns (`~/AutoLLM_data/raw/year=YYYY/symbol=SHXXXXXX.parquet`).
> - Test strategy: what runs offline (everything except the AKshare smoke), what runs with `--network`.
> - Symbol-normalisation rules (`600519` → `SH600519`, `000001` → `SZ000001`, `688981` → `SH688981`, etc.) with a table of edge cases.
> - PIT universe approach: where do we get historical CSI 300 membership from AKshare, how do we snapshot, what API does the consumer call (`Fetcher.universe("csi300", on_date)`).
>
> After approval, implement test-first per R4. Suggested ordering:
> 1. `data_fetcher/symbols.py` + tests for normalisation.
> 2. `data_fetcher/storage.py` + tests for parquet round-trip and idempotent writes.
> 3. `data_fetcher/akshare_client.py` + offline tests with mocked HTTP, **plus** a `--network` integration test for one symbol.
> 4. `data_fetcher/universe.py` + tests with a baked-in PIT membership fixture.
> 5. `data_fetcher/adapter.py` — `AKshareParquetAdapter` implementing the `DataAdapter` protocol.
> 6. `data_fetcher/cli.py` — `python -m data_fetcher pull/panel/universe`.
> 7. Tests pass → run a 5-symbol smoke pull, save the resulting parquet into `data_fetcher/tests/fixtures/` for Phases 2–3 to consume offline.
>
> **Phase-1 demo:** a one-liner in a notebook: `Fetcher().panel(["SH600519","SZ000001"], ["$open","$close"], "2024-01-01","2024-03-31").collect()` returns a Polars DataFrame with the expected dates and prices.
>
> **Stop and report** with the test summary and the demo output.
>
> ---
>
> ### Phase 2 — `alpha158/` (~half a day of work)
>
> Spec: `AutoLLM/proposal_version1.md` §6.2 and §7.2.
>
> Plan-mode design must include:
> - The DSL operator vocabulary (start from a list of {Std, Mean, Max, Min, IdxMax, IdxMin, Slope, Resi, Quantile, Rank, Corr, Log, Greater, Less, If, Add, Sub, Mul, Div, Ref, Delta} — confirm with the subagent's Alpha158 summary).
> - The `compute(adapter, universe, start, end) -> pl.LazyFrame` driver shape.
> - Cross-validation strategy: how to compare our 158 columns against Qlib's reference output without permanently coupling to Qlib.
>
> Implementation:
> 1. `alpha158/dsl.py` — the operator implementations as Polars expressions.
> 2. `alpha158/alpha158.py` — the 158 formulas as a Python list of `(name, dsl_expression_string)`.
> 3. `alpha158/compute.py` — the driver.
> 4. **Cross-validation step** (subagent for the heavy lifting):
>    - Subagent: "Set up a throwaway venv at `~/venvs/qlib_xref311` with Qlib + dependencies (Python 3.11). Run Alpha158 on the 5-symbol fixture from Phase 1 over a 1-year window. Save the output parquet to `~/AutoLLM_data/_qlib_xref/alpha158_5sym_1y.parquet`. Then **delete the venv**. Return: the path to the saved parquet, and a 1-line confirmation that the venv is deleted."
>    - Main thread: load both that parquet and our own output, assert column-by-column max-abs-diff ≤ 1e-6, fail loudly on any column where it isn't.
> 5. Any column that doesn't match: open an issue in `CLAUDE.md`'s "open questions" section, don't silently fix.
>
> **Phase-2 demo:** the cross-validation script prints "158/158 columns match within 1e-6 on 5 symbols × 252 days".
>
> **Stop and report.**
>
> ---
>
> ### Phase 3 — `STL pipelines/` MCTS baseline (~2 weeks of work)
>
> Spec: `AutoLLM/proposal_version1.md` §11 (full algorithm), §11.6 (data processing — triple-barrier three-class meta-labelling), §5.2 (F_A fitness), §5.6 (pre-fitness gates and post-fitness admission gates), §15 (compute environment), and `AutoLLM/STL pipelines/README.md`.
>
> **Important constraint.** v1 is **template-free Monte Carlo Tree Search, no deep learning**. Pure tree search + Polars rolling reductions on M1 16 GB. Do not introduce PyTorch, do not introduce a neural-policy prior. GP, TLINet, RAMTL are all v2 — explicitly out of scope.
>
> **Phase 3 covers only the base MCTS + base data pipeline.** Multi-horizon (S1), multi-fidelity (S2), and transfer-learning warm-start (S3) are Phase 3.5; BMA combiner (M2) and DecayMonitor (E4) are Phase 3.6. Get the base working first.
>
> Plan-mode design must include:
> - The AST node types and their serialisation format (text DSL), with the **hierarchical-feature** distinction (RawFeature vs DerivedFeature) per proposal §11.1.
> - The Polars-vectorised robustness semantics.
> - The MCTS engine per §11.3.1: typed-action expansion, UCB1, random rollout, backup, frequent-subtree avoidance.
> - The triple-barrier `LabelComputer` per §11.6.1 (vol-scaled barriers, three-class output {long, flat, short}, magnitude buckets per §11.6.2).
> - The `SampleUniquenessWeighter` per §11.6.3.
> - The deterministic `PrimaryModel` (TSMOM-with-vol-filter) per §11.6.4.
> - The F_A fitness shape per §5.2 + §11.4.
> - The pre-fitness gates per §5.6 (F2 hit-count, E1 leakage assertion, complexity, trivial-class collapse).
> - The post-fitness admission gates per §5.6 (F4 PR-AUC, F5 CVaR-5%, F1 shrunk Sharpe).
> - Test strategy: unit tests for **each** semantic rule against a hand-computed example; property-based test for MCTS × seed determinism; weighted-stats correctness test on synthetic data with known overlap; class-balance test on synthetic data where the right formula is known.
>
> Subagent dispatch (R3) for paper details — do this once at the start of Phase 3, save the summaries into `AutoLLM/CLAUDE.md`'s "research notes" section, never re-read these papers in main context:
> - Subagent: "Read `relative papers/02_NeuroSTL_2022_NN_STL_Interpretable_Classification.pdf`. Extract: the exact STL semantics, robustness-margin computation. ≤300 words."
> - Subagent: "Read `relative papers/04_STLDT_2024_Optimal_STL_Decision_Trees_MILP.pdf`. Extract: their MILP formulation, the size of universe / sequence length they handle. We need to call the MILP for ≤5 assets × 1 year as a sanity-check baseline. ≤300 words."
> - Subagent: "Search the web for **Mohammadinejad et al., 'Temporal logic explanations for dynamic decision systems using anchors and Monte Carlo Tree Search', AIJ 2023 (DOI 10.1016/j.artint.2023.103905), AAAI 2024 reprint**. Extract their MCTS-over-STL-AST machinery: state, action vocabulary, UCB constant, rollout policy. ≤500 words."
> - Subagent: "Read López de Prado *Advances in Financial Machine Learning* chapters 3 and 4 if accessible from the web (e.g. via the published Wiley page or a personal-site PDF if one is freely available). Extract the precise definitions of (a) triple-barrier method, (b) sample-uniqueness weights, (c) meta-labelling. We need formulas, not narrative. ≤500 words. If not findable, return 'not found' and the implementation will fall back to the formulas in proposal §11.6.1, §11.6.3, §11.6.4."
>
> Implementation order:
> 1. `STL pipelines/stl/grammar.py` — AST + DSL parser/printer + tests; RawFeature vs DerivedFeature split per §11.1.
> 2. `STL pipelines/stl/semantics.py` — vectorised robustness with Winsorize-1/99.
>    - Hand-test: 10-day price series; manually compute robustness for `G[0,2] ($close > 100)`; assert match.
> 3. `STL pipelines/stl/labels.py` — `TripleBarrierLabelComputer` per §11.6.1, three-class output {long, flat, short}, magnitude buckets per §11.6.2, rule-hash-keyed parquet cache per §11.6.5.
>    - Sanity check on the 5-symbol Phase-1 fixture: log empirical class distribution; warn if long < 15 % or > 40 %.
> 4. `STL pipelines/stl/sample_weights.py` — sample-uniqueness weighter per §11.6.3, parquet cache.
>    - Property-based test: synthetic data with known overlap pattern → `weight × N` matches the analytical effective-N formula.
> 5. `STL pipelines/stl/primary.py` — deterministic TSMOM-with-vol-filter primary model per §11.6.4.
>    - Sanity check: emits "long" on ~10–25 % of CSI 300 days.
> 6. `signal_miner/eval/weighted_stats.py` (shared, used by all four fitness components) — single source of truth for `weighted_mean`, `weighted_std`, `weighted_corr`, `weighted_sharpe`. Every metric in the system flows through here.
> 7. `STL pipelines/stl/fitness/` — split into one file per component per §11.4:
>    - `robustness_panel.py`, `class_margin.py`, `pre_fitness_gates.py`, `pr_auc.py`, `cvar.py`, `bayesian_shrinkage.py`, `orchestrator.py`. Each a pure function with at least one hand-computed unit test.
> 8. `STL pipelines/stl/search/random.py` — random-policy STL formula sampler (the floor MCTS must beat).
> 9. `STL pipelines/stl/search/mcts.py` — base MCTS engine per §11.3.1 (typed action mask, UCB1, random rollout, backup, frequent-subtree avoidance, per-asset parallelism via `multiprocessing.Pool` of size ≤ 8). **No multi-fidelity, no warm-start in Phase 3** — those are Phase 3.5.
> 10. `STL pipelines/cli.py` — `python -m stl mine --universe csi300 --asset SH600519 --horizon 5 --iterations 50000`.
> 11. Run on the 5-symbol Phase-1 fixture at horizon=5: confirm MCTS top-1 beats random-policy top-1 on at least 3 of 5 assets; confirm empirical long-class rate is 15–40 %.
> 12. **Required:** MILP sanity-check from §11.5 — confirm MCTS top-1 reaches ≥ 80 % of MILP optimum's separation margin on 5 assets × 1 year.
>
> **Phase-3 demo:** a notebook (`STL pipelines/notebooks/demo_phase3.ipynb`) that runs the MCTS miner on one asset at horizon=5 with ~20k iterations, prints the top-5 formulas in plain English, shows the three-class label panel, the robustness panel, and the realised long-only equity curve from `position_t = (ρ > 0 AND primary_t = long)` overlaid on buy-and-hold.
>
> **Stop and report.** Include in the report:
> - Empirical class distribution {long, flat, short} per asset.
> - Per-asset best F_A vs. random-policy baseline.
> - For each top-1 formula: three-class margin, FPR on flat+short, hit count, PR-AUC long, CVaR-5%, Sharpe (raw), Sharpe (Bayesian-shrunk), Sortino, MaxDD, CAGR, turnover_annual, time_in_market, IC_TS, conditional_mean_return_per_trade.
> - Top-3 formulas overall, DSL + plain-English + realised equity curves.
> - Wall-clock per asset, peak memory.
> - MILP sanity-check ratio.
>
> ---
>
> ### Phase 3.5 — Multi-horizon, multi-fidelity, transfer-learning warm-start (~1 week)
>
> Spec: proposal §11.3.2–§11.3.4. Build on the Phase-3 base.
>
> 1. `STL pipelines/stl/search/multi_horizon.py` — orchestrator that spawns 4 parallel MCTS runs at H ∈ {3, 5, 10, 20} per asset, each tagged with its horizon.
> 2. `STL pipelines/stl/search/multi_fidelity.py` — split rollouts into cheap-tier (30 % stratified subsample, first 70 % of iterations) and full-tier (last 30 %), with re-evaluation transition.
> 3. `STL pipelines/stl/search/warm_start.py` — sector + vol-regime nearest-neighbour identification, seed-tree initialiser per §11.3.4. Cold-start the first 30 % of universe vanilla.
> 4. Re-run on the 5-symbol fixture across all 4 horizons. Confirm: (a) wall-clock per asset doesn't blow up, (b) different horizons admit different formulas, (c) S3 reduces effective MCTS iterations needed for late assets by ≥ 30 %.
>
> **Stop and report.** Per-asset + per-horizon class distribution; horizon-by-horizon top-3 formulas; effective iteration savings from S3; multi-fidelity bias check (Spearman of cheap-tier ranks vs full-tier ranks ≥ 0.6).
>
> ---
>
> ### Phase 3.6 — BMA combiner + DecayMonitor + first end-to-end CSI 300 mine (~3 days)
>
> Spec: proposal §11.6.8.
>
> 1. `signal_miner/pool/bma_combiner.py` — per-asset top-K (default K=5) BMA-weighted blending, BIC weights, position rule per §11.6.8.
> 2. `signal_miner/pool/decay_monitor.py` — nightly job that re-evaluates every admitted formula's rolling 60d sample-weighted Sharpe, applies the eviction triggers per §11.6.8, writes to `decay_history`.
> 3. End-to-end run on full CSI 300, all 4 horizons, full v1 admission stack. Mine + gate + admit + BMA-blend.
>
> **Phase-3.6 demo:** a notebook (`STL pipelines/notebooks/demo_phase3_6.ipynb`) showing pool dashboard tagged by (asset, horizon), per-asset BMA-blended equity curves, DecayMonitor's first weekly report.
>
> **Stop and report.** Pool size, per-(asset, horizon) admission rates, BMA-blend Sharpe vs top-1 Sharpe per asset (sanity: blend should be within 10 % of top-1, ideally with lower vol), DecayMonitor's eviction count over a 1-year synthetic decay test.
>
> ---
>
> ### After Phase 3.6
>
> Don't start Phase 4 (`signal_miner/` shared services with the alpha-family gate and MTC stack) without an explicit go-ahead from me.
>
> Update `proposal_version1.md`'s decision log (§0) with one row per implementation choice you made that wasn't fully spec'd in the proposal. Be brief; one line per row.

---

## After Claude Code returns

Two follow-up actions for me (Owen):

1. **Spot-check rather than read every diff.** Most of the value of test-first + R4 (twice-failed → stop) is that you can trust passing tests. Skim the test files, run them yourself, and only read the implementation for the parts that surprise you.
2. **The Phase 2 cross-validation parquet is the single most important artifact for the next 3 months.** Hold on to it. Every time the DSL changes, re-run that diff. The day it stops matching is the day a silent bug shipped.

If Phase 1 fails because AKshare network egress is unreliable from the Mac, dump the failure log into a fresh chat with me here in Cowork — there are several mitigations (different endpoints, mirrors, tushare fallback) that are cheaper to debug interactively than to specify upfront.
