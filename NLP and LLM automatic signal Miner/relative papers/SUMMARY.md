# Relative Papers — STL Mining, RL Signal Mining & LLM Alpha Mining

A curated set of 10 research papers most relevant to the project goal:

> Auto-mining a pool of CTA strategies whose **building blocks are Signal Temporal Logic (STL) formulas**, with admission gated by IC, IR, Sharpe, turnover, absolute return, and correlation with existing strategies. Adjacent inspiration drawn from **RL-based formulaic-alpha mining** and **LLM-driven alpha mining**.

The selection deliberately spans three families that combine well into a single mining pipeline:

| Family | Why it matters for your project | Papers |
|---|---|---|
| **Signal Temporal Logic learning** | The actual representation/grammar your strategies will be expressed in. Key trade-off: differentiable vs. combinatorial inference; interpretability vs. expressiveness. | #1–#4 |
| **RL-driven formulaic-alpha mining** | The closest published analog of your auto-mining loop. Operator pool + tree expression + RL search + pool-level fitness. Directly portable to STL trees. | #5–#7 |
| **LLM-driven alpha mining** | Adds prior + reasoning to break out of the search-space-explosion problem; promising for small-data regimes because the LLM brings external knowledge. | #8–#10 |

---

## Recommended reading order

If you only have time for three, read **#5 (AlphaGen)**, **#3 (TLINet)**, and **#10 (Alpha-Jungle MCTS)** — they jointly cover the search loop, the differentiable representation, and the LLM-guided exploration mechanism.

A pragmatic full sequence that builds intuition:

1. #5 AlphaGen — the canonical RL formulaic-alpha pipeline (your skeleton)
2. #7 QuantFactor REINFORCE — what to fix in AlphaGen for stability
3. #6 Alpha² — adds MCTS + structured logical operators (closest in spirit to STL)
4. #2 Neural-STL — basic differentiable STL classifier
5. #3 TLINet — modern differentiable STL with structure learning
6. #4 STL Decision Trees (MILP) — interpretable, exact, small-data friendly (matches your data constraint)
7. #1 RAMTL — retrieval-augmented STL mining (good prior for warm-starting)
8. #8 Alpha-GPT — LLM as a researcher-in-the-loop
9. #9 AlphaAgent — multi-agent LLM alpha mining with regularization for decay
10. #10 Alpha-Jungle (LLM + MCTS) — likely the cleanest blueprint for an LLM-guided STL miner

---

## Paper-by-paper summary

### 1. Retrieval-Augmented Mining of Temporal Logic Specifications from Data (RAMTL)
- **Authors / Venue:** Gaia Saveri, Luca Bortolussi (FM 2024 / arXiv 2024-05)
- **arXiv:** https://arxiv.org/abs/2405.14355
- **PDF:** https://arxiv.org/pdf/2405.14355
- **Code:** *(not released as of writing)*

**What it does.** Learns STL formulae directly from labeled trajectories without committing to a fixed template. The structure of the formula is found via **information retrieval** over a precomputed dense vector database of millions of formula embeddings (a learned semantic-preserving encoder), and the parameters are tuned via **Bayesian Optimization** over the robustness margin.

**Why it matters for your project.** Pure GP / RL search over STL is high-variance and slow with limited data. RAMTL gives you a way to **warm-start the search** by retrieving formula skeletons that already separate similar signals — directly relevant to a "small data" regime where you cannot afford to learn structure from scratch. The vector DB also gives you a natural similarity metric that you can later reuse as a *correlation-with-existing-strategies* gate.

**Adapt for CTA mining.** Replace their CPS signals with rolling windows of price/return/feature panels per asset; replace the binary anomaly label with sign(forward return) or quantile labels; reuse the retrieval index across assets in your pool.

---

### 2. Learning Signal Temporal Logic through Neural Network for Interpretable Classification (Neural-STL)
- **Authors / Venue:** Danyang Li, Mingyu Cai, Cristian-Ioan Vasile, Roberto Tron (CDC / arXiv 2210.01910, Oct 2022)
- **arXiv:** https://arxiv.org/abs/2210.01910
- **PDF:** https://arxiv.org/pdf/2210.01910
- **Code:** *(not released)*

**What it does.** Builds a differentiable surrogate for STL semantics so an STL classifier can be trained with gradient descent end-to-end. Each network module corresponds to an STL operator; weights become the formula parameters; sparsity / pruning recovers interpretable formulas.

**Why it matters.** Establishes the *neuro-symbolic* paradigm that lets you optimize STL formulas with PyTorch, which is much more sample-efficient than pure search. Foundational reference before reading TLINet.

**Caveat for finance.** Their toy domains are smooth/clean; you'll need to think about how robust signed-distance behaves on noisy returns (Winsorize before evaluation, or use a smoothed-min surrogate as in their soft-min layer).

---

### 3. TLINet: Differentiable Neural Network Temporal Logic Inference
- **Authors / Venue:** Danyang Li, Mingyu Cai, Cristian-Ioan Vasile, Roberto Tron (arXiv 2024-05)
- **arXiv:** https://arxiv.org/abs/2405.06670
- **PDF:** https://arxiv.org/pdf/2405.06670
- **Code:** *(no public release found at time of writing)*

**What it does.** A successor to Neural-STL that learns **both structure and parameters** of STL formulas jointly through differentiable approximations of the `max`/`min` operators. Key engineering trick: smooth-max/min approximations that are tight enough not to misclassify the STL satisfaction value.

**Why it matters.** This is the most usable differentiable STL framework today and is the closest fit to your "limited data" constraint. Because everything is gradient-based you can attach domain-specific regularizers (turnover penalty, correlation penalty against your existing pool) directly to the loss — exactly the gates you want to use for admission.

**Suggested adaptation.** Define the loss as `loss = -IC + λ_turn · turnover + λ_corr · max_corr_with_pool + λ_complex · |formula|`. The last three terms are differentiable approximations that match your admission criteria.

---

### 4. Learning Optimal Signal Temporal Logic Decision Trees for Classification: A Max-Flow MILP Formulation
- **Authors / Venue:** arXiv 2407.21090 (2024)
- **arXiv:** https://arxiv.org/abs/2407.21090
- **PDF:** https://arxiv.org/pdf/2407.21090
- **Code:** *(none located)*

**What it does.** Casts STL decision-tree inference as a **mixed-integer linear program** with a max-flow objective. Yields globally optimal, small, interpretable trees of STL primitives.

**Why it matters for small data.** When you have very limited training trajectories, exact combinatorial methods often outperform RL — there is simply not enough data to feed a high-variance policy gradient. The MILP gives you a baseline that your fancier methods must beat. It is also the easiest method to *prove* a complexity bound on, which matters if you later need governance/explainability.

**Limit.** Scales to small alphabets and short formulas. Use as the per-asset baseline; not as the main miner.

---

### 5. Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning (AlphaGen)
- **Authors / Venue:** Yu, Xu, Yang, et al., **KDD 2023** (arXiv 2306.12964)
- **arXiv:** https://arxiv.org/abs/2306.12964
- **PDF:** https://arxiv.org/pdf/2306.12964
- **Code:** https://github.com/RL-MLDM/alphagen — **cloned locally → `repos/05_AlphaGen/`**

**What it does.** Mines a *pool* (not a single formula) of formulaic alphas by representing each alpha in **Reverse Polish Notation** and training a PPO policy over the token sequence. The reward is the *incremental contribution* of the new alpha to a downstream linear combination model — i.e. it explicitly penalizes redundancy with the existing pool.

**Why it is the most directly applicable paper here.** This is essentially the framework you described in the prompt, just with formulaic alphas instead of STL formulas. Substitute the operator dictionary (`add`, `mul`, `ts_mean`, `ts_corr`, ...) with the STL primitives (`G[a,b]`, `F[a,b]`, `U`, atomic predicates with thresholds) and you get an STL miner. Their pool-level reward already implements the "low correlation with existing strategies" admission rule.

**What to copy.** Their `AlphaCalculator` interface, the IC / Rank-IC / pool-IC reward, the gplearn baseline harness, the rolling-train/evaluate loop, and `alphagen_llm/` (already integrates LLM-based generation as a baseline — useful crossover for #8–#10).

**What to change for STL.** The token vocabulary, the legal-action mask (STL temporal operators must take an interval), and the evaluator (replace formulaic-alpha evaluation with STL robustness-based signal generation, e.g. position = `tanh(robustness)` or position = `sign(robustness)`).

---

### 6. Alpha²: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning
- **Authors / Venue:** Xu, Yin, Zhang, Liu, Jiang, Zhang (arXiv 2406.16505, 2024)
- **arXiv:** https://arxiv.org/abs/2406.16505
- **PDF:** https://arxiv.org/pdf/2406.16505
- **Code (pseudocode/structure only):** https://github.com/x35f/alpha2 — **cloned locally → `repos/06_Alpha2/`**

**What it does.** Replaces AlphaGen's flat PPO over token sequences with **MCTS over an expression tree**, plus an explicit *legality mask* and a vocabulary that includes **logical operators** (`if-then-else`, comparators) so the alpha is evaluable as a reasoning tree rather than a pure formula. Inspired by AlphaDev / AlphaZero.

**Why it matters for STL.** STL is fundamentally a logical language with quantitative semantics — Alpha²'s logical operators map cleanly to STL atomic predicates and Boolean composition. Their tree search + legality mask is the right backbone for STL because most random STL token strings are syntactically invalid.

**Caveat.** The released repo is *pseudocode + structure*, not a runnable system. Use it as a design reference, not a drop-in.

---

### 7. QuantFactor REINFORCE: Mining Steady Formulaic Alpha Factors with Variance-Bounded REINFORCE
- **Authors / Venue:** Zhao, Zhang, Qin, Yang (arXiv 2409.05144, 2024)
- **arXiv:** https://arxiv.org/abs/2409.05144
- **PDF:** https://arxiv.org/pdf/2409.05144
- **Code:** *(no public release found)*

**What it does.** Drops the critic network from AlphaGen-style PPO, replaces it with a **theoretically grounded greedy baseline** for vanilla REINFORCE, and proves an upper bound on the gradient variance. Empirically more stable than AlphaGen on the same benchmarks.

**Why it matters for your project.** Stability is *the* practical issue when you train an RL miner on limited data — you'll regularly see high-variance updates collapse the policy. QFR's baseline is cheap to implement (no second network) and pairs naturally with a small operator vocabulary, which you'll have for STL. Apply it as your default upgrade over the AlphaGen reward path.

---

### 8. Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment
- **Authors / Venue:** Wang et al. (arXiv 2308.00016, 2023; demo at EMNLP 2025)
- **arXiv:** https://arxiv.org/abs/2308.00016
- **PDF:** https://arxiv.org/pdf/2308.00016
- **Code (third-party reimplementation):** https://github.com/parthmodi152/alpha-gpt

**What it does.** Wraps a researcher–LLM dialogue loop around an alpha search. The researcher describes an idea in natural language, a *Knowledge Compilation* module fetches similar precedents from external memory, the LLM emits valid alpha expressions, an *Alpha Search* module evaluates them, and a *Thoughts Decompiler* explains results back in natural language. The loop keeps cycling.

**Why it matters.** This is the paradigm you should adopt for the *seed-generation* phase of your pipeline. For STL, you'd ask the LLM in natural language ("a strategy that goes long when 5d momentum has been positive for at least 3 consecutive days and 20d realized vol is below its 60d mean") and have it emit a valid STL formula in your DSL. Then your RL/MCTS module takes over to refine.

**Limit.** The loop assumes a competent quant-in-the-loop. Your auto-miner will need to either replace the human with another agent (#9) or use an automatic critic (#10).

---

### 9. AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay
- **Authors / Venue:** *KDD 2025* (arXiv 2502.16789)
- **arXiv:** https://arxiv.org/abs/2502.16789
- **PDF:** https://arxiv.org/pdf/2502.16789
- **Code:** https://github.com/RndmVariableQ/AlphaAgent — **cloned locally → `repos/09_AlphaAgent/`**

**What it does.** Three cooperating LLM agents:
- **Idea Agent** — proposes hypotheses from financial theory or recent news/trends.
- **Factor Agent** — turns each hypothesis into a concrete factor expression and is regularized for **(a)** AST-similarity novelty against the existing pool, **(b)** semantic alignment with the original hypothesis, **(c)** complexity bounded by the AST depth/size.
- **Eval Agent** — runs backtest, computes IC/IR/Sharpe/etc., and feeds errors back into the next iteration.

**Why this is the closest published architecture to what you described.** Your admission gates (IC, IR, Sharpe, turnover, abs return, correlation with existing) map almost one-to-one onto AlphaAgent's three regularization mechanisms plus its Eval Agent's metrics. The AST-similarity novelty check is exactly the "low correlation with existing strategies" admission rule, computed on representation similarity rather than only return similarity (which is more robust).

**Reported result.** 11.0% annual excess return (IR=1.5) on CSI 500 and 8.74% (IR=1.05) on S&P 500, 2021-01 to 2024-12, **after** transaction cost.

**Adapt for STL.** Replace the Factor Agent's expression DSL with your STL DSL; keep the AST-similarity check verbatim (STL formulas are already ASTs). The Idea Agent is also where your RL miner from #5–#7 can plug in as an *alternative* idea source, giving you a hybrid LLM+RL miner.

---

### 10. Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining
- **Authors / Venue:** Yu Shi, Yitong Duan, Jian Li (arXiv 2505.11122, 2025)
- **arXiv:** https://arxiv.org/abs/2505.11122
- **PDF:** https://arxiv.org/pdf/2505.11122
- **Code:** *(not released as of writing)*

**What it does.** Uses an LLM as the *proposal/refinement engine* inside an **MCTS** loop. Each MCTS node is a partial formula; the LLM picks promising expansions; backtest IC drives the value estimate. Adds a **frequent-subtree avoidance** mechanism (penalize partial formulas that already appear too often in the visited tree) to fight homogenization — directly analogous to AlphaAgent's AST novelty regularizer but built into the search itself.

**Why this is probably the cleanest blueprint to follow.** It combines all three threads into one system:
- LLM (#8, #9) for prior + interpretation
- MCTS (#6) for structured search
- explicit novelty mechanism (your "low correlation" admission rule)
- IC-driven backtest reward (your other admission rules can plug into the same scalar)

**Recommended action.** Treat this paper as the architecture target and build progressively from AlphaGen (#5) → swap PPO for MCTS (#6) → swap critic for QFR baseline (#7) → swap proposal distribution for an LLM (#10) → wrap with multi-agent governance (#9).

---

## What is in this folder

```
relative papers/
├── SUMMARY.md                       <- this file
├── download_pdfs.sh                 <- shell script to fetch the 10 PDFs from arXiv
├── download_pdfs.py                 <- Python equivalent (no curl/wget needed)
└── repos/
    ├── 05_AlphaGen/                 <- full repo for paper #5 (AlphaGen, KDD'23)
    ├── 06_Alpha2/                   <- pseudocode/structure for paper #6 (Alpha²)
    └── 09_AlphaAgent/               <- full repo for paper #9 (AlphaAgent, KDD'25)
```

The PDFs themselves are **not** included because the sandbox's network policy does not allow direct fetches from `arxiv.org`. Run `download_pdfs.sh` (or `python download_pdfs.py`) on your host machine to populate the folder — both scripts target this same directory and will write the 10 PDFs alongside this `SUMMARY.md`.

## Where to start coding

A minimal first iteration that uses what's already here:

1. Open `repos/05_AlphaGen/` and run their pipeline once on your CSI 500 / your-asset-pool data so you have a working RL alpha-mining loop.
2. Replace the token vocabulary in `alphagen/data/expression.py` with STL primitives (`G[a,b]`, `F[a,b]`, `not`, atomic predicates with learnable thresholds).
3. Replace `AlphaCalculator` with an STL-robustness-based scorer that returns `IC` and `Rank IC` of `tanh(robustness_t)` against forward returns.
4. Add three admission filters in `train_maskable_ppo.py` *after* a candidate is proposed but *before* it joins the pool: turnover threshold, max-correlation with pool, and absolute-return floor on the candidate's stand-alone backtest. This mirrors AlphaAgent's regularization stack.
5. Once stable, fold in the Idea Agent from `repos/09_AlphaAgent/` as an alternative proposal distribution alongside the RL policy.
