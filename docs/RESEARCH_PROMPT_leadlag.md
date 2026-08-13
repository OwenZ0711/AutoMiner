# Research prompt — lead-lag effects in Chinese futures (copy-paste into Claude)

Copy everything below the line into a Claude research session (claude.ai with
web search / deep research works best).

---

I am building a lead-lag factor-mining system for Chinese futures markets and
need a rigorous research report. My setup: 1-minute OHLCV + open-interest bars
for ~95 futures products across all 6 Chinese exchanges (SHFE, DCE, CZCE, GFEX,
INE, CFFEX), 2005–2026. I will detect pairs where product i's price moves lead
product j's by some lag (minutes to days), validate them statistically, and
feed validated pairs into a genetic-programming miner that evolves trading
signals for the follower. Compute is a MacBook M1 (8 cores, 16 GB RAM), Python
with NumPy/Polars/SciPy/numba — no deep learning. Please research and write a
structured report on the following, with references (papers with arXiv/DOI
links) and concrete recommendations:

1. **Lead-lag estimators for asynchronous high-frequency data.** Compare:
   lagged cross-correlation functions; the Hayashi–Yoshida estimator and the
   Hoffmann–Rosenbaum–Yoshida lead-lag estimator; Lévy-area / path-signature
   based lead-lag detection (Gyurkó–Lyons line of work); frequency-domain
   methods (phase slope index, imaginary coherency); thermal optimal path;
   transfer entropy. For each: assumptions, what breaks at 1-minute bars,
   computational cost, and whether it handles two assets trading in different
   sessions (one day-only, one with a night session).

2. **The Epps effect and stale-price bias** at minute frequency: how do they
   fake or hide lead-lag between a liquid and an illiquid asset, and what are
   the accepted corrections?

3. **Valid inference for lead-lag.** The null hypothesis is "no lead-lag GIVEN
   strong contemporaneous correlation" — not independence. What test statistics
   are properly centered under that null (asymmetry statistics like
   ρ(ℓ)−ρ(−ℓ), the Huth–Abergel lead-lag ratio, Granger/ADL Wald tests with
   controls)? What permutation/bootstrap schemes preserve contemporaneous
   dependence and autocorrelation while destroying lag structure (block
   bootstrap variants, time-reversal tests)? How should Bartlett/HAC variance
   be computed for cross-correlations of heteroskedastic, autocorrelated
   returns?

4. **Multiple testing** when scanning ~4,500 pairs × ~60 lags × many rolling
   windows: best practice from the factor-zoo literature (Harvey–Liu–Zhu,
   Benjamini–Hochberg vs Benjamini–Yekutieli, deflated Sharpe ratio, CPCV /
   PBO). How do practitioners combine sup-over-lag statistics with FDR across
   pairs?

5. **Lead-lag networks.** Methods that turn pairwise lead-lag scores into a
   directed network and extract structure: skew-symmetric matrix
   decompositions, HodgeRank / least-squares ranking on graphs, SerialRank,
   Hermitian-matrix spectral clustering for directed graphs (Cucuringu et al.),
   and their applications to equity/futures lead-lag (e.g. Bennett, Cucuringu,
   Reinert). What is known about lead-lag cluster structure in commodity
   markets?

6. **Empirical evidence for lead-lag in Chinese commodity futures
   specifically.** Supply-chain pairs (iron ore → rebar → hot-rolled coil,
   coking coal → coke, oil → PX → PTA, soybean → soybean meal/oil), the
   equity-index ↔ commodity link, SHFE copper ↔ LME/INE copper, day vs night
   session information flow, and anything on lead-lag between the same
   commodity's futures on different Chinese exchanges. Chinese-language
   academic and broker research is in scope if findable.

7. **Chinese futures market microstructure facts that affect estimation**:
   staggered introduction of night sessions since 2013 and later changes /
   COVID suspension; daily price limits and their cascade behavior; the
   2020 switch from double-sided to single-sided volume/open-interest counting;
   the 2015 CFFEX stock-index-futures restrictions; dominant-contract roll
   conventions (volume vs open-interest rules, non-adjacent month jumps like
   01→05→09). Cite primary sources where possible.

8. **From detected pair to tradeable signal.** Literature on converting
   lead-lag relationships into trading strategies: expected decay/half-life of
   lead-lag alpha, how fast such effects have historically been arbitraged
   away, realistic transaction costs and slippage for Chinese futures
   (per-exchange fee structures, tick sizes), and execution at minute
   frequency (next-bar-open fills, limit-lock untradeability).

9. **Warm-starting genetic programming with known relationships.** Any work on
   seeding GP / symbolic regression with validated economic structures
   (beyond Ren–Qin–Li arXiv:2412.00896), grammar-constrained GP for trading
   factors, and dimension/type-aware operator constraints.

10. **A recommended v1 stack.** Given everything above, propose: which
    estimator to use as the cheap all-pairs screen, which for confirmation,
    which for tradeability testing; which null/permutation scheme; which FDR
    procedure; and a validation protocol (rolling windows, embargo,
    out-of-sample confirmation) — all feasible on the stated hardware.

Format: a structured report with a comparison table per section, a final
"recommended stack" section, and a full reference list. Flag anything where
the literature is thin or contradictory. Prefer primary sources over blog
posts; include key Chinese-market institutional details with dates.
