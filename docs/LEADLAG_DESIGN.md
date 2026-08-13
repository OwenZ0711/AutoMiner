# Lead-Lag CTA Factor Mining — Infrastructure Design

> Status: design approved-for-build pending user review. Produced 2026-05 from a
> three-perspective design panel (econometric / linear-algebra / computational),
> each adversarially critiqued (all verdicts: *sound-with-fixes*; fixes folded in),
> plus a full profile of `Future_minute_data/` and a close read of the GP warm-start
> paper (arXiv:2412.00896) and the AlphaGen reference implementation.

---

## 0. Hypothesis and goal

Movements of futures product *i* systematically precede movements of product *j*
by some lag ℓ (minutes to days). Economic priors make this plausible in Chinese
futures: supply chains (iron ore → rebar → hot coil; coking coal → coke; oil →
PX → PTA → short fiber / bottle chips; soybean → meal / oil), substitution
(L/PP/V plastics, RU/NR/BR rubbers), cross-market twins (SHFE CU ↔ INE BC),
macro blocks (index futures vs industrial metals; AU ↔ AG).

Goal: a pipeline that (1) **detects and statistically validates** lead-lag pairs
from minute bars, and (2) **feeds validated pairs into the GP miner** as
cross-asset terminals and warm-start structures, so GP evolves tradeable
follower signals. Output holding horizon: mid-to-high frequency (minutes → days).

---

## 1. Data reality (from profiling `Future_minute_data/`)

Facts every layer must respect:

| Fact | Consequence |
|---|---|
| Historical tree (2005–2025-07-04) has **only real YYMM contract files**; no continuous series | We build our own dominant-contract calendar + back-adjusted continuous series |
| `8888` (2026 tree only) = **OI-weighted index across all months** — fractional prices, not tradeable | Usable as a *feature* (term-structure blend), never as price backbone |
| `9999` = raw-stitched dominant with **unadjusted roll gaps** (eggs: +9.5% fake jump at a roll) | Never compute returns across a 9999 splice |
| **Hard data hole 2025-07-05 → 2026-01-04** | Mask lookbacks crossing it; metadata flag |
| Sessions: day 09:00–10:15 / 10:30–11:30 / 13:30–15:00; night none / →23:00 / →01:00 / →02:30 by product group; CFFEX 09:30–15:00 (index) / 15:15 (bonds), no night | Union-clock + per-product masks; longest contiguous segment 90 min (day) / 330 min (night) |
| Session calendars are **time-varying**: night sessions launched product-by-product from 2013-07, COVID suspension 2020-02→05, end-time changes (RB night 01:00→23:00 in 2016-05), CFFEX hours changed 2016 | A per-product per-date session table, **inferred from the data itself**, is a first-class artifact (single largest engineering item) |
| Volume/OI counting: SHFE/DCE/CZCE switched double→single-sided ~2020-01; CFFEX always single | Volume/OI features need a break adjustment or era flag |
| Zero-volume padded bars in historical tree (far months up to 84%); 2026 tree writes only traded minutes | Mask = `volume > 0`, not "row exists"; normalize the two trees to one convention |
| Sporadic all-NaN rows (SC2009); mixed-case filenames in 2026; Saturday folders hold post-midnight Friday tails | Ingest filters; case-normalize symbols; map bars to **exchange trading day** (bob ≥ 21:00 → next trading day) |
| `type` column constant (=14); `sequence` = candle color | Drop both on ingest |
| Daily price limits (涨跌停) lock products for hours; unlock produces synchronized jumps | Limit-locked bars get their own mask channel; tradability veto in GP fitness |
| CFFEX 2015-09 → ~2017 restrictions collapsed index-futures volume ~99% | Tradeability-by-era flags on CFFEX legs |
| Listings: INE SC 2018, BC 2020, GFEX SI 2022 / LC 2023 / PS 2024+, LH 2021 | Min-history policy; ~40–60 products usable at 1m, rest at 15m+ |

---

## 2. Mathematical formulation

### 2.1 Clocks and masks

Master clock = union of all exchange trading minutes, broken into **contiguous
session segments** at every break point (10:15, 11:30, 15:00, 23:00, 01:00, …).
For product *i*: validity mask `M[t,i] = 1` iff minute *t* is in *i*'s session
per the **time-varying calendar**, the dominant contract printed a real bar
(`volume > 0`), the bar is not roll-masked, and not limit-locked (limit-lock is
a separate channel `LIM[t,i]`).

Three clock views: **union grid** (primary, with masks), **intersection grid**
(09:30–10:15 ∪ 10:30–11:30 ∪ 13:30–15:00 ≈ 195 min — the true common window),
and a **resample pyramid** {1m, 5m, 15m, 60m, 1d} for multi-scale scans.

### 2.2 Returns

Per product: our own dominant-contract series (argmax daily volume with
no-lookback switch rule + minimum-persistence to avoid whipsaw), back-adjusted
multiplicatively at our own roll dates; log returns strictly intra-contract;
roll-crossing returns nulled. Raw dominant kept alongside for tradability checks.

### 2.3 Prewhitening

`x[t,i] = r[t,i] / σ̂[t,i]`, σ̂ = strictly causal EWMA vol (span ≈ 1 trading day)
× trailing intraday-seasonality profile (per session-template stratum).
Winsorize at 0.1/99.9% on trailing windows. Optionally project out the top 1–3
eigenmodes of the trailing contemporaneous correlation matrix (market/sector
factor) — run raw and factor-partialed in parallel and flag disagreements,
because over-partialing deletes genuine signal when the leader *is* the factor
proxy (copper for metals).

### 2.4 The lagged cross-correlation tensor (the workhorse)

For lag ℓ ∈ [1, L], L ≈ 60 bars (**not** 240 — longest contiguous segment is
90 min day / 330 min night; longer wall-clock lags are handled on the 15m/60m/1d
pyramid levels instead):

```
C[i,j,ℓ] = Σ_t  x̃[t,i]·x̃[t+ℓ,j]·M[t,i]·M[t+ℓ,j]   /   n[i,j,ℓ]
```

computed as **segment-blocked GEMMs**: break the panel at session boundaries,
compute `Z[:-ℓ]ᵀ Z[ℓ:]` within each segment, sum across segments. Per-element
masks cannot express "the pair (t, t+ℓ) must not straddle a break" — blocking
does. Normalization uses **three Gram products per lag** (`x·x`, `x²·mask`,
`mask·x²`) so each pair gets its *joint-valid* variance — global column
standardization alone mis-scales pairs whose overlap is day-only vs day+night.

Negative lags come free: `C[i,j,−ℓ] = C[j,i,ℓ]`. Tensor is 95×95×61 ≈ 2 MB;
a full pass is ~10 TFLOP ≈ **1–2 minutes** on the M1 via Accelerate. Never
materialize the per-window history tensor (≈21 GB); persist summaries only.

### 2.5 Screening and direction statistics

- **Detection**: `sup_ℓ |C[i,j,ℓ]|` over positive lags.
- **Direction**: `D[i,j] = Σ_ℓ (|C[i,j,ℓ]| − |C[j,i,ℓ]|)` — absolute values,
  because signed skew scores conflate direction-of-lead with sign-of-relationship
  (a *negative-beta* lead reads backwards otherwise; verified by simulation in
  panel critique). Relationship sign is carried as a separate attribute.
- **Lévy-area screen**: per-segment-restarted cumulative paths P give
  `Â = (PᵀΔP − ΔPᵀP)/2T` in two GEMMs (seconds) — parameter-free shortlist,
  numerically identical to uniform-weight aggregation of skew cross-covariances
  (verified to machine precision in critique). Use |·| variant for direction.

### 2.6 Confirmatory estimators (on shortlisted pairs only)

1. **HRY shifted Hayashi–Yoshida** — asynchronicity-native (day-only CFFEX vs
   night metals): overlap-interval cross-covariance with *j*'s intervals
   shifted by θ; θ̂ = argmax |C(θ)|. Needs compiled kernels (numba) — pure
   Python is ~2 days/core; numba is ~3 min across 8 cores. Run at edge-admission
   events, not per window.
2. **Binned-lag predictive regression (the tradeability estimator)**:
   `x_j(t+1) ~ Σ_b β_b·X̄_i(t;b) + own-lags_j + sector factor + session dummies`,
   lag bins {1, 2–5, 6–15, 16–30, 31–60} bars, HAC (Newey–West, bandwidth ≈ one
   trading day) + trading-day-block bootstrap. This is what GP ultimately
   consumes: signed, sized, *incremental* predictability net of momentum and beta.
3. **Phase Slope Index** (frequency domain) as a common-factor-immune
   cross-check: purely contemporaneous mixing ⇒ real coherency ⇒ PSI = 0.

### 2.7 Network layer (linear algebra, all O(N³) ≈ instant)

Pair-score matrix S = symmetric (co-movement) + skew-symmetric A (lead-lag flow).
- **HodgeRank**: solve graph-Laplacian least squares for a global leader score
  s; the **curl ratio** ‖A − grad s‖²/‖A‖² diagnoses whether a global hierarchy
  exists (high curl ⇒ trade pairs, not a ranking).
- **Hermitian clustering** (H = iA, Cucuringu et al.): complex eigenvectors
  cluster products into lead-lag blocks (black chain, oil chain, oilseeds);
  entry phases order nodes along the flow.
- **RMT preprocessing**: Marchenko-Pastur clipping of C₀ eigenvalues; null edge
  for singular values of C_ℓ from circular-shift simulation.

---

## 3. Statistical inference (the part most published results get wrong)

### 3.1 The right null

H0 is **"no lead-lag GIVEN contemporaneous dependence"**, not independence.
Bid-ask bounce (minute-return autocorrelation φ ≈ −0.02..−0.05) leaks
contemporaneous correlation ρ₀ into small lags mechanically: ρ(1) ≈ ρ₀·φ gives
|t| ≈ 5–20 for tightly co-moving sector pairs with *zero* genuine lead-lag.
Naive day-shift permutations (which destroy ρ₀) therefore "discover" every
correlated sector-mate. Fixes:

- Test **asymmetry statistics** — ρ(ℓ)−ρ(−ℓ), the Huth–Abergel LLR, or the
  binned-ADL Wald with own-lag and factor controls — which are null-centered
  under symmetric leakage.
- Calibrate with permutation schemes that **preserve contemporaneous
  dependence**: day-block time-reversal of one series, or day-pair-preserving
  reshuffles; and/or the full Bartlett variance *with cross-terms*
  ρ_xy(k+ℓ)ρ_xy(k−ℓ).

### 3.2 Permutation mechanics

Shift **value and mask jointly**, by whole **exchange trading days** (night
bars belong to the *next* trading day), **within strata of identical session
templates** (pre/post night-session launch, COVID suspension, etc.) — otherwise
the null misaligns the intraday vol U-shape and changes n[i,j,ℓ] itself.
Decimate observed and null statistics on the *same* lag grid.

### 3.3 Multiplicity budget

sup-over-lags collapses (pair, lag) hypotheses → N(N−1)/2 ≈ 4.5k pair-level
p-values (Westfall–Young style). Then two pre-registered families:
- **Family A (economic priors, declared up front)**: ~50–100 supply-chain /
  substitution / cross-market pairs → Benjamini–Hochberg at q = 0.05.
- **Family B (blind all-pairs)**: Benjamini–Yekutieli at q = 0.05–0.10
  (c(M) ≈ 9.7 — deliberately harsh; survivors are strong).

Every scan (windows × pairs × lags actually examined) is logged into the
estimation-factory trial ledger so the downstream DSR / CPCV admission gate
haircuts for the *detection* search too, not just the GP search.

### 3.4 Stability and lifecycle

Rolling windows W ∈ {60, 125, 250} trading days stepped monthly. Note
overlapping windows: 250 daily-stepped 20-day windows ≈ **12 independent
looks** — stability gates count effective independent windows, not raw window
count. Per-edge: sign consistency, lag-mode concentration (±2 bars), AR(1)
half-life of edge strength, day/night split, vol-tercile split, era tags.
Admission requires same-sign out-of-window confirmation; retirement on 3-month
rolling-t decay. New listings (<3y, GFEX) follow an explicit reduced-window
policy and are flagged.

### 3.5 Artifact blacklist

Known-fake edge classes are blacklisted so GP cannot rediscover them:
stale-price edges (liquid leads illiquid mechanically — require the *illiquid*
leg to pass a staleness gate), limit-cascade edges (sector-wide limit locks with
staggered unlock), roll-date artifacts (roll dates cluster sector-wide; ±1–2 day
exclusion), open-gap effects (lags spanning session breaks route to a separate
overnight-gap model, never pooled with intra-session lags).

---

## 4. Infrastructure layers (build order)

```
futures_miner/
├── data/        L0  ingest: CSV → parquet (~AutoLLM_data/futures/), filters,
│                    symbol case-normalization, trading_date mapping
│                L1  session-calendar inference (per product per date, from data);
│                    dominant-contract calendar; back-adjusted continuous series;
│                    limit-lock detection; PIT universe/liquidity table
├── panel/       L2  aligned wide panels (union clock + masks), prewhitening,
│                    resample pyramid, factor partialing — memory-mapped float32
├── leadlag/     L3  segment-blocked GEMM scanner (3-Gram normalization),
│                    Lévy-area screen, asymmetry stats, permutation engine,
│                    HY/HRY (numba), binned-ADL + HAC, PSI, network layer
│                L4  edge table: append-only parquet, as-of dated, config-hashed
│                    (leader, follower, lag, strength, sign, q-value, stability,
│                    half-life, regime/era tags) + trial ledger
├── gp/          L5  cross-asset GP terminals reading the edge table as-of:
│                    LEAD_RET(j,r,b) (r-th leader's binned return),
│                    LEAD_OI_CHG, LEAD_VOLRATIO, LEAD_STALE (staleness age),
│                    LL_STRENGTH (edge health) — lag-clamped to validated ℓ*;
│                    warm-start engine: frozen-structure GP (paper) seeded per
│                    validated edge; long/short futures backtest with per-product
│                    costs, limit/tradability vetoes
└── tests/           unit + golden-master; reproducibility: re-run any historical
                     as-of from config hash ⇒ diff-equal stored rows
```

Key integration discipline (from critiques): **one sanctioned code path** for GP
to touch foreign series (the lag-aligned terminal layer) — eob-only indexing,
≥1-bar execution buffer, embargo extended by max(window + ℓ*), pair table frozen
at month m−1 for use in month m.

### Memory discipline (hard rule from the user)

The machine has 16 GB and the raw data is 68 GB. Every stage obeys:

1. **Batch by batch, always.** Ingest and processing loop over bounded chunks —
   one product, one year, or one month at a time — and write each chunk's
   result to disk before touching the next. No stage ever holds the full
   dataset in RAM.
2. **Per-stage peak RAM budget: ~4 GB.** Leaves room for the OS and the editor.
   Each stage is tested on a small batch first and its peak is measured before
   a full run.
3. **Lazy reads only.** Polars `scan_parquet` + streaming collect, or explicit
   chunk loops. Eager full-table reads are banned in pipeline code.
4. **float32 for panels, memory-mapped wide arrays** for the scanner; masks
   bit-packed. Intermediate tensors are summarized then discarded — the
   per-window history tensor (~21 GB) is never materialized.
5. **Subsets are fine.** Development and validation run on a slice (e.g. 10
   liquid products × 3 years) before any full pass. Full passes are
   opt-in, run one clock at a time.

### Compute budget (M1, 8 cores, 16 GB — all verified by critics)

| Task | Cost |
|---|---|
| Full-history tensor pass (one clock) | ~1–2 min (segment-blocked GEMM, ~0.3–1 TFLOPS) |
| Rolling daily-stepped 20y scan | ~10–30 min |
| Permutation null (200 reps, decimated lags, survivors only) | ~10–30 min → overnight with slop |
| HY/HRY on ~500 pairs (numba) | ~minutes |
| Wide 1m panel in RAM (float32) | ~1.3 GB (X) + 1.3 GB (M) |
| Raw parquet on disk | ~4–6 GB |

The binding constraint is **calendar/roll/limit engineering, not FLOPs**.

---

## 5. Connection to the GP warm-start paper

The paper (arXiv:2412.00896) freezes a validated alpha's *tree structure* and
searches only node contents (restricted crossover at identical positions, point
mutation, duplicate rejection, elitism-1). Transfer: each validated lead-lag edge
becomes a frozen **structure template**, e.g.

```
signal_j = f( LEAD_RET(j, r=1, b), own_feature_j, window )
```

with GP filling in f's node contents (which transformation, which bin, which
follower feature, thresholds) per structure — one small run per edge, many runs
in parallel, exactly the paper's decorrelation mechanism. Fitness = cost-adjusted
long/short Sharpe on the follower (AlphaGen's known gap — post-hoc costs — fixed
by putting costs inside fitness).

---

## 6. Open decisions for the user

1. Liquidity floor for the v1 universe (suggest: top ~50 products by 1m ADV).
2. Primary scan clock (suggest: 5m for discovery, 1m confirmation on survivors).
3. Whether factor-partialed or raw edges gate GP terminals (suggest: run both,
   admit intersection first).
4. Family-A prior pair list — user's economic knowledge belongs here.
