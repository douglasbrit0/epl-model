# Experiment log

Judge: walk-forward backtest (`run_backtest.py` protocol). Each 7-day block is refit on all matches
strictly before its first kickoff. Evaluation set: 3,090 matches, 2018/19 → 2026/27 (to 2026-09-20).
Comparisons are paired on identical fixtures; 95% CIs come from a bootstrap that resamples whole
blocks (4,000 draws). "P(not better)" is the bootstrap share of resamples where the change does not
improve. Δ = new − old, so negative is better for both log-loss (LL) and RPS.

Acceptance bar for model changes: LL **and** RPS improve versus production over ≥ 3 full seasons, with
the CI excluding zero. Bug fixes are accepted on correctness, then re-measured to set the new baseline.

---

## E0 — Speed: vectorised likelihood + analytic gradient (accepted, no model change)

- **Hypothesis:** the 2–3 min fits come from finite-difference gradients through Python loops; an exact
  gradient gives the same optimum far faster.
- **Change:** precomputed design arrays, vectorised NLL, analytic gradient (`jac=True`). Same objective,
  same bounds and same optimiser.
- **Checks:**
  - Gradient vs finite differences: ≤ 1e-7.
  - Objective identical at the same parameters (difference 0.0).
  - Fitted models agree to ~1e-5 in scoreline probabilities at 2019 and 2022 cutoffs.
  - At the 2025 cutoff the new fit reaches a lower NLL (−8.6e-6): the original optimiser had stopped
    early on long-relegated teams whose data had decayed away.
- **Metrics:** LL 0.98344 → 0.98361, RPS 0.20420 → 0.20425. The difference comes entirely from the
  decay parameter xi flipping between near-equal candidates in 20 of 296 blocks (audit B8); with the
  same xi, predictions agree.
- **Runtime:** one fit 136 s → 0.4 s; full walk-forward ~12 CPU-hours → 54 s; with the later
  vectorised simulator, the daily pipeline went from ~45+ min (estimated) to 5 s.

## E1 — Bug fixes B1, B3, B4 + promoted-team shrinkage (superseded by E2)

- **Changes:**
  - Name map (B1): added `Nott'm Forest`, `Hull` and `Coventry` (football-data) and `Hull City AFC`
    and `Coventry City FC` (football-data.org).
  - Season labelling (B3): the 2019/20 restart (June–August 2020) now belongs to 2019/20.
  - xG blend (B4): candidates are scored on actual goals, and the xG input is optional.
  - Promoted teams (B2, v1): teams absent from the previous season get a Gaussian prior on attack and
    defence. Its centre is the mean first-season rating of past newcomers relative to established
    teams; its precision is the inverse variance of those ratings (empirical Bayes, no tuning). They
    are fitted with that shrinkage, never frozen. Never-seen teams take the prior instead of 0.
- **Result vs original:** LL −0.0049 [−0.0089, −0.0010], RPS −0.0014 [−0.0027, −0.0002].
- **Problem:** 2023/24 and 2024/25 got significantly worse (LL +0.008 each, CI excluding 0).
  Diagnosis: returning teams' previous-spell data outweighed the prior. Leicester 2024/25 (+1.78 total
  LL) opened near mid-table on its 2022/23 rating instead of the newcomer prior.

## E2 — E1 + split earlier EPL spells (accepted → **new baseline**)

- **Hypothesis:** a team's previous EPL spell, which ended in relegation, says little about its
  strength after promotion. Treating the returning team as a new entity lets the prior plus its
  current-spell results set the rating.
- **Change:** matches before a team's current unbroken run of EPL seasons are renamed
  "<team> (earlier spell)". Teams absent from both the current and previous season count as earlier
  spells entirely.
- **Result vs original baseline:**

| Season | Original LL | New LL | ΔLL [95% CI] | Original RPS | New RPS | ΔRPS [95% CI] |
|---|---|---|---|---|---|---|
| 2018/19 | 0.9142 | 0.8998 | −0.0144 [−0.0376, +0.0066] | 0.1920 | 0.1879 | −0.0041 [−0.0113, +0.0024] |
| 2019/20 | 0.9823 | 0.9752 | −0.0070 [−0.0144, −0.0006] | 0.2007 | 0.1984 | −0.0023 [−0.0046, −0.0002] |
| 2020/21 | 1.0240 | 1.0141 | −0.0099 [−0.0187, −0.0000] | 0.2185 | 0.2161 | −0.0024 [−0.0052, +0.0007] |
| 2021/22 | 0.9508 | 0.9501 | −0.0008 [−0.0120, +0.0107] | 0.1944 | 0.1936 | −0.0008 [−0.0044, +0.0028] |
| 2022/23 | 1.0162 | 1.0113 | −0.0049 [−0.0144, +0.0043] | 0.2159 | 0.2142 | −0.0017 [−0.0048, +0.0014] |
| 2023/24 | 0.9334 | 0.9347 | +0.0013 [−0.0072, +0.0101] | 0.1917 | 0.1919 | +0.0002 [−0.0023, +0.0028] |
| 2024/25 | 0.9800 | 0.9847 | +0.0047 [−0.0000, +0.0097] | 0.2011 | 0.2027 | +0.0016 [+0.0001, +0.0032] |
| 2025/26 | 1.0526 | 1.0321 | −0.0205 [−0.0344, −0.0077] | 0.2169 | 0.2106 | −0.0064 [−0.0107, −0.0024] |
| **All (3,090)** | 0.9834 | **0.9768** | **−0.0066 [−0.0108, −0.0025]** | 0.2042 | **0.2022** | **−0.0020 [−0.0033, −0.0007]** |
| 2023/24–2025/26 | 0.9887 | 0.9838 | −0.0048 [−0.0109, +0.0008] | 0.2033 | 0.2017 | −0.0015 [−0.0034, +0.0002] |

  Season rows use the original (pre-B3) season labels, so 2019/20 and 2020/21 differ by the 66 restart
  matches. Versus E1, E2 improves LL by −0.0017 [−0.0035, +0.0001] overall and by −0.0034
  [−0.0061, −0.0008] over 2023/24–2025/26.
- **Decision:** accepted as the bug-fix baseline. P(not better) is 0.0005 for LL and 0.0008 for RPS
  overall. Caveat: 2024/25 is still slightly worse (borderline on LL, significant on RPS). All three
  promoted sides went straight down that season, which the old frozen prior happened to suit.
- **New baseline vs closing market:**
  - LL gap +0.0203 [+0.0130, +0.0277] (was +0.0269); RPS gap +0.0067 [+0.0044, +0.0091] (was +0.0087).
  - The promoted-team gap fell from +0.043 to +0.020, in line with other matches.
  - Calibration temperature 1.09 → 1.05.
  - Largest remaining gaps: first-choice GK absent +0.041, key outfielder absent +0.033, European game
    in the previous 1–4 days +0.033, market favourite < 45% +0.032.
- Files: `reports/audit/baseline2_*.csv` and `reports/audit/baseline2_reliability_by_outcome.png`.

## Pipeline fixes (no effect on backtest metrics)

- **B6:** the track record scores only the last pre-kickoff prediction per fixture and model version.
  The backtest part of `performance.json` now comes from a committed `reports/backtest_summary.json`.
- **B7:**
  - Upcoming fixtures are filtered to SCHEDULED/TIMED and future kickoffs.
  - The season simulation uses the whole season's fixture list minus pairings already in the results
    history, so postponed matches and results lagging in football-data.co.uk are simulated, not dropped.
  - The simulator is vectorised: 10k seasons in 0.5 s instead of ~6 min, statistically equivalent to
    the original (max 2.6σ over 60 team-probabilities at 20k runs).
- **Robustness:** completed seasons are downloaded once. Only the latest two are refreshed, and a
  failed refresh falls back to the cached file with a warning.
- **Sparse fits:** `scoreline_matrix` clips log-rates to the likelihood's bounds, which prevents NaN
  predictions in the archive's first weeks.

---

# Phase 2 — optimisation and model structure

All Phase 2 comparisons are paired against the production model at the time: E2 for E3, then E3 for
everything after. Every new hyperparameter is nested-tuned: season S uses the value with the best 1X2
log-loss over evaluated seasons before S, and the first season uses the "off"/default value.

## E3 — Nested walk-forward tuning of decay and promoted-prior strength (accepted → **production**)

- **Hypothesis:** the per-refit choice of xi (best of 5 on the last 20% of training data, scored on
  scoreline likelihood) is noisy and flips week to week (audit B8). Choosing fixed settings by
  out-of-sample 1X2 log-loss on earlier seasons is more stable, and also tunes the promoted-prior
  strength, which E2 set by empirical Bayes.
- **Change:**
  - Grid: xi ∈ {0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0065} ×
    prior_scale ∈ {0, 0.125, 0.25, 0.5, 1, 2, 4} (70 configurations, each a full walk-forward).
  - Season S uses the best configuration on seasons < S; 2018/19 uses the Dixon–Coles default
    (xi 0.002, scale 1).
  - Production uses the best over all evaluated seasons, saved in `reports/tuning.json` by
    `run_backtest.py` and read by the daily job.
- **Selections:** 2019: (0.001, 0) · 2020: (0.001, 0.5) · 2021: (0.002, 0.5) · 2022: (0.0015, 0.5) ·
  2023–2026: (0.0025 or 0.002, 0.5). Production: **xi 0.0025, prior_scale 0.5**.
- **Grid history (disclosed):** the first grid stopped at prior_scale 0.5, and the selection kept
  landing on that edge, so the grid was extended to {0, 0.125, 0.25}. With the first grid the
  result was LL −0.0029 [−0.0053, −0.0007], RPS −0.0010 [−0.0018, −0.0003]. The extended grid is
  the one reported and used. Its 2019/20 choice, made on one season of history, picked
  prior_scale 0 and cost 2019/20 some accuracy.

| Season | E2 LL | E3 LL | ΔLL [95% CI] | E2 RPS | E3 RPS | ΔRPS [95% CI] |
|---|---|---|---|---|---|---|
| 2018 | 0.8998 | 0.8994 | -0.0004 [-0.0042, +0.0032] | 0.1879 | 0.1878 | -0.0001 [-0.0012, +0.0009] |
| 2019 | 0.9699 | 0.9755 | +0.0056 [-0.0059, +0.0192] | 0.1983 | 0.2005 | +0.0022 [-0.0016, +0.0066] |
| 2020 | 1.0262 | 1.0225 | -0.0037 [-0.0088, +0.0018] | 0.2193 | 0.2178 | -0.0015 [-0.0031, +0.0001] |
| 2021 | 0.9501 | 0.9521 | +0.0020 [-0.0037, +0.0077] | 0.1936 | 0.1943 | +0.0007 [-0.0011, +0.0025] |
| 2022 | 1.0113 | 1.0026 | -0.0087 [-0.0182, +0.0001] | 0.2142 | 0.2110 | -0.0032 [-0.0063, -0.0001] |
| 2023 | 0.9347 | 0.9272 | -0.0075 [-0.0183, +0.0032] | 0.1919 | 0.1894 | -0.0025 [-0.0061, +0.0010] |
| 2024 | 0.9847 | 0.9806 | -0.0041 [-0.0079, -0.0004] | 0.2027 | 0.2015 | -0.0012 [-0.0024, -0.0001] |
| 2025 | 1.0321 | 1.0298 | -0.0023 [-0.0054, +0.0009] | 0.2106 | 0.2098 | -0.0007 [-0.0018, +0.0003] |
| 2026 (partial) | 1.0216 | 1.0209 | -0.0007 [-0.0154, +0.0178] | 0.1962 | 0.1959 | -0.0003 [-0.0041, +0.0043] |
| **All** | 0.9768 | **0.9745** | −0.0024 [−0.0052, +0.0004] | 0.2022 | **0.2014** | −0.0008 [−0.0017, +0.0001] |
| **2023/24–2025/26** | 0.9838 | **0.9792** | **−0.0046 [−0.0088, −0.0007]** | 0.2017 | **0.2002** | **−0.0015 [−0.0029, −0.0002]** |

- **Decision:** accepted, with a caveat.
  - It clears the bar on the pre-specified latest-three-full-seasons window (LL and RPS CIs both
    exclude 0) and improves in 6 of 8 full seasons.
  - Over all seasons the improvement is marginal: P(not better) is 0.049 for LL and 0.048 for RPS,
    and the CIs just include 0.
  - It also makes production deterministic: one fixed configuration instead of a weekly re-pick.
- **Runtime:** `run_backtest.py` now takes ~3 min (70 walk-forwards on 8 cores).

## E4 — Team-specific home advantage with shrinkage (rejected)

- **Change:** each team gets a home-advantage deviation with a Gaussian penalty of precision τ ∈
  {off, 3, 10, 30, 100, 300}, nested-selected.
- **Result vs E3:** LL +0.0000 [−0.0005, +0.0006], RPS +0.0000 [−0.0002, +0.0002]; latest three
  seasons LL −0.0001 [−0.0003, +0.0001].
- **Why rejected:** no measurable effect. The fitted deviations are plausible (Villa, City and
  Newcastle strongest at home), but with ~19 home games a season they are too noisy to beat the
  pooled value.

## E5 — Calibration layer (rejected, both variants)

- **E5a, coherent:** shrink team effects toward the established-team average at prediction time,
  factor k ∈ {0.85 … 1.1}, nested-selected. Because it still produces a Dixon–Coles grid, all
  markets stay consistent. Result: LL +0.0004 [−0.0001, +0.0010], RPS +0.0001 [−0.0001, +0.0003].
  The selection settles on k = 0.95.
- **E5b, post-hoc temperature scaling of 1X2:** not scoreline-coherent, included for reference.
  Temperature fitted on earlier seasons (0.94–1.06). Result: LL +0.0003 [−0.0003, +0.0009], RPS
  +0.0001 [−0.0001, +0.0002].
- **Why rejected:** after the E2 fixes the model is close to calibrated (in-sample T = 1.05), and a
  walk-forward recalibration only adds estimation noise.

## E6 — Partial pooling of all team effects (hierarchical MAP; rejected)

- **Change:** a Gaussian (ridge) prior on every attack and defence effect, precision ∈ {0, 0.5, 1,
  2, 5, 10}, nested-selected. This is the posterior mode of a hierarchical model with a common
  team-effect variance. A full PyMC version was not run: it would need ~330 MCMC refits per
  backtest and a heavy new dependency, and its point predictions differ from this MAP mainly
  through posterior averaging.
- **Result vs E3:** LL +0.0000 [−0.0008, +0.0008], RPS +0.0000 [−0.0002, +0.0002]; latest three
  seasons LL +0.0011 [−0.0006, +0.0027].
- **Why rejected:** the decay weights and promoted-team prior already provide the useful shrinkage.

## E7 — Dynamic ratings: Gaussian random walk on attack/defence (rejected)

- **Change:** `models/dynamic.py`. A state-space model fitted by MAP:
  - Every team's attack and defence follow a random walk over 28-day blocks; the league scoring
    level and home advantage follow stiffer walks (so, for example, the empty-stadium season is
    absorbed).
  - All matches count fully; there is no exponential decay.
  - Steps across a season boundary can have extra variance (×1 or ×4) and mean reversion (φ = 1
    or 0.85).
  - Step precision ∈ {100, 300, 1000}. 12 configurations, nested-selected; promoted-team prior and
    earlier-spell split as in production.
- **Result vs E3:** LL +0.0005 [−0.0013, +0.0023], RPS −0.0000 [−0.0006, +0.0005]; latest three
  seasons LL +0.0007 [−0.0018, +0.0033]. Only 2022/23 improves significantly.
- **Why rejected:**
  - No gain.
  - Selection always went to the stiffest walk (precision 1000), and looser walks were clearly
    worse (in-hindsight LL 0.980–0.983 at 100 vs 0.974 at 1000). The data favours slowly moving
    ratings, which exponential decay already provides.
  - Each backtest takes ~82 min vs seconds. The grid edge was not extended: the trend points back
    to the static model.

## E8 — Goal correlation: bivariate Poisson and Frank copula (rejected)

- **Hypothesis:** the draw over-prediction in tight matches (top decile: predicted 0.315, observed
  0.269) comes from the Dixon–Coles low-score correction; a different dependence structure could
  fix it.
- **Change:** keep the production model's expected goals, and replace the low-score correction
  with (a) a bivariate Poisson with a shared component λ3, with marginal means preserved, or
  (b) a Frank copula on the Poisson marginals. The single dependence parameter is fitted by
  weighted MLE on the training data at every refit.
- **Results vs E3:**
  - Bivariate Poisson: LL −0.0008 [−0.0018, +0.0003], RPS −0.0000 [−0.0001, +0.0000]; latest
    three seasons LL +0.0007 [−0.0008, +0.0022].
  - Frank copula: LL −0.0005 [−0.0019, +0.0008], RPS −0.0000 [−0.0002, +0.0001]; latest three
    seasons LL +0.0013 [−0.0003, +0.0030].
- **Why rejected:**
  - Neither clears the bar on either window.
  - Both narrow the top-decile draw gap (to 0.299/0.275 and 0.296/0.259), but they lower the
    average draw probability below the observed rate (0.228 and 0.225 vs 0.232), so the gains
    cancel.
  - The draw bias in tight games is not a dependence-structure problem. It more likely reflects
    match context (motivation, absences), which is the Phase 3 feature work.

## Phase 2 summary

| Experiment | Decision | ΔLL vs production, all [95% CI] | ΔLL, 2023/24–2025/26 |
|---|---|---|---|
| E3 nested tuning (xi, prior strength) | **accepted** | −0.0024 [−0.0052, +0.0004] | **−0.0046 [−0.0088, −0.0007]** |
| E4 team home advantage | rejected | +0.0000 [−0.0005, +0.0006] | −0.0001 [−0.0003, +0.0001] |
| E5a coherent recalibration | rejected | +0.0004 [−0.0001, +0.0010] | +0.0003 [−0.0009, +0.0015] |
| E5b 1X2 temperature scaling | rejected | +0.0003 [−0.0003, +0.0009] | +0.0002 [−0.0008, +0.0011] |
| E6 partial pooling (hierarchical MAP) | rejected | +0.0000 [−0.0008, +0.0008] | +0.0011 [−0.0006, +0.0027] |
| E7 dynamic random-walk ratings | rejected | +0.0005 [−0.0013, +0.0023] | +0.0007 [−0.0018, +0.0033] |
| E8a bivariate Poisson | rejected | −0.0008 [−0.0018, +0.0003] | +0.0007 [−0.0008, +0.0022] |
| E8b Frank copula | rejected | −0.0005 [−0.0019, +0.0008] | +0.0013 [−0.0003, +0.0030] |

Production after Phase 2: E3. LL 0.9745, RPS 0.2014 (2018/19 → 2026/27).
- **Gap to closing market:** about +0.018 LL.
- **Conclusion:** the structure of the goal model is close to its ceiling. The remaining gap must
  come from information the model does not have (Phase 3).
- **Reproduction:** every experiment can be re-run with the scripts in `experiments/` (see its
  README).

---

# Market-efficiency tests (can any strategy beat the betting market?)

Scripts: `experiments/market_tests.py`, `experiments/championship.py`, `experiments/line_shopping.py`.
Prices are Pinnacle pre-match (`PSH…`, collected days before kickoff) and closing (`PSCH…`), with
market-average and Bet365 fallbacks; Over/Under uses `P>2.5`/`PC>2.5`. Blends are fitted only on
seasons before the one evaluated. P&L is per 1-unit stake, with a block bootstrap by matchweek.
CLV = log(price taken / Pinnacle closing price); positive means the price moved our way.

## M1 — Does the model add information to the market? (EPL 1X2: no)

- **Test:** walk-forward conditional-logit blend, score = a·log p_market + b·log p_model + c.
- **Model weight b by season** (fitted on earlier seasons): 0.30, 0.36, 0.22, 0.17, 0.04, −0.00,
  0.00, −0.01. It fell to zero from 2023 on.
- **Out of sample:** the blend is worse than the market alone.
  - vs closing: LL +0.0039 [+0.0008, +0.0070], RPS +0.0009 [+0.0003, +0.0015].
  - vs pre-match: LL +0.0038 [+0.0008, +0.0069].

## M2 — Betting before the line moves (EPL 1X2: loses)

- Flat stakes at the Pinnacle pre-match price when expected value clears 0/2/5%.
  - Blend: ROI −6.0% / −5.0% / −3.3%; CLV −0.6% / −0.4% / −0.2%.
  - Model only: ROI −1.6% / −1.8% / −2.8%; CLV −0.4% / −0.4% / −0.3%.
- All P&L CIs include 0, but every CLV is negative: the market moves against these bets.

## M3 — Over/Under 2.5 (EPL: no edge)

- **Blend:** model weight 0.07–0.27, but the blend doesn't beat the market out of sample (LL vs
  closing +0.0017 [−0.0001, +0.0035]).
- **Bets:** ROI −0.4% to −5%; CLV about 0.

## M4 — Championship (E1), same model with two-way newcomer priors (no edge)

- **Setup:** nested-tuned (production xi 0.002, prior_scale 0.5). Clubs relegated from the Premier
  League get their own prior (+0.18 attack, +0.17 defence vs the Championship average); League One
  promotees get theirs (−0.10, −0.14). Relegation status comes from the previous season's EPL table.
- **1X2:** model LL 1.0534 vs closing market 1.0375. Model weight in the blend is negative in every
  season (−0.02 to −0.19). Blend vs closing: LL +0.0015 [+0.0002, +0.0027].
- **Bets:**
  - 1X2: ROI −4% to −13%, CLV −0.7% to −1.2%.
  - Over/Under: ROI −3.5% to −10%. One cell, the blend at EV > 5%, made +20% ROI, but on only
    44 bets [−3.4, +21.8 units] with CLV −1.8%. It is one of 12 strategies tried: noise.
- The Championship market is, if anything, less beatable by this model than the EPL.

## M5 — Line shopping against Pinnacle, no model (inconclusive with this data)

- **Rule:** bet the best available pre-match price (football-data `Max*`) when it beats
  Pinnacle's de-vigged pre-match price by the threshold.

| | Edge > 0% | > 2% | > 4% | > 6% |
|---|---|---|---|---|
| EPL ROI (bets) | +0.9% (2,792) | −1.7% (690) | +15.7% (209) | +36.7% (77) |
| EPL mean CLV | +3.9% | +6.7% | +11.3% | +16.1% |
| Championship ROI (bets) | −3.9% (2,435) | −1.3% (550) | −20.1% (116) | −46.8% (23) |
| Championship mean CLV | +3.6% | +5.5% | +7.8% | +9.2% |

- **Every P&L CI includes 0.** CLV this large alongside losses in the Championship suggests many
  `Max*` quotes were not really available: stale lines, palpable errors that would be voided, or
  exchange prices before commission.
- **Verdict:** this approach is the only one tested that gets positive CLV. It cannot be validated
  on football-data's best-price column; it needs timestamped quotes from bookmakers you can actually
  bet with. Bookmakers also restrict accounts that do this.

## Conclusion

A goals-based model, however well tuned, carries no information that the EPL or Championship
1X2/totals markets lack. Feature work that only re-derives public information (form, rest,
injuries already in the news) is unlikely to change that.

Routes that could produce an edge, all needing new data:
1. **Timing:** bet before the market prices team news (needs timestamped odds and a news/lineup feed).
2. **Line shopping** with verified, timestamped soft-book prices (M5).
3. **Genuinely private information or modelling:** e.g. tracking-data models.

None is achievable with the free data currently in the repo.
