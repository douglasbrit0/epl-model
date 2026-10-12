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
