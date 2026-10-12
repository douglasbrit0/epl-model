# Phase 1 audit — epl-model (2026-10-12)

Scope: the code as first pushed to `main`. Nothing in `src/` has been changed yet. Every number here
comes from the **unmodified** model code, run in a scratch harness with one change: a patched copy of
`data/team_name_map.csv` (see B1), without which the backtest cannot load data. The model is
goals-only (alpha = 0): the xG input is absent by decision (see B4).

Supporting files in `reports/audit/` (CSV/PNG are gitignored): per-match walk-forward predictions,
per-season metrics, calibration table, market-gap segments, reliability plot.

## 1. Baseline numbers (original code, goals-only Dixon–Coles)

Walk-forward over 296 seven-day blocks, 2018/19 → 2026/27 (to 2026-09-20), 3,090 matches. Each block is
refit on all matches strictly before its first kickoff. Market = closing 1X2 (Pinnacle where available,
market average otherwise), de-vigged by the power method. "Pre-match" = Pinnacle odds collected before
kickoff (`PSH/PSD/PSA`), the closest thing to opening odds in the archive. Gap = model − closing
market (positive = model worse). 95% CIs come from a paired bootstrap that resamples whole blocks
(4,000 draws).

| Season | n | Model LL | Close LL | Pre LL | Model RPS | Close RPS | LL gap [95% CI] | RPS gap [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 2018/19 | 380 | 0.9142 | 0.8890 | 0.8906 | 0.1920 | 0.1847 | +0.025 [−0.000, +0.054] | +0.007 [−0.001, +0.016] |
| 2019/20* | 314 | 0.9823 | 0.9767 | 0.9785 | 0.2007 | 0.1979 | +0.006 [−0.016, +0.026] | +0.003 [−0.004, +0.009] |
| 2020/21* | 446 | 1.0240 | 0.9922 | 0.9987 | 0.2185 | 0.2098 | +0.032 [+0.012, +0.052] | +0.009 [+0.002, +0.015] |
| 2021/22 | 380 | 0.9508 | 0.9360 | 0.9356 | 0.1944 | 0.1889 | +0.015 [−0.003, +0.031] | +0.006 [+0.000, +0.011] |
| 2022/23 | 380 | 1.0162 | 0.9620 | 0.9662 | 0.2159 | 0.1974 | +0.054 [+0.030, +0.079] | +0.018 [+0.010, +0.027] |
| 2023/24 | 380 | 0.9334 | 0.8979 | 0.9068 | 0.1917 | 0.1801 | +0.036 [+0.011, +0.058] | +0.012 [+0.004, +0.019] |
| 2024/25 | 380 | 0.9800 | 0.9667 | 0.9709 | 0.2011 | 0.1962 | +0.013 [−0.002, +0.029] | +0.005 [−0.000, +0.010] |
| 2025/26 | 380 | 1.0526 | 1.0152 | 1.0173 | 0.2169 | 0.2055 | +0.037 [+0.018, +0.057] | +0.011 [+0.005, +0.018] |
| 2026/27 (partial)† | 50 | 1.0346 | 1.0596 | 1.0562 | 0.1993 | 0.2071 | −0.025 [−0.050, +0.004] | −0.008 [−0.018, +0.003] |
| **All** | 3,090 | **0.9834** | **0.9565** | 0.9601 | **0.2042** | **0.1955** | **+0.027 [+0.019, +0.035]** | **+0.009 [+0.006, +0.011]** |
| 2023/24–2025/26 | 1,140 | 0.9887 | 0.9599 | 0.9650 | 0.2033 | 0.1939 | +0.029 [+0.017, +0.040] | +0.009 [+0.006, +0.013] |

\* 66 post-lockdown 2019/20 matches (July 2020) are labelled 2020/21 by the code's season rule (bug B3).
† 2026/27 has no Pinnacle closing prices in the archive, so its benchmark is the market average. With
n = 50 it isn't evidence of anything.

**Reading:** the model trails the closing market in every full season, by about 0.027 log-loss and
0.009 RPS overall. It also trails the earlier Pinnacle pre-match line, by 0.023 log-loss. For scale,
that gap is larger than the difference between pre-match and closing odds.

## 2. Bugs (ranked by impact)

| # | Severity | Bug | Evidence |
|---|---|---|---|
| B1 | **Blocker** | `team_name_map.csv` lacks football-data's `Nott'm Forest` (it lists `Nottingham Forest`), `Hull` and `Coventry`; football-data.org lacks `Hull City AFC` and `Coventry City FC`. `run_backtest.py` and `run_daily.py` both crash on load. | `AssertionError: Unmapped football_data team names: ['Coventry', 'Hull', "Nott'm Forest"]` |
| B2 | **High** | **Promoted-team prior is applied to the wrong teams, then frozen.** `_derive_promoted_priors` compares the latest season *in the training data* with the one before it. (a) In opening-weekend refits the new season isn't in the data yet, so the prior goes to *last* season's promoted sides; the truly promoted teams get a stale rating from their previous EPL spell, or 0 (league average) if never seen (Ipswich 2024/25, Sunderland 2025/26). (b) From week 2 on, promoted teams use the static prior **for the whole season** and their own results are ignored. Production currently rates Coventry, Hull and Ipswich identically. | Set-logic replay (§5); `promoted_flag` columns in the backtest. Matches involving a promoted team carry 41% of the total log-loss gap on 26% of matches (§4). |
| B3 | **Medium** | **Season labelling** (`month >= 7`) puts the 66 July-2020 restart matches in 2020/21. Knock-on effects: Norwich and Watford were never treated as promoted in 2021/22, Leeds got the prior for a second season, and walk-forward blocks and per-season reporting are shifted. | Backtest season counts 314/446; 2021/22 prior applied to `['Brentford', 'Leeds United']` |
| B4 | **High (dormant)** | **xG blend (alpha) is chosen against blended targets.** `_mean_log_loss` scores the validation tail on `(1−α)·goals + α·npxG`, so higher alpha always "wins" (fractional targets have lower Poisson loss). Given pure-noise npxG, `fit()` picks α = 1. Dormant while goals-only, but `choose_validated_model` would always prefer the xG variant. The xG input is also a hard requirement in both entry points, which conflicts with the goals-only decision. | Synthetic test: same parameters score 2.988 on goals vs 2.677 on α=1 targets; selected α = 1.0 |
| B5 | **High (ops)** | **The daily job cannot finish in time.** One fit takes ~2–3 min (98% of it finite-difference gradients through Python list comprehensions; ~69 parameters → ~70 likelihood evaluations per gradient step). `choose_validated_model` runs ~18 fits goals-only, or ~40 with xG, against `timeout-minutes: 45`. | cProfile: 172.8 s for one 5-candidate fit; median 121 s per walk-forward refit on 8 parallel workers |
| B6 | Medium | **Track record over-weights fixtures.** Every daily prediction of a fixture is logged and scored, so a fixture predicted on 7 days counts 7 times. Production should score the last pre-kickoff prediction. Also, `performance.json` takes backtest metrics from `reports/walk_forward.csv`, which is gitignored and never exists on the Actions runner, so that part is always empty in production. | `track_record.py`, `pipeline._performance_payload` |
| B7 | Low–Med | Season simulation drops postponed fixtures (it only fetches fixtures from today onward), and football-data.org fixtures aren't filtered by status (FINISHED/POSTPONED can appear as "upcoming"). The simulator is also slow: ~1.2 min for 10k runs. | `run_daily.py`, `simulation/season.py` |
| B8 | Low | Hyperparameter selection is noisy: xi flips between 0 and 0.005 across consecutive weeks (0.0: 90 weeks, 0.002: 87, 0.003: 63, 0.001: 43, 0.005: 13). It's selected on scoreline log-likelihood over the last 20% of training data, and rho sometimes hits its −0.2 bound (min −0.199). | Backtest columns `xi`, `rho` |
| B9 | Low | `ratings.json` shows fitted ratings for promoted teams while predictions use the prior. The promoted-prior cache is a module global keyed only by (season, row count). The prior is fit on a single-season scale (sum-to-zero over that season's 20 teams) but applied in the multi-season scale. | Code review |
| B10 | Doc | `walk_forward` says "Monday-start calendar weeks"; the blocks are actually 7-day blocks anchored on 1 July. This is conservative (a block's later matches don't use its earlier results), so not a leak, but slightly pessimistic versus a daily live refit. | Code review |

## 3. Leakage, look-ahead and contamination

No leakage found in model fitting or evaluation.
- `fit` drops everything at or after the cutoff and asserts it; walk-forward cutoffs are the
  block's first kickoff date (normalized), and the existing test covers it.
- In-fit tuning (xi, alpha) uses only a chronological tail of pre-cutoff data. That's leak-free, but
  biased (B4) and noisy (B8).
- Promoted priors only use seasons before the cutoff, and partial-season data only up to it.
- Name mapping is an exact lookup with canonical pass-through; no fuzzy matching, so no cross-team
  contamination. It's just incomplete (B1).
- Closing odds are used only as a benchmark and for the diagnostic P&L, never as an input. Any
  market-blend experiment must use pre-match (`PSH`) or opening odds, never closing.
- Gap: kickoff times are discarded (E0 has `Time` from 2019/20), so the day is the finest grain. That's
  fine for daily refits, but feature work on rest days or the 4-day European window should use times.

## 4. Calibration

- **Base rates are fine:** mean predicted H/D/A is 0.434/0.238/0.328 vs observed 0.438/0.232/0.330.
  No systematic home, draw or away bias.
- **Mildly overconfident overall:** the best single temperature on the log-probabilities is T = 1.09
  (T > 1 means probabilities are too extreme).
- **Draws:** in the top decile of predicted draw probability (mean 0.313) only 23.9% were draws
  (z = −2.8; the market averages 0.284 on the same matches). The model inflates draws in tight,
  low-scoring matchups, likely because rho sits near its bound. This is a model-structure issue, not
  just calibration.
- **Favourites:** well calibrated except the 65–75% band (predicted 0.696, observed 0.630,
  n = 414; the market says 0.684). Above 75% it's fine (0.814 vs 0.830).
- Plot: `reports/audit/baseline_reliability_by_outcome.png` (decile bins with 95% intervals; the market
  on the same bins). Table: `baseline_calibration.csv`.

## 5. Where the model loses to the market

Per-match log-loss gap (model − closing market), paired block bootstrap. Overall gap = +0.027.
Segments overlap (e.g. promoted teams also play in late season), so treat this as triage, not
attribution.

| Segment | n | LL gap [95% CI] | Share of total gap |
|---|---|---|---|
| First-choice GK absent (oracle, 2020/21+) | 451 | **+0.050** [+0.031, +0.068] | 27% |
| Promoted team involved | 788 | **+0.043** [+0.025, +0.061] | **41%** |
| Either side played in Europe 1–4 days before | 381 | **+0.042** [+0.020, +0.062] | 19% |
| Late season (last ~8 rounds) | 640 | **+0.040** [+0.023, +0.056] | 31% |
| Market favourite < 45% (close games) | 948 | +0.038 [+0.023, +0.053] | 43% |
| Key outfielder absent (oracle, 2020/21+) | 963 | +0.037 [+0.023, +0.051] | 43% |
| Big-six vs big-six | 242 | +0.032 [+0.009, +0.057] | 9% |
| No promoted team | 2,302 | +0.022 [+0.013, +0.030] | 59% |
| Early season (first ~6 rounds) | 530 | +0.020 [−0.001, +0.043] | 13% |
| Market favourite > 75% | 339 | +0.004 [−0.018, +0.025] | 2% |

Takeaways for later phases:
1. **Promoted teams** are the biggest single lever, and most of that gap should go just from fixing B2.
2. **Player availability** matters: the gap roughly doubles when the first-choice goalkeeper is out,
   and is clearly larger when a key outfielder is out. This supports Phase 3 item 1 and treating
   goalkeepers separately.
3. **European midweek games** and **late season** (motivation, rotation) are next. Neither is modelled.
4. Big six and early season aren't special once promoted teams are accounted for.

Definitions and data:
- **Key outfielder:** one of the team's top 3 by FPL points over its previous 8 matches, among
  players with ≥ 60% of minutes in that window. Absent = 0 minutes in the match (oracle, i.e. known
  only after the match).
- **First-choice GK:** any goalkeeper with ≥ 60% of minutes over the same window.
- **Data sources:** player minutes from the vaastav/Fantasy-Premier-League archive (MIT code; the data
  itself is from the FPL API, under FPL's terms of use). European fixtures from openfootball (CC0).
- **Coverage gaps:** Champions League is complete for 2017/18–2025/26. Europa League covers only
  2020/21–2024/25 and the Conference League 2021/22–2024/25, so the Europe segment undercounts.

## 6. Dixon–Coles correctness

- **Likelihood and rho correction:** match the original paper's τ(0,0) = 1−λμρ, τ(0,1) = 1+λρ,
  τ(1,0) = 1+μρ, τ(1,1) = 1−ρ.
- **Reference implementation:** checked against penaltyblog 1.0.4 on 2023/24 data, both unweighted and
  with ξ = 0.002 decay. Fitted log-rates agree to ≤ 1.2e-4, weighted mean NLL is identical to 6 d.p.
  (2.987503 / 2.989118), rho agrees to 1e-4 (−0.0201 / −0.0333), and home advantage agrees to 1e-4.
- **Poisson part:** cross-checked against a statsmodels Poisson GLM with sum-to-zero contrasts (home
  advantage 0.1965 vs 0.1966 at ρ ≈ 0).
- **Time decay:** `exp(−ξ·age)` relative to the last training date, normalised by the sum of weights.
  That's equivalent to decay measured from the prediction date, and ξ is per day (Dixon–Coles'
  0.0065 per half-week ≈ 0.0019 per day).
- **Sum-to-zero:** last team = −sum of the others, for both attack and defence. Correct, but the
  constraint spans every team in the training window, including long-relegated ones. So "0" isn't
  "average current EPL team", which matters for unseen teams (B2) and the prior scale (B9).
- **Rates** are clipped to log-rate [−5, 4] and τ is floored at 1e-9. Harmless in practice.
- **Scoreline grid:** 0–10 goals, renormalised; the published 7×7 isn't renormalised, as the README
  documents.

## 7. Runtime profile

| Step | Time | Note |
|---|---|---|
| One parameter fit (8 seasons) | ~35 s | 98% in `approx_derivative` (finite-difference gradient) |
| `fit()` with 5-value xi grid + priors | ~173 s | serial |
| Full walk-forward (296 refits) | 88 min on 8 cores (~12 CPU-hours) | median 121 s/refit, max 390 s |
| `choose_validated_model` (daily) | ~45 min goals-only (est.), ~1.5–2 h with xG | exceeds the 45-min Actions timeout |
| Season simulation, 10k runs | ~1.2 min | per-fixture `rng.choice` in Python |

The obvious fixes are a vectorised likelihood with an analytic gradient, precomputed team indices, and
warm starts. Expect a fit to drop from ~35 s to well under a second.

## 8. Status (2026-10-12, branch `phase1-fixes`)

All of B1–B7 are fixed and B10 is corrected; details and metrics are in `reports/experiments.md`.

- **B1:** name map completed.
- **B2:** promoted teams are fitted with empirical-Bayes shrinkage toward the newcomer prior, earlier
  EPL spells are split off, and never-seen teams take the prior.
- **B3:** the 2019/20 restart is labelled 2019/20.
- **B4:** the xG blend is selected on actual goals, and the xG input is optional (goals-only by default).
- **B5:** analytic gradient; daily pipeline ~5 s, full backtest ~30–55 s.
- **B6:** each fixture is scored once; backtest metrics come from a committed summary.
- **B7:** fixture status filtering, postponed and lagging matches kept, vectorised simulator.
- **B10:** walk-forward docstring corrected.

Still open:
- **B8** (noisy xi selection): Phase 2 nested tuning.
- **B9:** the ratings display shows each team's fitted, shrunk rating, which is now what predictions
  use, so the display/prediction inconsistency is gone. The prior's scale mismatch remains an
  approximation.

New baseline (E2): LL 0.9768, RPS 0.2022. Paired improvement over the original: LL −0.0066
[−0.0108, −0.0025], RPS −0.0020 [−0.0033, −0.0007]. Gap to the closing market: +0.020 LL, +0.0067 RPS.
