# EPL Prediction Model

A standalone Python model and daily data pipeline for Premier League scoreline distributions. The contract in `output/` is designed for consumption by DougBoard; no dashboard code is included here.

## Data Sources And Readiness

- Historical results and closing odds come from football-data.co.uk `E0.csv` seasons 2017/18 onward. The ingestion prefers `PSCH`/`PSCD`/`PSCA` and falls back to `AvgCH`/`AvgCD`/`AvgCA` where Pinnacle closing prices are absent.
- Upcoming fixtures come from football-data.org v4 (`PL`). Set `FOOTBALL_DATA_API_KEY` in the environment.
- xG is optional and the model runs goals-only by default. Understat's robots.txt disallows crawling, so no scraper is included. If you have a permitted npxG export, put it at `data/raw/understat_npxg.csv` (`kickoff_utc,home,away,npxg_home,npxg_away`) or provide its URL through the `UNDERSTAT_NPXG_URL` Actions secret; when present it must join every historical match. Without it, runs log a warning and continue goals-only.
- The football-data.org fixture endpoint does not provide future closing odds. Market fields are emitted only when odds are actually present in the fixture input; no odds source is invented.

- Each daily run first saves two timestamped, append-only snapshots on the `data` branch under `logs/`. Neither is used by the model yet, and a failure in either only logs a warning:
  - FPL player availability (status, chance of playing, injury news) from the public bootstrap-static API, in `logs/fpl_snapshots/`.
  - When the `ODDS_API_KEY` secret is set, prices from The Odds API in `logs/odds_snapshots/`. It uses about 12 of the free tier's 500 monthly credits per run:
    - EPL 1X2 and Over/Under from UK and EU bookmakers (including Pinnacle) and US exchanges (Kalshi, Polymarket, which are the venues accessible from Texas).
    - Championship 1X2 and Over/Under from UK and EU bookmakers.
    - EPL season (outright) markets from EU bookmakers and US exchanges.

The sources' exact team names are normalized through `data/team_name_map.csv`. Unknown teams and unmatched xG records raise errors. Raw CSVs and xG exports belong in `data/raw/`; processed match data is parquet under `data/processed/`.

## Setup

Python 3.10 or newer is required.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set `FOOTBALL_DATA_API_KEY` before the production command. Completed football-data.co.uk seasons are downloaded once and cached in `data/raw/`; the latest two are refreshed each run, falling back to the cached copy if the download fails.

```sh
python run_backtest.py
python run_daily.py
python -m pytest
```

`run_backtest.py` (about 30 s) writes walk-forward predictions, model/market metrics, calibration bins, flat-stake edge tables with bootstrap intervals, and a reliability plot into `reports/`, plus the committed `reports/backtest_summary.json` that feeds `performance.json`. Results, the audit and every model change are in `reports/audit.md` and `reports/experiments.md`.

## Model

The baseline is a Dixon–Coles weighted Poisson model. For each match, the two log-rates combine the league mean, home advantage, attacking strength, and opposing defensive strength. Attack and defence each use a sum-to-zero parameterization. The likelihood applies the standard low-score `rho` correction, with exponential recency weights `exp(-xi * days_ago)`; SciPy L-BFGS-B estimates the parameters using an analytic gradient. `xi` is selected on a chronological holdout. Seasons run July–June, except that the 2019/20 restart (June–August 2020) belongs to 2019/20.

When an npxG export is supplied, the xG variant fits `alpha * npxG + (1 - alpha) * goals` and selects `alpha` by log-loss on actual goals over a chronological holdout.

Promoted teams: the prior is the average first-season rating of past newcomers relative to established teams, from completed seasons before the fit cutoff, with precision equal to the inverse variance of those ratings. Teams that did not play the previous season are fitted with that Gaussian prior as a penalty, so their ratings move with their results rather than being frozen. Matches from a team's earlier EPL spells are treated as a separate team, so a returning side starts from the prior. Teams with no data take the prior directly.

The model builds scoreline probabilities through 10 goals per side. Match outcome, over 2.5, BTTS, expected goals, and top scorelines are aggregated from that distribution. The contract's 7x7 scoreline matrix contains the original probabilities for scores 0–6; it is not renormalized, so the remaining probability mass represents scorelines above six.

Walk-forward evaluation groups by matchweek when an input `matchweek` field exists and excludes the entire week from training. The football-data.co.uk archive does not consistently include matchweek, so its fallback groups matches into fixed seven-day blocks within each season. Closing 1X2 odds are de-vigged by both the power method and Shin's method. Flat-stake edge results are diagnostic, not a recommendation.

## DougBoard Contract

`schemas/` contains Draft 2020-12 JSON Schemas. Before writing output, the publisher validates all five files and stages a complete snapshot; failed validation leaves the previous local snapshot untouched.

- `latest.json`: generation time, model version, and last data date.
- `fixtures.json`: upcoming fixtures, 1X2/over/BTTS probabilities, expected goals, top five scorelines, 0–6 scoreline grid, and optional market probabilities/edges.
- `ratings.json`: current team effects and dated history.
- `season.json`: 10,000-run title/top-four/top-six/relegation probabilities, expected points, and matchweek history.
- `performance.json`: rolling model/market log-loss and RPS, calibration bins, cumulative flat-stake P&L.

Every prediction is appended to `logs/prediction_log.jsonl` with `fixture_id`, `model_version`, `generated_at`, and a stable `prediction_id`; duplicates are rejected and existing entries are never rewritten. New results can be scored through `epl_model.evaluation.track_record.score_prediction_log`.

The GitHub Actions workflow runs at 06:00 UTC daily and supports manual dispatch. Required repository secrets are `FOOTBALL_DATA_API_KEY` and `UNDERSTAT_NPXG_URL`. To publish via Supabase, also configure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`; create the `epl-model` Storage bucket (or set `SUPABASE_BUCKET`). Files are uploaded into `snapshots/<generated_at>/`, and the root `latest.json` is advanced last so incomplete uploads are not selected as current. The data branch still stores a backup snapshot and the append-only log so chart histories and prediction IDs survive daily GitHub runners.

## Known Follow-Ups

1. Choose a permitted xG source (Understat disallows crawling); until then the model is goals-only.
2. Add a genuine historical matchweek identifier where available. football-data.co.uk's E0 files do not consistently expose one, so the evaluator uses documented seven-day within-season blocks as its fallback.
3. See `reports/audit.md` for the current backtest; the model still trails the closing market by about 0.020 log-loss.
