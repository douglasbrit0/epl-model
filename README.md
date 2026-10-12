# EPL Prediction Model

A standalone Python model and daily data pipeline for Premier League scoreline distributions. The contract in `output/` is designed for consumption by DougBoard; no dashboard code is included here.

## Data Sources And Readiness

- Historical results and closing odds come from football-data.co.uk `E0.csv` seasons 2017/18 onward. The ingestion prefers `PSCH`/`PSCD`/`PSCA` and falls back to `AvgCH`/`AvgCD`/`AvgCA` where Pinnacle closing prices are absent.
- Upcoming fixtures come from football-data.org v4 (`PL`). Set `FOOTBALL_DATA_API_KEY` in the environment.
- Understat is the intended xG source. Its public site is reachable, but match-level non-penalty xG extraction could not be verified in this implementation environment. No total-xG or alternate-provider substitution is made. Supply `data/raw/understat_npxg.csv` with `kickoff_utc,home,away,npxg_home,npxg_away`; the daily workflow accepts a verified export URL through the `UNDERSTAT_NPXG_URL` Actions secret. Runs fail before publishing if the export is missing, malformed, or cannot join every historical match.
- The football-data.org fixture endpoint does not provide future closing odds. Market fields are emitted only when odds are actually present in the fixture input; no odds source is invented.

The sources' exact team names are normalized through `data/team_name_map.csv`. Unknown teams and unmatched xG records raise errors. Raw CSVs and xG exports belong in `data/raw/`; processed match data is parquet under `data/processed/`.

## Setup

Python 3.10 or newer is required.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Provide the verified Understat export at `data/raw/understat_npxg.csv` and set `FOOTBALL_DATA_API_KEY` before the production command. Network ingestion downloads football-data.co.uk history as needed.

```sh
python run_backtest.py
python run_daily.py
python -m pytest
```

`run_backtest.py` writes walk-forward predictions, model/market metrics, calibration bins, flat-stake edge tables with bootstrap intervals, and a reliability plot into `reports/`. Results are intentionally not claimed in this repository until real inputs have been ingested and this backtest has run.

## Model

The baseline is a Dixon–Coles weighted Poisson model. For each match, the two log-rates combine the league mean, home advantage, attacking strength, and opposing defensive strength. Attack and defence each use a sum-to-zero parameterization. The likelihood applies the standard low-score `rho` correction, with exponential recency weights `exp(-xi * days_ago)`; SciPy L-BFGS-B estimates the parameters. `xi` is selected on a chronological holdout.

The xG variant uses non-penalty xG and tests `alpha * npxG + (1 - alpha) * goals` on a chronological holdout. The goals-only model remains available for selection and benchmarking. Promotion priors average fitted attack/defence effects from completed first EPL seasons in the historical archive; incomplete seasons and data on/after the fit cutoff are excluded. Unknown/new promoted sides receive that prior, or neutral effects when the archive contains no completed promoted-team seasons.

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

1. Verify and own an automated Understat match-level non-penalty-xG exporter; the repository deliberately requires an explicit export until that extraction is verified.
2. Add a genuine historical matchweek identifier where available. football-data.co.uk's E0 files do not consistently expose one, so the evaluator uses documented seven-day within-season blocks as its fallback.
3. Run and publish a complete real-data backtest before treating model-vs-market metrics as an established result.
