"""Run the daily ingest, refit, simulate, schema-check, and publish pipeline."""

from pathlib import Path
import json

import pandas as pd

from epl_model.data.cleaning import attach_optional_xg
from epl_model.data.football_data import download_history, load_history
from epl_model.data.fpl import take_snapshot
from epl_model.data.football_data_org import season_fixtures, unplayed, upcoming_fixtures
from epl_model.data.names import normalize_match_teams
from epl_model.data.storage import save_matches
from epl_model.evaluation.track_record import score_prediction_log
from epl_model.pipeline import choose_validated_model, make_payloads
from epl_model.publish.contract import publish_snapshot
from epl_model.seasons import season_start_year

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run daily production steps; failure before publish leaves prior output intact."""
    # Snapshot player availability first: it builds point-in-time history and never blocks the run.
    take_snapshot(ROOT / "logs" / "fpl_snapshots")
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    history_paths = download_history(raw)
    results = normalize_match_teams(load_history(history_paths), "football_data")
    matches = attach_optional_xg(results, raw / "understat_npxg.csv")
    save_matches(matches, processed / "matches.parquet")
    now = pd.Timestamp.now(tz="UTC")
    ten_day_fixtures = upcoming_fixtures(days=10, now=now.to_pydatetime())
    ten_day_fixtures = normalize_match_teams(ten_day_fixtures, "football_data_org")
    full_season_fixtures = normalize_match_teams(season_fixtures(season_start_year(now)), "football_data_org")
    full_season_fixtures = unplayed(full_season_fixtures, matches, season_start_year(now))
    ten_day_fixtures["kickoff_utc"] = pd.to_datetime(ten_day_fixtures["kickoff_utc"], utc=True)
    full_season_fixtures["kickoff_utc"] = pd.to_datetime(full_season_fixtures["kickoff_utc"], utc=True)
    model = choose_validated_model(matches)
    log_path = ROOT / "logs" / "prediction_log.jsonl"
    if log_path.exists():
        score_prediction_log(log_path, matches, ROOT / "logs" / "scored_predictions.jsonl")
    previous_payloads = {}
    for filename in ("ratings.json", "season.json"):
        previous_file = ROOT / "output" / filename
        if previous_file.exists():
            previous_payloads[filename] = json.loads(previous_file.read_text(encoding="utf-8"))
    payloads, log_records = make_payloads(model, matches, ten_day_fixtures, full_season_fixtures, previous_payloads=previous_payloads)
    publish_snapshot(payloads, ROOT / "output", ROOT / "schemas", log_records)
    print(f"Published {len(payloads['fixtures.json'])} fixtures with {payloads['latest.json']['model_version']}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
