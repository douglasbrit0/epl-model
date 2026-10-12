import json
from datetime import datetime, timezone

import pytest

from epl_model.publish.contract import append_prediction_log, publish_snapshot, validate_payloads


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def valid_payloads():
    generated = datetime.now(timezone.utc).isoformat()
    rating = {"team": "Arsenal", "attack": 0.1, "defence": 0.2, "overall": 0.3}
    season_row = {"team": "Arsenal", "p_title": 0.2, "p_top4": 0.7, "p_top6": 0.9, "p_relegation": 0.01, "exp_points": 75}
    return {
        "latest.json": {"generated_at": generated, "model_version": "test", "data_through": "2026-10-11"},
        "fixtures.json": [],
        "ratings.json": {"generated_at": generated, "model_version": "test", "teams": [rating], "history": [{"date": "2026-10-12", **rating}]},
        "season.json": {"generated_at": generated, "model_version": "test", "teams": [season_row], "history": []},
        "performance.json": {"generated_at": generated, "model_version": "test", "rolling": [], "calibration_bins": [], "cumulative_flat_stake_pnl": 0},
    }


def test_contract_rejects_invalid_payloads():
    payloads = valid_payloads()
    payloads["latest.json"]["data_through"] = "not-a-date"
    with pytest.raises(ValueError, match="latest.json"):
        validate_payloads(payloads, ROOT / "schemas")


def test_failed_validation_keeps_previous_snapshot(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "latest.json"
    marker.write_text("previous snapshot")
    payloads = valid_payloads()
    payloads["season.json"]["teams"][0]["p_title"] = 2
    with pytest.raises(ValueError):
        publish_snapshot(payloads, output, ROOT / "schemas", [])
    assert marker.read_text() == "previous snapshot"


def test_prediction_log_is_append_only_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    record = {"prediction_id": "abc", "fixture_id": "fdo:1", "model_version": "v1", "generated_at": "2026-10-12T06:00:00Z"}
    append_prediction_log(path, [record])
    with pytest.raises(ValueError, match="duplicate"):
        append_prediction_log(path, [record])
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 1


def test_daily_performance_reports_power_and_shin_market_scores(tmp_path):
    from epl_model.evaluation.track_record import score_prediction_log

    log = tmp_path / "prediction_log.jsonl"
    scored = tmp_path / "scored_predictions.jsonl"
    record = {
        "prediction_id": "p1", "fixture_id": "fdo:42", "model_version": "v1",
        "generated_at": "2026-10-12T06:00:00Z", "kickoff_utc": "2026-10-10T15:00:00Z",
        "home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
    }
    log.write_text(json.dumps(record) + "\n")
    results = __import__("pandas").DataFrame([{
        "fixture_id": "fd:2026-10-10:Arsenal:Chelsea", "kickoff_utc": "2026-10-10T00:00:00Z",
        "home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 1,
        "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 4.0,
    }])

    scored_rows = score_prediction_log(log, results, scored)

    assert len(scored_rows) == 1
    assert "market_log_loss" in scored_rows
    assert "market_shin_rps" in scored_rows
    assert "flat_stake_profit" in scored_rows
