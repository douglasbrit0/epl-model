import json

import numpy as np
import pandas as pd

from epl_model.data.football_data_org import unplayed, upcoming_fixtures
from epl_model.evaluation.track_record import score_prediction_log
from epl_model.models.dixon_coles import ScorelineForecast
from epl_model.simulation.season import simulate_season


def test_track_record_scores_only_the_last_pre_kickoff_prediction(tmp_path):
    log, scored = tmp_path / "log.jsonl", tmp_path / "scored.jsonl"
    base = {"fixture_id": "fdo:1", "model_version": "v1", "kickoff_utc": "2026-10-10T15:00:00Z", "home": "Arsenal", "away": "Chelsea"}
    rows = [
        {**base, "prediction_id": "early", "generated_at": "2026-10-08T06:00:00Z", "p_home": 0.2, "p_draw": 0.3, "p_away": 0.5},
        {**base, "prediction_id": "last", "generated_at": "2026-10-10T06:00:00Z", "p_home": 0.6, "p_draw": 0.2, "p_away": 0.2},
        {**base, "prediction_id": "after", "generated_at": "2026-10-11T06:00:00Z", "p_home": 0.9, "p_draw": 0.05, "p_away": 0.05},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = pd.DataFrame([{"kickoff_utc": "2026-10-10T00:00:00Z", "home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 0}])
    out = score_prediction_log(log, results, scored)
    assert out["prediction_id"].tolist() == ["last"]


def test_unplayed_keeps_postponed_and_lagging_results_but_drops_played():
    results = pd.DataFrame([
        {"kickoff_utc": "2026-08-16", "home": "Arsenal", "away": "Chelsea", "home_goals": 1, "away_goals": 0},
        {"kickoff_utc": "2025-12-01", "home": "Arsenal", "away": "Fulham", "home_goals": 1, "away_goals": 0},  # previous season
    ])
    fixtures = pd.DataFrame([
        {"kickoff_utc": "2026-08-17T15:00:00Z", "home": "Arsenal", "away": "Chelsea", "status": "FINISHED"},  # rescheduled date, already in history
        {"kickoff_utc": "2026-09-20T15:00:00Z", "home": "Arsenal", "away": "Fulham", "status": "FINISHED"},  # result not in history yet
        {"kickoff_utc": "2026-09-01T15:00:00Z", "home": "Chelsea", "away": "Arsenal", "status": "POSTPONED"},
    ])
    remaining = unplayed(fixtures, results, 2026)
    assert set(zip(remaining.home, remaining.away)) == {("Arsenal", "Fulham"), ("Chelsea", "Arsenal")}


def test_upcoming_fixtures_drop_finished_and_postponed(monkeypatch):
    payload = {"matches": [
        {"id": i, "utcDate": "2026-10-17T14:00:00Z", "homeTeam": {"name": "Arsenal FC"}, "awayTeam": {"name": "Chelsea FC"}, "matchday": 8, "status": status}
        for i, status in enumerate(["TIMED", "FINISHED", "POSTPONED", "SCHEDULED"])
    ]}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("epl_model.data.football_data_org.requests.get", lambda *args, **kwargs: Response())
    fixtures = upcoming_fixtures(api_key="x", now=pd.Timestamp("2026-10-12", tz="UTC").to_pydatetime())
    assert fixtures["status"].tolist() == ["TIMED", "SCHEDULED"]


def test_simulation_applies_points_then_goal_difference_tiebreak():
    certain = np.zeros((4, 4)); certain[3, 0] = 1.0  # home always wins 3-0
    forecasts = [ScorelineForecast("1", "B", "C", certain, 3, 0)]
    table = pd.DataFrame([
        {"team": "A", "points": 3, "gf": 1, "ga": 0},
        {"team": "B", "points": 0, "gf": 0, "ga": 0},
        {"team": "C", "points": 0, "gf": 0, "ga": 0},
    ])
    result = simulate_season(forecasts, ["A", "B", "C"], table, simulations=50, relegation_places=1).set_index("team")
    assert result.loc["B", "p_title"] == 1.0  # B and A tie on points; B has the better goal difference
    assert result.loc["C", "p_relegation"] == 1.0
    assert result.loc["A", "exp_points"] == 3
