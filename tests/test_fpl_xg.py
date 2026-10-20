import numpy as np
import pandas as pd

from epl_model.data.fpl_xg import attach_team_xg, live_team_xg
from epl_model.models import DixonColesModel


def test_live_team_xg_sums_players_per_side_and_skips_double_gameweeks():
    bootstrap = {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Spurs"}, {"id": 3, "name": "Man City"}],
                 "elements": [{"id": 10, "team": 1}, {"id": 11, "team": 1}, {"id": 20, "team": 2}, {"id": 30, "team": 3}]}
    fixtures = [
        {"id": 100, "event": 1, "finished": True, "kickoff_time": "2026-08-22T14:00:00Z", "team_h": 1, "team_a": 2},
        {"id": 200, "event": 2, "finished": True, "kickoff_time": "2026-08-29T14:00:00Z", "team_h": 3, "team_a": 2},
        {"id": 201, "event": 2, "finished": True, "kickoff_time": "2026-08-31T19:00:00Z", "team_h": 2, "team_a": 1},
    ]
    lives = {
        1: {"elements": [
            {"id": 10, "stats": {"expected_goals": "0.80"}, "explain": [{"fixture": 100}]},
            {"id": 11, "stats": {"expected_goals": "0.45"}, "explain": [{"fixture": 100}]},
            {"id": 20, "stats": {"expected_goals": "0.30"}, "explain": [{"fixture": 100}]},
        ]},
        2: {"elements": [  # Spurs play twice: their player's xG cannot be split between fixtures
            {"id": 20, "stats": {"expected_goals": "1.10"}, "explain": [{"fixture": 200}, {"fixture": 201}]},
            {"id": 30, "stats": {"expected_goals": "0.90"}, "explain": [{"fixture": 200}]},
        ]},
    }
    frame = live_team_xg(bootstrap, fixtures, lives)
    assert frame[["home", "away", "fpl_xg_home", "fpl_xg_away"]].values.tolist() == [["Arsenal", "Tottenham Hotspur", 1.25, 0.3]]


def test_xg_blend_uses_goals_where_xg_is_missing_and_keeps_scoring_level():
    data = pd.DataFrame({"home_goals": [2, 1, 3, 0], "away_goals": [1, 1, 0, 2],
                         "fpl_xg_home": [1.0, 0.5, np.nan, np.nan], "fpl_xg_away": [0.5, 1.0, np.nan, np.nan]})
    model = DixonColesModel(xg_columns=("fpl_xg_home", "fpl_xg_away"))
    home, away = model._targets(data, 0.5)
    assert home[2:].tolist() == [3, 0] and away[2:].tolist() == [0, 2]  # no xG: goals unchanged
    assert np.isclose(home[:2].sum() + away[:2].sum(), 5)  # rescaled xG keeps the goal total


def test_attach_team_xg_matches_on_uk_date_and_teams():
    matches = pd.DataFrame({"kickoff_utc": pd.to_datetime(["2026-08-22", "2026-08-23"], utc=True), "home": ["Arsenal", "Chelsea"], "away": ["Fulham", "Everton"]})
    xg = pd.DataFrame({"kickoff_utc": pd.to_datetime(["2026-08-22T19:30:00Z"], utc=True), "home": ["Arsenal"], "away": ["Fulham"], "fpl_xg_home": [1.9], "fpl_xg_away": [0.4]})
    out = attach_team_xg(matches, xg)
    assert out["fpl_xg_home"].tolist()[0] == 1.9 and np.isnan(out["fpl_xg_home"].tolist()[1])
