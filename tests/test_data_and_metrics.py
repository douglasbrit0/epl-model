import numpy as np
import pandas as pd
import pytest

from epl_model.data.football_data import _closing_odds
from epl_model.data.cleaning import join_xg
from epl_model.data.names import normalize_team_names
from epl_model.evaluation.metrics import closing_odds_probabilities, multiclass_scores


def test_pinnacle_closing_odds_preferred_with_average_fallback():
    source = pd.DataFrame({
        "PSCH": [2.0, np.nan], "PSCD": [3.4, np.nan], "PSCA": [4.0, np.nan],
        "AvgCH": [2.1, 2.5], "AvgCD": [3.5, 3.2], "AvgCA": [3.9, 3.0],
    })
    result = _closing_odds(source)
    assert result.loc[0, "odds_home"] == 2.0
    assert result.loc[0, "odds_source"] == "pinnacle"
    assert result.loc[1, "odds_home"] == 2.5
    assert result.loc[1, "odds_source"] == "market_average"


def test_exact_team_map_rejects_unknown_names():
    assert normalize_team_names(pd.Series(["Man City"]), "football_data").iloc[0] == "Manchester City"
    with pytest.raises(AssertionError, match="Unmapped"):
        normalize_team_names(pd.Series(["Unknown FC"]), "football_data")


def test_xg_join_accepts_pre_normalized_canonical_names():
    kickoff = pd.Timestamp("2024-01-01", tz="UTC")
    results = pd.DataFrame([{"fixture_id": "x", "kickoff_utc": kickoff, "home": "Manchester City", "away": "Arsenal", "home_goals": 2, "away_goals": 0}])
    xg = pd.DataFrame([{"kickoff_utc": kickoff, "home": "Manchester City", "away": "Arsenal", "npxg_home": 1.5, "npxg_away": 0.3}])
    joined = join_xg(results, xg)
    assert joined.loc[0, "npxg_home"] == 1.5


@pytest.mark.parametrize("method", ["power", "shin"])
def test_odds_devig_returns_distribution(method):
    probabilities = closing_odds_probabilities((2.1, 3.4, 3.6), method)
    assert np.all(probabilities > 0)
    assert np.isclose(probabilities.sum(), 1)


def test_multiclass_metrics_are_finite():
    result = multiclass_scores(np.array([[0.6, 0.2, 0.2], [0.2, 0.5, 0.3]]), np.array([0, 1]))
    assert all(np.isfinite(value) and value >= 0 for value in result.values())
