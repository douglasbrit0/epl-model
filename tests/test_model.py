import numpy as np

from epl_model.evaluation.metrics import outcome_probabilities, score_probabilities
from epl_model.models import DixonColesModel
from epl_model.pipeline import make_payloads
from epl_model.publish.contract import validate_payloads
from tests.test_temporal import sample_matches
from pathlib import Path


def fitted_model():
    model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,))
    return model.fit(sample_matches(), "2024-06-01T00:00:00Z")


def test_scoreline_matrix_is_normalized_and_nonnegative():
    matrix = fitted_model().scoreline_matrix("Arsenal", "Chelsea")
    assert matrix.shape == (11, 11)
    assert np.all(matrix >= 0)
    assert np.isclose(matrix.sum(), 1.0)


def test_derived_markets_are_coherent():
    matrix = fitted_model().scoreline_matrix("Arsenal", "Chelsea")
    outcomes = outcome_probabilities(matrix)
    markets = score_probabilities(matrix)
    assert np.isclose(outcomes.sum(), 1.0)
    assert np.isclose(markets["p_home"] + markets["p_draw"] + markets["p_away"], 1.0)
    assert 0 <= markets["p_over_2_5"] <= 1
    assert 0 <= markets["p_btts"] <= 1


def test_pipeline_payloads_match_dougboard_schemas():
    matches = sample_matches()
    model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,)).fit(matches, "2024-06-01T00:00:00Z")
    fixture = {
        "fixture_id": "fdo:123", "kickoff_utc": "2024-06-10T15:00:00Z",
        "home": "Arsenal", "away": "Chelsea",
    }
    payloads, records = make_payloads(model, matches, __import__("pandas").DataFrame([fixture]), __import__("pandas").DataFrame([fixture]))

    validate_payloads(payloads, Path(__file__).resolve().parents[1] / "schemas")

    assert len(payloads["fixtures.json"][0]["top_scorelines"]) == 5
    assert len(payloads["fixtures.json"][0]["scoreline_matrix"]) == 7
    assert records[0]["fixture_id"] == "fdo:123"
