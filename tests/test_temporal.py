from datetime import date, timedelta

import pandas as pd

from epl_model.models import DixonColesModel


def sample_matches():
    teams = ["Arsenal", "Chelsea", "Liverpool", "Spurs"]
    start = date(2024, 1, 1)
    rows = []
    for index in range(48):
        home = teams[index % len(teams)]
        away = teams[(index + 1) % len(teams)]
        rows.append(
            {
                "kickoff_utc": start + timedelta(days=index * 3),
                "home": home,
                "away": away,
                "home_goals": index % 4,
                "away_goals": (index // 2) % 3,
            }
        )
    return pd.DataFrame(rows)


def test_fit_excludes_matches_on_or_after_as_of_date():
    matches = sample_matches()
    cutoff = pd.Timestamp("2024-04-20", tz="UTC")
    model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,))

    model.fit(matches, cutoff)

    assert model.training_max_date < cutoff
    assert model.training_rows == int(
        (pd.to_datetime(matches["kickoff_utc"], utc=True) < cutoff).sum()
    )


def test_same_day_result_cannot_change_fit():
    matches = sample_matches()
    cutoff = pd.Timestamp("2024-04-20", tz="UTC")
    model = DixonColesModel(xi_candidates=(0.0,), alpha_candidates=(0.0,))
    model.fit(matches, cutoff)
    original = (model.mu, model.home_advantage, model.training_rows)

    late_match = matches.iloc[[0]].copy()
    late_match.loc[:, "kickoff_utc"] = cutoff
    late_match.loc[:, "home_goals"] = 99
    model.fit(pd.concat([matches, late_match], ignore_index=True), cutoff)

    assert (model.mu, model.home_advantage, model.training_rows) == original


def test_walk_forward_holds_out_complete_week_before_prediction():
    import numpy as np

    from epl_model.evaluation.backtest import walk_forward

    class StubModel:
        def fit(self, matches, as_of_date):
            self.training_max_date = pd.to_datetime(matches["kickoff_utc"], utc=True).max()
            assert self.training_max_date < as_of_date
            return self

        def predict(self, fixtures):
            forecast = {"matrix": np.array([[0.2, 0.2], [0.3, 0.3]]), "xg_home": 0.6, "xg_away": 0.5}
            return [type("Forecast", (), forecast)() for _ in range(len(fixtures))]

    matches = sample_matches()
    matches["fixture_id"] = [f"fixture-{index}" for index in matches.index]
    predictions = walk_forward(matches, model_factory=StubModel)
    assert not predictions.empty
    for _, group in predictions.groupby(["season", "matchweek"]):
        assert group["fit_data_through"].nunique() == 1
        cutoff = pd.to_datetime(group["kickoff_utc"], utc=True).min().normalize()
        assert pd.to_datetime(group["fit_data_through"], utc=True).max() < cutoff


def test_nested_tuning_chooses_each_season_from_earlier_seasons_only():
    from epl_model.evaluation.tuning import DEFAULT_CONFIG, nested_select

    rows = []
    for season in (2018, 2019, 2020):
        for xi in (0.001, 0.002):
            # xi=0.001 is perfect in 2018 and useless afterwards; xi=0.002 the reverse.
            good = (xi == 0.001) == (season == 2018)
            rows.append({"season": season, "matchweek": 1, "xi": xi, "prior_scale": 1.0, "outcome": 0,
                         "p_home": 0.9 if good else 0.1, "p_draw": 0.05 if good else 0.45, "p_away": 0.05 if good else 0.45})
    _, chosen, production = nested_select(pd.DataFrame(rows))
    assert chosen[2018] == DEFAULT_CONFIG  # nothing earlier to tune on
    assert chosen[2019]["xi"] == 0.001  # only 2018 is visible; the 2019 result must not leak in
    assert chosen[2020]["xi"] in (0.001, 0.002)
    assert production["xi"] == 0.002
