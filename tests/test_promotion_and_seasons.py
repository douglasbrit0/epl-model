import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from epl_model.data.names import normalize_team_names
from epl_model.models import DixonColesModel
from epl_model.seasons import season_labels, season_start_year


def league(seasons=((2021, "ABCDEF"), (2022, "ABCDEG"), (2023, "ABCDEH")), promoted_wins=None):
    """Double round-robin per season; a newcomer arrives each season. Results are deterministic."""
    rng = np.random.default_rng(0)
    rows = []
    for year, teams in seasons:
        day = pd.Timestamp(f"{year}-08-10", tz="UTC")
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                if promoted_wins and promoted_wins in (home, away):
                    hg, ag = (3, 0) if home == promoted_wins else (0, 3)
                rows.append({"kickoff_utc": day, "home": home, "away": away, "home_goals": hg, "away_goals": ag})
                day += pd.Timedelta(days=4)
    return pd.DataFrame(rows)


class SmallLeagueModel(DixonColesModel):
    min_season_matches = 20


def model():
    return SmallLeagueModel(xi_candidates=(0.0,), alpha_candidates=(0.0,))


def test_2020_restart_matches_belong_to_2019_20():
    assert season_start_year("2020-07-26") == 2019
    assert season_start_year("2020-09-12") == 2020
    assert season_start_year("2024-05-19") == 2023
    labels = season_labels(pd.Series(pd.to_datetime(["2020-03-07", "2020-07-01", "2020-09-12"], utc=True)))
    assert labels.tolist() == [2019, 2019, 2020]


def test_unseen_promoted_team_takes_prior_not_league_average():
    data = league()
    fitted = model().fit(data, "2023-08-01")  # H has never played
    assert "H" not in fitted.attack
    assert fitted.prior_strength > 0
    assert fitted._unseen[0] + fitted._unseen[1] < 0  # newcomers are weaker than established teams
    expected_away_rate = np.exp(fitted.mu + fitted._unseen[0] - fitted.defence["A"])
    matrix = fitted.scoreline_matrix("A", "H", max_goals=30)
    assert (matrix.sum(axis=0) * np.arange(31)).sum() == pytest.approx(expected_away_rate, rel=0.02)


def test_promoted_team_rating_moves_with_its_results():
    data = league(promoted_wins="H")
    cutoff = data.loc[data.kickoff_utc >= "2023-07-01", "kickoff_utc"].iloc[20]
    fitted = model().fit(data, cutoff)
    assert "H" in fitted.promoted_teams
    assert fitted.attack["H"] + fitted.defence["H"] > sum(fitted._unseen)  # dominant results beat the prior


def test_promoted_prior_uses_only_data_before_cutoff():
    data = league()
    cutoff = pd.Timestamp("2023-09-01", tz="UTC")
    base = model().fit(data, cutoff)
    future = league(seasons=((2024, "ABCDEZ"),), promoted_wins="Z")
    with_future = model().fit(pd.concat([data, future], ignore_index=True), cutoff)
    assert with_future.prior_delta == base.prior_delta
    assert with_future.prior_strength == base.prior_strength
    assert with_future.attack == base.attack


def test_gradient_matches_finite_differences_with_promotion_penalty():
    data = league()
    fitted = model()
    training = fitted._strict_training_data(data, "2023-10-01")
    fitted._derive_promoted_priors(training, 2023)
    fitted.teams = sorted(set(training.home) | set(training.away))
    design = fitted._design(training, 0.002, 0.0, fitted._promotion_penalty(training, 2023))
    params = np.random.default_rng(3).normal(0, 0.1, 2 + 2 * (len(fitted.teams) - 1) + 1)
    numeric = approx_fprime(params, lambda x: fitted._nll_and_grad(x, design)[0], 1e-7)
    assert np.allclose(numeric, fitted._nll_and_grad(params, design)[1], atol=1e-5)


def test_xg_blend_is_selected_on_actual_goals():
    data = league()
    rng = np.random.default_rng(1)
    strength = {"A": 2.4, "B": 1.8, "C": 1.4, "D": 1.1, "E": 0.8, "F": 0.6, "G": 0.6, "H": 0.6}
    data["home_goals"] = rng.poisson([strength[h] * 1.2 / strength[a] ** 0.5 for h, a in zip(data.home, data.away)])
    data["away_goals"] = rng.poisson([strength[a] / strength[h] ** 0.5 for h, a in zip(data.home, data.away)])
    data["npxg_home"] = rng.gamma(4, 1.4 / 4, len(data))  # pure noise: no team information
    data["npxg_away"] = rng.gamma(4, 1.1 / 4, len(data))
    fitted = SmallLeagueModel(xi_candidates=(0.0,), alpha_candidates=(0.0, 1.0)).fit(data, "2023-12-01")
    assert fitted.alpha == 0.0


def test_football_data_spellings_for_current_teams_are_mapped():
    mapped = normalize_team_names(pd.Series(["Nott'm Forest", "Hull", "Coventry"]), "football_data")
    assert mapped.tolist() == ["Nottingham Forest", "Hull City", "Coventry City"]
    assert normalize_team_names(pd.Series(["Hull City AFC", "Coventry City FC"]), "football_data_org").tolist() == ["Hull City", "Coventry City"]


def test_returning_team_starts_from_prior_not_its_earlier_spell():
    # F dominates 2021, is relegated for 2022, and returns in 2023.
    data = league(seasons=((2021, "ABCDEF"), (2022, "ABCDEG")), promoted_wins="F")
    fitted = model().fit(data, "2023-08-01")
    assert "F" not in fitted.attack
    assert "F (earlier spell)" in fitted.attack
    assert fitted.scoreline_matrix("F", "A").tolist() == fitted.scoreline_matrix("H", "A").tolist()
    legacy = SmallLeagueModel(xi_candidates=(0.0,), alpha_candidates=(0.0,), split_spells=False).fit(data, "2023-08-01")
    assert legacy.attack["F"] > fitted._unseen[0]


def test_dynamic_model_gradient_and_temporal_cutoff():
    from epl_model.models.dynamic import DynamicDixonColesModel

    class SmallDynamic(DynamicDixonColesModel):
        min_season_matches = 20

    data = league()
    captured = {}
    fitted = SmallDynamic(block_days=60)
    original = fitted._objective

    def spy(params, design, layout):
        captured["args"] = (design, layout)
        return original(params, design, layout)

    fitted._objective = spy
    fitted.fit(data, "2023-10-01")
    assert fitted.training_max_date < pd.Timestamp("2023-10-01", tz="UTC")
    design, layout = captured["args"]
    size = 2 * layout["blocks"] + 2 * layout["blocks"] * layout["count"] + 1
    params = np.random.default_rng(5).normal(0, 0.1, size)
    numeric = approx_fprime(params, lambda x: original(x, design, layout)[0], 1e-7)
    assert np.allclose(numeric, original(params, design, layout)[1], atol=1e-5)
