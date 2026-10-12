"""Canonical joins between results and xG records."""

import pandas as pd

from epl_model.data.names import normalize_match_teams


def join_xg(results: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """Join normalized team xG by exact UTC date and teams; unmatched games fail."""
    result = normalize_match_teams(results, "football_data")
    expected = normalize_match_teams(xg, "understat")
    for frame in (result, expected):
        frame["match_date"] = pd.to_datetime(frame["kickoff_utc"], utc=True).dt.date
    key = ["match_date", "home", "away"]
    joined = result.merge(
        expected[key + ["npxg_home", "npxg_away"]],
        how="left",
        on=key,
        validate="one_to_one",
        indicator=True,
    )
    missing = joined.loc[joined["_merge"] != "both", key]
    if not missing.empty:
        sample = missing.head(5).to_dict("records")
        raise AssertionError(f"Results without exact Understat xG match: {len(missing)}; examples={sample}")
    return joined.drop(columns=["_merge", "match_date"])
