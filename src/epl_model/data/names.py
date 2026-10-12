"""Strict team-name normalization backed by the checked-in source mapping."""

from pathlib import Path

import pandas as pd


DEFAULT_MAP = Path(__file__).resolve().parents[3] / "data" / "team_name_map.csv"


def normalize_team_names(values: pd.Series, source: str, mapping_path: Path = DEFAULT_MAP) -> pd.Series:
    """Map exact source spellings to canonical names; unknown names are an error."""
    if source not in {"football_data", "understat", "football_data_org", "fpl"}:
        raise ValueError(f"Unknown team-name source {source!r}")
    table = pd.read_csv(mapping_path, dtype=str).fillna("")
    lookup = dict(zip(table[source], table["canonical"], strict=True))
    canonical = set(table["canonical"])
    raw = values.astype(str).str.strip()
    unknown = sorted(set(raw) - set(lookup) - canonical)
    if unknown:
        raise AssertionError(f"Unmapped {source} team names: {unknown}")
    return raw.map(lambda value: lookup.get(value, value))


def normalize_match_teams(frame: pd.DataFrame, source: str, mapping_path: Path = DEFAULT_MAP) -> pd.DataFrame:
    """Normalize both sides of every match, asserting every team resolves."""
    result = frame.copy()
    result["home"] = normalize_team_names(result["home"], source, mapping_path)
    result["away"] = normalize_team_names(result["away"], source, mapping_path)
    if result[["home", "away"]].isna().any().any():
        raise AssertionError("At least one match did not resolve both teams")
    return result
