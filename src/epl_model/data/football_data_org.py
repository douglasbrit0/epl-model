"""Upcoming EPL fixtures from the football-data.org v4 API."""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
COLUMNS = ["fixture_id", "kickoff_utc", "home", "away", "matchweek", "status"]


def _fetch(params: dict, api_key: str | None) -> pd.DataFrame:
    token = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
    if not token:
        raise RuntimeError("FOOTBALL_DATA_API_KEY is required for football-data.org fixtures")
    response = requests.get(API_URL, headers={"X-Auth-Token": token}, params=params, timeout=30)
    response.raise_for_status()
    matches = response.json().get("matches")
    if not isinstance(matches, list):
        raise ValueError("football-data.org response does not contain a matches array")
    records = [
        {
            "fixture_id": f"fdo:{match['id']}",
            "kickoff_utc": pd.Timestamp(match["utcDate"]),
            "home": match["homeTeam"]["name"],
            "away": match["awayTeam"]["name"],
            "matchweek": match.get("matchday"),
            "status": match.get("status", "SCHEDULED"),
        }
        for match in matches
    ]
    return pd.DataFrame(records, columns=COLUMNS)


def upcoming_fixtures(api_key: str | None = None, days: int = 10, now: datetime | None = None) -> pd.DataFrame:
    """Fetch EPL matches still to kick off in the next ``days``; FOOTBALL_DATA_API_KEY is required."""
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    params = {"dateFrom": current.date().isoformat(), "dateTo": (current.date() + timedelta(days=days)).isoformat()}
    fixtures = _fetch(params, api_key)
    keep = fixtures["status"].isin({"SCHEDULED", "TIMED"}) & (pd.to_datetime(fixtures["kickoff_utc"], utc=True) >= current)
    return fixtures.loc[keep].reset_index(drop=True)


def season_fixtures(season_start_year: int, api_key: str | None = None) -> pd.DataFrame:
    """Fetch every non-cancelled EPL match of a season, including played and postponed ones."""
    fixtures = _fetch({"season": season_start_year}, api_key)
    return fixtures.loc[fixtures["status"] != "CANCELLED"].reset_index(drop=True)


def unplayed(fixtures: pd.DataFrame, results: pd.DataFrame, season_start_year: int) -> pd.DataFrame:
    """Season fixtures whose home/away pairing has no result in that season's history yet.

    Each pairing occurs once per season, so this is robust to rescheduled dates. Results from
    football-data.co.uk can lag a few days; finished matches missing from the history therefore
    stay in the simulation instead of vanishing from both the table and the remainder.
    """
    from epl_model.seasons import season_labels

    season = results.loc[season_labels(results["kickoff_utc"]) == season_start_year]
    played = set(zip(season["home"], season["away"]))
    keep = [(home, away) not in played for home, away in zip(fixtures["home"], fixtures["away"])]
    return fixtures.loc[keep].reset_index(drop=True)
