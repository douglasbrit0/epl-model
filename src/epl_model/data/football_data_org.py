"""Upcoming EPL fixtures from the football-data.org v4 API."""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API_URL = "https://api.football-data.org/v4/competitions/PL/matches"


def upcoming_fixtures(api_key: str | None = None, days: int = 10, now: datetime | None = None) -> pd.DataFrame:
    """Fetch scheduled EPL matches; FOOTBALL_DATA_API_KEY is required by the API."""
    token = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
    if not token:
        raise RuntimeError("FOOTBALL_DATA_API_KEY is required for football-data.org fixtures")
    current = now or datetime.now(timezone.utc)
    params = {"dateFrom": current.date().isoformat(), "dateTo": (current.date() + timedelta(days=days)).isoformat()}
    response = requests.get(API_URL, headers={"X-Auth-Token": token}, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise ValueError("football-data.org response does not contain a matches array")
    records = []
    for match in matches:
        records.append(
            {
                "fixture_id": f"fdo:{match['id']}",
                "kickoff_utc": pd.Timestamp(match["utcDate"]),
                "home": match["homeTeam"]["name"],
                "away": match["awayTeam"]["name"],
                "matchweek": match.get("matchday"),
            }
        )
    return pd.DataFrame(records, columns=["fixture_id", "kickoff_utc", "home", "away", "matchweek"])
