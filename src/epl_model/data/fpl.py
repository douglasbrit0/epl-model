"""Daily point-in-time player-availability snapshots from the Fantasy Premier League API.

Past injury reports are not available retrospectively, so the only honest availability history is
the one saved as it happens. Each run stores one compressed CSV of every player's status, chance of
playing and news, stamped with the UTC fetch time. The API is public and unauthenticated; one
request per day. FPL data is for personal, non-commercial use under the FPL terms.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from epl_model.data.names import normalize_team_names

LOGGER = logging.getLogger(__name__)
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PLAYER_FIELDS = [
    "id", "code", "web_name", "first_name", "second_name", "element_type", "team", "status",
    "chance_of_playing_this_round", "chance_of_playing_next_round", "news", "news_added",
    "minutes", "starts", "expected_goals", "expected_assists", "expected_goals_conceded", "now_cost",
]


def fetch_bootstrap(timeout: int = 30) -> dict:
    """Download the FPL bootstrap-static payload."""
    response = requests.get(BOOTSTRAP_URL, headers={"User-Agent": "epl-model availability snapshot"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("elements"), list) or not isinstance(payload.get("teams"), list):
        raise ValueError("FPL bootstrap-static response lacks elements/teams")
    return payload


def snapshot_frame(payload: dict, fetched_at: datetime) -> pd.DataFrame:
    """Flatten a bootstrap payload into one row per player, stamped with the fetch time."""
    players = pd.DataFrame(payload["elements"]).reindex(columns=PLAYER_FIELDS)
    teams = {team["id"]: team["name"] for team in payload["teams"]}
    next_events = [event["id"] for event in payload.get("events", []) if event.get("is_next")]
    frame = players.rename(columns={"id": "player_id", "team": "fpl_team_id"})
    frame.insert(0, "fetched_at", pd.Timestamp(fetched_at).tz_convert("UTC").isoformat().replace("+00:00", "Z"))
    frame.insert(1, "next_gameweek", next_events[0] if next_events else None)
    frame["team"] = normalize_team_names(frame["fpl_team_id"].map(teams), "fpl")
    frame["position"] = frame["element_type"].map(POSITIONS)
    for column in ("expected_goals", "expected_assists", "expected_goals_conceded"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def save_snapshot(frame: pd.DataFrame, directory: Path) -> Path:
    """Write a snapshot as ``fpl_<UTC timestamp>.csv.gz``; existing files are never overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = frame["fetched_at"].iloc[0].replace(":", "").replace("-", "")
    path = directory / f"fpl_{stamp}.csv.gz"
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, compression="gzip")
    return path


def take_snapshot(directory: Path, now: datetime | None = None) -> Path | None:
    """Fetch and store today's snapshot; failures are logged, never raised, so the pipeline continues."""
    try:
        frame = snapshot_frame(fetch_bootstrap(), now or datetime.now(timezone.utc))
        path = save_snapshot(frame, directory)
    except Exception as error:  # noqa: BLE001 - availability data must never block predictions
        LOGGER.warning("FPL availability snapshot failed: %s", error)
        return None
    LOGGER.info("Saved FPL availability snapshot %s (%d players)", path.name, len(frame))
    return path


def load_snapshots(directory: Path) -> pd.DataFrame:
    """Load every stored snapshot (empty frame if none)."""
    paths = sorted(directory.glob("fpl_*.csv.gz"))
    if not paths:
        return pd.DataFrame(columns=["fetched_at", *PLAYER_FIELDS])
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
