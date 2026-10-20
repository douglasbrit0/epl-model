"""Team expected goals per match from FPL player data (Opta xG, available from 2022/23).

FPL publishes each player's ``expected_goals`` per fixture. Summing a team's players gives that
team's xG for the match (penalties included: FPL has no non-penalty split). Historical seasons come
from the vaastav/Fantasy-Premier-League archive (MIT-licensed code; data from the FPL API under FPL's
terms of use), saved as ``data/raw/fpl_archive/<season>_merged_gw.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from epl_model.data.names import normalize_team_names

LOGGER = logging.getLogger(__name__)
ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
COLUMNS = ["kickoff_utc", "home", "away", "fpl_xg_home", "fpl_xg_away"]


def team_xg_from_archive(paths: list[Path]) -> pd.DataFrame:
    """One row per match with both teams' summed player xG; seasons without xG are skipped."""
    frames = []
    for path in sorted(paths):
        players = pd.read_csv(path, low_memory=False)
        if not {"expected_goals", "team", "fixture", "was_home", "kickoff_time"}.issubset(players.columns):
            continue
        players = players.assign(
            expected_goals=pd.to_numeric(players["expected_goals"], errors="coerce").fillna(0.0),
            team=normalize_team_names(players["team"], "fpl"),
            was_home=players["was_home"].astype(str).str.lower().eq("true"),
        )
        by_side = players.groupby(["fixture", "was_home"]).agg(team=("team", "first"), xg=("expected_goals", "sum"), kickoff=("kickoff_time", "first"))
        home = by_side.xs(True, level="was_home")
        away = by_side.xs(False, level="was_home")
        both = home.join(away, lsuffix="_home", rsuffix="_away", how="inner")
        frames.append(pd.DataFrame({
            "kickoff_utc": pd.to_datetime(both["kickoff_home"], utc=True),
            "home": both["team_home"],
            "away": both["team_away"],
            "fpl_xg_home": both["xg_home"].round(3),
            "fpl_xg_away": both["xg_away"].round(3),
        }))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def attach_team_xg(matches: pd.DataFrame, team_xg: pd.DataFrame) -> pd.DataFrame:
    """Add fpl_xg_home/away to matches on UK match date and teams (NaN where unavailable)."""
    def key(frame: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(frame["kickoff_utc"], utc=True).dt.tz_convert("Europe/London").dt.date.astype(str)
        return dates + "|" + frame["home"] + "|" + frame["away"]

    lookup = team_xg.assign(_key=key(team_xg)).drop_duplicates("_key").set_index("_key")[["fpl_xg_home", "fpl_xg_away"]]
    out = matches.drop(columns=["fpl_xg_home", "fpl_xg_away"], errors="ignore").copy()
    keys = key(out)
    out["fpl_xg_home"] = keys.map(lookup["fpl_xg_home"]).to_numpy()
    out["fpl_xg_away"] = keys.map(lookup["fpl_xg_away"]).to_numpy()
    return out


FPL_API = "https://fantasy.premierleague.com/api"


def live_team_xg(bootstrap: dict, fixtures: list, lives: dict[int, dict]) -> pd.DataFrame:
    """Current-season team xG from FPL's live gameweek data.

    Each player's gameweek xG is credited to the fixture listed in its ``explain`` block, on the side
    of the player's FPL team. Players in two fixtures that gameweek, or whose current team played in
    neither side of the fixture (transferred since), are skipped. Matches where any player was skipped
    for a double fixture get no xG and fall back to goals in the model.
    """
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    player_team = {player["id"]: player["team"] for player in bootstrap["elements"]}
    by_id = {fixture["id"]: fixture for fixture in fixtures if fixture.get("finished")}
    totals: dict[int, dict[int, float]] = {}
    ambiguous: set[int] = set()
    for live in lives.values():
        for element in live.get("elements", []):
            fixture_ids = [entry["fixture"] for entry in element.get("explain", [])]
            xg = float(element.get("stats", {}).get("expected_goals") or 0.0)
            if len(fixture_ids) > 1:
                if xg > 0:
                    ambiguous.update(fixture_ids)
                continue
            if not fixture_ids or fixture_ids[0] not in by_id:
                continue
            fixture, team = by_id[fixture_ids[0]], player_team.get(element["id"])
            if team in (fixture["team_h"], fixture["team_a"]):
                totals.setdefault(fixture["id"], {}).setdefault(team, 0.0)
                totals[fixture["id"]][team] += xg
    rows = []
    for fixture_id, fixture in by_id.items():
        if fixture_id in ambiguous:
            continue
        sums = totals.get(fixture_id, {})
        rows.append({
            "kickoff_utc": pd.Timestamp(fixture["kickoff_time"]),
            "home": teams[fixture["team_h"]],
            "away": teams[fixture["team_a"]],
            "fpl_xg_home": round(sums.get(fixture["team_h"], 0.0), 3),
            "fpl_xg_away": round(sums.get(fixture["team_a"], 0.0), 3),
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame["home"] = normalize_team_names(frame["home"], "fpl")
        frame["away"] = normalize_team_names(frame["away"], "fpl")
    return frame


def fetch_live_team_xg(timeout: int = 30) -> pd.DataFrame:
    """Download the current season's finished-match team xG (one request per finished gameweek)."""
    import requests

    headers = {"User-Agent": "epl-model team xG"}
    get = lambda path: requests.get(f"{FPL_API}/{path}", headers=headers, timeout=timeout)  # noqa: E731
    responses = {"bootstrap": get("bootstrap-static/"), "fixtures": get("fixtures/")}
    for response in responses.values():
        response.raise_for_status()
    fixtures = responses["fixtures"].json()
    events = sorted({fixture["event"] for fixture in fixtures if fixture.get("finished") and fixture.get("event")})
    lives = {}
    for event in events:
        response = get(f"event/{event}/live/")
        response.raise_for_status()
        lives[event] = response.json()
    return live_team_xg(responses["bootstrap"].json(), fixtures, lives)


def load_team_xg(archive_dir: Path, current_season: int, first_season: int = 2022) -> pd.DataFrame:
    """Archive xG for completed seasons (downloaded once) plus live FPL xG for the current season.

    Any download problem logs a warning and returns whatever is available; matches without xG are
    trained on goals alone.
    """
    import requests

    archive_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(first_season, current_season):
        name = f"{start}-{(start + 1) % 100:02d}"
        path = archive_dir / f"{name}_merged_gw.csv"
        if not path.exists():
            try:
                response = requests.get(ARCHIVE_URL.format(season=name), timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("Could not download FPL archive %s: %s", name, error)
                continue
        paths.append(path)
    frames = [team_xg_from_archive(paths)]
    try:
        frames.append(fetch_live_team_xg())
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Could not fetch current-season FPL xG: %s", error)
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
