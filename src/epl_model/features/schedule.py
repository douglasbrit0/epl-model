"""Schedule features: European fixtures and rest days around each league match.

European fixtures come from openfootball's champions-league repository (CC0), Football.TXT files
saved as ``data/raw/openfootball/<season>_<cl|el|conf>.txt``. Coverage: Champions League every season;
Europa League 2020/21-2024/25; Conference League 2021/22-2024/25. Missing competitions mean flags are
0 where a club actually played in Europe. Domestic cups are not covered by any source used here.

Per team-match (all known before kickoff, since fixtures are published in advance):
- ``europe_prev``: played a European match 1-4 days before.
- ``europe_next``: has a European match 1-4 days after.
- ``short_rest``: at most 3 days since its previous league or European match.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from epl_model.data.names import DEFAULT_MAP

MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
DATE = re.compile(r"^\s*\[?(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?")
ENGLISH = re.compile(r"([A-Za-z0-9&'.\- ]+?)\s*\(ENG\)")
FEATURES = ("europe_prev", "europe_next", "short_rest")
COVARIATES = ("europe_prev", "opp_europe_prev", "europe_next", "opp_europe_next", "short_rest", "opp_short_rest")


def _canonical_lookup() -> dict[str, str]:
    table = pd.read_csv(DEFAULT_MAP, dtype=str).fillna("")
    lookup = {}
    for _, row in table.iterrows():
        for name in (row["football_data_org"], row["canonical"]):
            if name:
                lookup[name] = row["canonical"]
                lookup[name.removesuffix(" FC").removesuffix(" AFC")] = row["canonical"]
    lookup["Brighton & Hove Albion"] = "Brighton"
    return lookup


def european_matches(directory: Path) -> pd.DataFrame:
    """Dates of English clubs' European matches (team = canonical name)."""
    lookup = _canonical_lookup()
    rows = []
    for path in sorted(directory.glob("*.txt")):
        season, competition = path.stem.split("_")
        first_year, date = int(season[:4]), None
        for line in path.read_text(encoding="utf-8").splitlines():
            match = DATE.match(line)
            if match:
                month, day = MONTHS[match.group(2)], int(match.group(3))
                year = int(match.group(4)) if match.group(4) else (first_year if month >= 7 else first_year + 1)
                date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
                line = line[match.end():]
            if " v " not in line or date is None:
                continue
            for name in ENGLISH.findall(line):
                name = re.sub(r"^(?:[\d\s.:]+|v\s+)+", "", name).strip()  # drop kickoff time and "v " prefix
                if name in lookup:
                    rows.append({"team": lookup[name], "date": date, "competition": competition})
    return pd.DataFrame(rows, columns=["team", "date", "competition"]).drop_duplicates()


def schedule_features(matches: pd.DataFrame, europe: pd.DataFrame) -> pd.DataFrame:
    """Per (team, league match date) schedule features."""
    league = pd.concat([
        pd.DataFrame({"team": matches["home"], "date": pd.to_datetime(matches["kickoff_utc"], utc=True).dt.normalize()}),
        pd.DataFrame({"team": matches["away"], "date": pd.to_datetime(matches["kickoff_utc"], utc=True).dt.normalize()}),
    ])
    european = europe.assign(date=pd.to_datetime(europe["date"], utc=True).dt.normalize())
    rows = []
    for team, games in league.groupby("team"):
        dates = games["date"].sort_values().drop_duplicates()
        euro = european.loc[european["team"] == team, "date"].sort_values()
        everything = pd.concat([dates, euro]).sort_values().drop_duplicates().to_numpy()
        euro_values = euro.to_numpy()
        for date in dates.to_numpy():
            gaps = (date - euro_values) / pd.Timedelta(days=1)
            earlier = everything[everything < date]
            rest = (date - earlier[-1]) / pd.Timedelta(days=1) if len(earlier) else 99.0
            rows.append({
                "team": team,
                "date": pd.Timestamp(date),
                "europe_prev": float(((gaps >= 1) & (gaps <= 4)).any()),
                "europe_next": float(((gaps <= -1) & (gaps >= -4)).any()),
                "short_rest": float(rest <= 3),
            })
    return pd.DataFrame(rows)


def attach_schedule(matches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Add covariates: own flags act on the team's scoring, the opponent's flags via ``opp_*``."""
    lookup = features.drop_duplicates(["team", "date"]).set_index(["team", "date"])[list(FEATURES)]
    out = matches.copy()
    day = pd.to_datetime(out["kickoff_utc"], utc=True).dt.normalize()
    sides = {side: lookup.reindex(pd.MultiIndex.from_arrays([out[side], day])).reset_index(drop=True).fillna(0.0) for side in ("home", "away")}
    for side, other in (("home", "away"), ("away", "home")):
        for name in FEATURES:
            out[f"{side}_{name}"] = sides[side][name].to_numpy()
            out[f"{side}_opp_{name}"] = sides[other][name].to_numpy()
    return out
