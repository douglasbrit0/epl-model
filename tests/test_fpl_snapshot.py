from datetime import datetime, timezone

import pandas as pd
import pytest

from epl_model.data import fpl


def payload():
    return {
        "teams": [{"id": 1, "name": "Man City"}, {"id": 2, "name": "Nott'm Forest"}],
        "events": [{"id": 6, "is_next": False}, {"id": 7, "is_next": True}],
        "elements": [
            {"id": 10, "code": 100, "web_name": "Keeper", "element_type": 1, "team": 1, "status": "a",
             "chance_of_playing_next_round": None, "news": "", "news_added": None, "expected_goals": "0.00"},
            {"id": 11, "code": 101, "web_name": "Striker", "element_type": 4, "team": 2, "status": "i",
             "chance_of_playing_next_round": 25, "news": "Hamstring injury - 25% chance of playing",
             "news_added": "2026-10-11T09:00:00Z", "expected_goals": "2.31"},
        ],
    }


def test_snapshot_rows_are_timestamped_and_use_canonical_teams():
    frame = fpl.snapshot_frame(payload(), datetime(2026, 10, 12, 6, tzinfo=timezone.utc))
    assert frame["fetched_at"].unique().tolist() == ["2026-10-12T06:00:00Z"]
    assert frame["team"].tolist() == ["Manchester City", "Nottingham Forest"]
    assert frame["position"].tolist() == ["GK", "FWD"]
    assert frame.loc[1, "chance_of_playing_next_round"] == 25
    assert frame["next_gameweek"].unique().tolist() == [7]


def test_snapshots_are_append_only_and_round_trip(tmp_path):
    frame = fpl.snapshot_frame(payload(), datetime(2026, 10, 12, 6, tzinfo=timezone.utc))
    path = fpl.save_snapshot(frame, tmp_path)
    with pytest.raises(FileExistsError):
        fpl.save_snapshot(frame, tmp_path)
    loaded = fpl.load_snapshots(tmp_path)
    assert path.name == "fpl_20261012T060000Z.csv.gz"
    assert loaded["news"].fillna("").tolist() == ["", "Hamstring injury - 25% chance of playing"]


def test_snapshot_failure_never_raises(tmp_path, monkeypatch):
    def broken():
        raise ConnectionError("FPL down")

    monkeypatch.setattr(fpl, "fetch_bootstrap", broken)
    assert fpl.take_snapshot(tmp_path) is None
    assert list(tmp_path.iterdir()) == []
    assert fpl.load_snapshots(tmp_path).empty
