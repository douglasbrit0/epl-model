"""EPL season labelling shared by fitting, evaluation and the daily pipeline."""

from __future__ import annotations

import pandas as pd

# 2019/20 was suspended in March 2020 and completed in June-July 2020.
_RESTART_2019_20 = (pd.Timestamp("2020-06-01", tz="UTC"), pd.Timestamp("2020-09-01", tz="UTC"))


def season_start_year(value: object) -> int:
    """Return the starting year of the EPL season a UTC date belongs to (2019/20 -> 2019).

    Seasons run July-June, except that the 2019/20 restart (June-August 2020) belongs to 2019/20.
    """
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if _RESTART_2019_20[0] <= stamp < _RESTART_2019_20[1]:
        return 2019
    return stamp.year if stamp.month >= 7 else stamp.year - 1


def season_labels(dates: pd.Series) -> pd.Series:
    """Vectorised ``season_start_year`` for a Series of dates."""
    stamps = pd.to_datetime(dates, utc=True)
    labels = stamps.dt.year.where(stamps.dt.month >= 7, stamps.dt.year - 1)
    restart = (stamps >= _RESTART_2019_20[0]) & (stamps < _RESTART_2019_20[1])
    return labels.where(~restart, 2019).astype(int)
