"""Slot and calendar features from rating timestamps (proxy for showtime context)."""

from __future__ import annotations

import pandas as pd

SLOT_NAMES = ("fri_prime", "sat_matinee", "sat_prime", "sun_matinee", "weekday_other")


def infer_slot(dt: pd.Timestamp) -> str:
    """Map UTC timestamp to a coarse showtime bucket (course proxy)."""
    if pd.isna(dt):
        return "weekday_other"
    dow = int(dt.dayofweek)  # Mon=0
    hour = int(dt.hour)
    if dow == 4 and hour >= 17:
        return "fri_prime"
    if dow == 5 and hour < 17:
        return "sat_matinee"
    if dow == 5 and hour >= 17:
        return "sat_prime"
    if dow == 6 and hour < 17:
        return "sun_matinee"
    return "weekday_other"


def add_slot_calendar(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    out = df.copy()
    slots = out[dt_col].apply(infer_slot)
    for s in SLOT_NAMES:
        out[f"slot_{s}"] = (slots == s).astype(int)
    out["cal_month"] = out[dt_col].dt.month.fillna(1).astype(int)
    out["cal_dow"] = out[dt_col].dt.dayofweek.fillna(0).astype(int)
    out["cal_is_weekend"] = out["cal_dow"].isin([4, 5, 6]).astype(int)
    return out
