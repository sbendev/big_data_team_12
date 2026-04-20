"""
Demographic segments (MovieLens proxy) and theater archetypes for pilot narrative.

MovieLens users carry age, gender, and ZIP — not theater patrons. We use age×gender
bins as stand-in segments and document a pilot ZIP for local-market framing.
"""

from __future__ import annotations

import pandas as pd

# Pilot ZIP: Scottsdale, AZ — example suburban market (course narrative only).
PILOT_ZIP_EXAMPLE = "85254"

ARCHETYPES: dict[str, dict] = {
    "suburban_8plex": {
        "description": "Eight-screen suburban multiplex; family and mainstream skew.",
        "pilot_zip": PILOT_ZIP_EXAMPLE,
        "segment_prior_boost": {
            # Multiplier on Match Score contribution by segment id (scheduler can use).
            "age18_34_F": 1.05,
            "age18_34_M": 1.05,
            "age35_54_F": 1.08,
            "age35_54_M": 1.05,
            "under18_F": 1.1,
            "under18_M": 1.08,
            "55plus_F": 0.98,
            "55plus_M": 0.98,
        },
    },
    "urban_boutique": {
        "description": "Four-screen urban arthouse; skew toward older and art/genre titles.",
        "pilot_zip": "10012",
        "segment_prior_boost": {
            "age18_34_F": 1.02,
            "age18_34_M": 1.0,
            "age35_54_F": 1.06,
            "age35_54_M": 1.06,
            "under18_F": 0.92,
            "under18_M": 0.92,
            "55plus_F": 1.08,
            "55plus_M": 1.06,
        },
    },
}


def assign_segment(age: int, gender: str) -> str:
    g = "M" if str(gender).upper().startswith("M") else "F"
    if age < 18:
        return f"under18_{g}"
    if age < 35:
        return f"age18_34_{g}"
    if age < 55:
        return f"age35_54_{g}"
    return f"55plus_{g}"


def add_segments(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out["segment"] = [
        assign_segment(int(a), str(g)) for a, g in zip(out["age"], out["gender"])
    ]
    return out
