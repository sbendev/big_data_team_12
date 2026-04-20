"""Structured-noise synthetic performance model for indie theater showings."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

VENUE_CAPACITY: dict[str, int] = {
    "Trylon Cinema":     175,
    "Riverview Theater": 250,
    "Lagoon":            350,
    "Heights Theater":   500,
}
DEFAULT_CAPACITY = 200

BASE_OCCUPANCY = 0.45

DOW_MULT: dict[str, float] = {
    "Mon": 0.70, "Tue": 0.72, "Wed": 0.75, "Thu": 0.78,
    "Fri": 1.25, "Sat": 1.35, "Sun": 1.10,
}

TICKET_PRICE_EVENING = 13.0
TICKET_PRICE_MATINEE = 10.0


def _parse_hour(showtime_str: str) -> int | None:
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)", str(showtime_str).strip(), re.IGNORECASE)
    if not m:
        return None
    h = int(m.group(1))
    period = m.group(3).upper()
    if period == "PM" and h != 12:
        h += 12
    elif period == "AM" and h == 12:
        h = 0
    return h


def _is_matinee(showtime_str: str) -> bool:
    h = _parse_hour(showtime_str)
    return h is not None and h < 17


def generate_performance(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
    noise_std: float = 0.15,
) -> pd.DataFrame:
    """
    Attach synthetic performance columns to a Film Times DataFrame.

    Required columns: ``Day``, ``Showtime``, ``Venue``.
    Optional column: ``tmdb_popularity`` (shifts occupancy signal if present).

    Added columns
    -------------
    capacity : int
        Estimated seat count for the venue.
    is_matinee : bool
        True if showtime is before 17:00.
    dow_mult : float
        Day-of-week occupancy multiplier.
    slot_mult : float
        0.85 for matinees, 1.0 for evenings.
    tmdb_pop_mult : float
        TMDB popularity normalised to [0.80, 1.20]; 1.0 when no API data.
    occupancy_rate : float
        Fraction of seats filled, clipped to [0.05, 0.98].
    seats_filled : int
        round(occupancy_rate × capacity).
    ticket_price : float
        $13.00 evening / $10.00 matinee.
    revenue : float
        seats_filled × ticket_price.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    out["capacity"] = out["Venue"].map(VENUE_CAPACITY).fillna(DEFAULT_CAPACITY).astype(int)
    out["is_matinee"] = out["Showtime"].apply(_is_matinee)
    out["dow_mult"] = out["Day"].map(DOW_MULT).fillna(0.75)
    out["slot_mult"] = out["is_matinee"].map({True: 0.85, False: 1.0}).fillna(1.0)

    if "tmdb_popularity" in out.columns:
        pop = out["tmdb_popularity"].fillna(0.0).astype(float)
        p_min, p_max = pop.min(), pop.max()
        if p_max > p_min:
            out["tmdb_pop_mult"] = 0.80 + (pop - p_min) / (p_max - p_min) * 0.40
        else:
            out["tmdb_pop_mult"] = 1.0
    else:
        out["tmdb_pop_mult"] = 1.0

    noise = np.clip(rng.normal(1.0, noise_std, size=len(out)), 0.6, 1.4)
    raw_occ = (
        BASE_OCCUPANCY
        * out["dow_mult"].values
        * out["slot_mult"].values
        * out["tmdb_pop_mult"].values
        * noise
    )
    out["occupancy_rate"] = np.clip(raw_occ, 0.05, 0.98)
    out["seats_filled"] = (out["occupancy_rate"] * out["capacity"]).round().astype(int)
    out["ticket_price"] = (
        out["is_matinee"].map({True: TICKET_PRICE_MATINEE, False: TICKET_PRICE_EVENING})
        .fillna(TICKET_PRICE_EVENING)
    )
    out["revenue"] = (out["seats_filled"] * out["ticket_price"]).round(2)

    return out
