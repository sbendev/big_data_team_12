"""Greedy assignment of slate titles to screens × slots using Match Scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from theateriq.segments import ARCHETYPES


@dataclass
class ScreenSlot:
    screen_id: int
    slot_key: str


def apply_archetype_boost(
    grid: pd.DataFrame,
    archetype: str = "suburban_8plex",
) -> pd.DataFrame:
    """Multiply scores by segment boosts from theater archetype (narrative prior)."""
    cfg = ARCHETYPES.get(archetype, {})
    boosts = cfg.get("segment_prior_boost", {})
    out = grid.copy()
    if not boosts:
        return out
    mult = out["segment"].map(lambda s: float(boosts.get(s, 1.0)))
    out["match_score_boosted"] = out["match_score_0_100"] * mult
    return out


def greedy_schedule(
    grid: pd.DataFrame,
    screens: int,
    slot_order: list[str],
    *,
    score_col: str = "match_score_0_100",
    archetype: str | None = "suburban_8plex",
) -> pd.DataFrame:
    """
    Each (screen, slot) picks the best unused movie using max segment score for that cell.

    Simplification: for each cell we take max over segments as demand proxy.
    One film cannot occupy two cells (no split prints in this prototype).
    """
    g = apply_archetype_boost(grid, archetype) if archetype else grid.copy()
    col = "match_score_boosted" if archetype and "match_score_boosted" in g.columns else score_col

    agg = g.groupby(["item_id", "title", "slot_key"], as_index=False)[col].max()
    agg = agg.rename(columns={col: "best_segment_score"})

    assignments = []
    used_movies: set[int] = set()
    screen_slot_list = [ScreenSlot(s + 1, sl) for s in range(screens) for sl in slot_order]

    for cell in screen_slot_list:
        cand = agg[(agg["slot_key"] == cell.slot_key) & (~agg["item_id"].isin(used_movies))]
        if cand.empty:
            assignments.append(
                {
                    "screen_id": cell.screen_id,
                    "slot_key": cell.slot_key,
                    "item_id": None,
                    "title": "(no title available)",
                    "best_segment_score": 0.0,
                    "confidence_label": "consider_dropping",
                }
            )
            continue
        best = cand.sort_values("best_segment_score", ascending=False).iloc[0]
        used_movies.add(int(best["item_id"]))
        score = float(best["best_segment_score"])
        if score >= 75:
            label = "high_confidence"
        elif score >= 50:
            label = "promotional_support"
        else:
            label = "consider_dropping"
        assignments.append(
            {
                "screen_id": cell.screen_id,
                "slot_key": cell.slot_key,
                "item_id": int(best["item_id"]),
                "title": str(best["title"]),
                "best_segment_score": score,
                "confidence_label": label,
            }
        )

    return pd.DataFrame(assignments)
