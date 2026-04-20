"""Synthetic weekly bookable slate (titles a mid-market theater could actually play)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_weekly_slate(
    ratings_train: pd.DataFrame,
    items: pd.DataFrame,
    n_titles: int = 14,
    n_tail_explore: int = 4,
    seed: int = 42,
) -> list[int]:
    """
    Build a slate: popular titles plus a few long-tail picks (exploration).

    Real theaters book from distributor availability; we approximate with
    popularity on the training split plus random tail items.
    """
    rng = np.random.default_rng(seed)
    counts = ratings_train.groupby("item_id").size()
    head_n = max(0, n_titles - n_tail_explore)
    popular = counts.nlargest(head_n).index.astype(int).tolist()
    all_items = set(items["item_id"].astype(int))
    remaining = list(all_items - set(popular))
    if len(remaining) >= n_tail_explore:
        tail = rng.choice(remaining, size=n_tail_explore, replace=False).tolist()
    else:
        tail = remaining
    slate = popular + [int(x) for x in tail]
    # De-dupe preserving order
    seen: set[int] = set()
    ordered: list[int] = []
    for m in slate:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered[:n_titles]
