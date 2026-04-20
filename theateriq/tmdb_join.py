"""
TMDB enrichment spec and optional fetch.

Join keys: normalized movie title + release year parsed from MovieLens `title`
(e.g. "Toy Story (1995)" → query "Toy Story", year=1995).

TMDB_API_KEY: set in environment to call the API; without it, `enrich_movies`
returns zero-filled TMDB feature columns so Stage 2 still runs on MovieLens genres.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

TMDB_STAGE2_FEATURES = [
    "tmdb_popularity",
    "tmdb_vote_average",
    "tmdb_vote_count",
    "tmdb_runtime",
    "tmdb_budget_log1p",
    "tmdb_revenue_log1p",
]


@dataclass(frozen=True)
class TmdbJoinSpec:
    title_source_col: str = "title"
    year_source_col: str = "release_year"
    tmdb_search_endpoint: str = "https://api.themoviedb.org/3/search/movie"
    tmdb_movie_endpoint: str = "https://api.themoviedb.org/3/movie"


def parse_title_year(ml_title: str) -> tuple[str, int | None]:
    m = re.search(r"^(.*)\s*\((\d{4})\)\s*$", str(ml_title).strip())
    if not m:
        return str(ml_title).strip(), None
    return m.group(1).strip(), int(m.group(2))


def _budget_features(budget: float | None, revenue: float | None) -> tuple[float, float]:
    b = 0.0 if budget is None or (isinstance(budget, float) and pd.isna(budget)) else float(budget)
    r = 0.0 if revenue is None or (isinstance(revenue, float) and pd.isna(revenue)) else float(revenue)
    return math.log1p(max(b, 0.0)), math.log1p(max(r, 0.0))


def fetch_tmdb_row(movie_id_tmdb: int, api_key: str, session: Any) -> dict[str, float]:
    import requests

    url = f"https://api.themoviedb.org/3/movie/{movie_id_tmdb}"
    r = session.get(url, params={"api_key": api_key}, timeout=30)
    r.raise_for_status()
    d = r.json()
    budg_log, rev_log = _budget_features(d.get("budget"), d.get("revenue"))
    return {
        "tmdb_popularity": float(d.get("popularity") or 0.0),
        "tmdb_vote_average": float(d.get("vote_average") or 0.0),
        "tmdb_vote_count": float(d.get("vote_count") or 0.0),
        "tmdb_runtime": float(d.get("runtime") or 0.0),
        "tmdb_budget_log1p": budg_log,
        "tmdb_revenue_log1p": rev_log,
    }


def enrich_movies(
    items: pd.DataFrame,
    *,
    api_key: str | None = None,
    sleep_s: float = 0.25,
    max_movies: int | None = None,
) -> pd.DataFrame:
    """
    Add TMDB_STAGE2_FEATURES to items. Without api_key, fills zeros and documents
    missing budget as 'not joined' (log1p(0)).
    """
    import requests

    out = items.copy()
    for c in TMDB_STAGE2_FEATURES:
        out[c] = 0.0

    key = api_key or os.environ.get("TMDB_API_KEY")
    if not key:
        return out
    if max_movies == 0:
        return out

    session = requests.Session()
    n = 0
    for idx, row in out.iterrows():
        if max_movies is not None and n >= max_movies:
            break
        title, year = parse_title_year(row["title"])
        if year is None:
            continue
        resp = session.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": key, "query": title, "year": year},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            time.sleep(sleep_s)
            continue
        tmdb_id = results[0]["id"]
        feats = fetch_tmdb_row(tmdb_id, key, session)
        for k, v in feats.items():
            out.at[idx, k] = v
        n += 1
        time.sleep(sleep_s)

    return out
