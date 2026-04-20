from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

GENRE_COLS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def default_data_dir() -> Path:
    env = os.environ.get("THEATERIQ_ML100K_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "ml-100k"


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    root = data_dir or default_data_dir()
    path = root / "u.data"
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def load_users(data_dir: Path | None = None) -> pd.DataFrame:
    root = data_dir or default_data_dir()
    path = root / "u.user"
    return pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="python",
    )


def load_items(data_dir: Path | None = None) -> pd.DataFrame:
    root = data_dir or default_data_dir()
    path = root / "u.item"
    cols = (
        ["item_id", "title", "release_date", "video_release", "imdb_url"]
        + GENRE_COLS
    )
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1", engine="python")
    df["release_year"] = df["title"].str.extract(r"\((\d{4})\)").astype("float")
    return df
