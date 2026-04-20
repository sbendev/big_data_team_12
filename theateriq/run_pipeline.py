#!/usr/bin/env python3
"""End-to-end TheaterIQ prototype on MovieLens 100K (train → ALS → XGB → schedule)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from theateriq.data import default_data_dir, load_items, load_ratings, load_users
from theateriq.features import SLOT_NAMES, add_slot_calendar
from theateriq.scheduler import greedy_schedule
from theateriq.segments import add_segments
from theateriq.slate import synthetic_weekly_slate
from theateriq.stage1_als import export_segment_vectors, train_als
from theateriq.stage2_xgb import (
    build_supervised_frame,
    default_slot_profiles,
    match_score_grid,
    train_ranker,
)
from theateriq.stage2_xgb import predict_proba as ranker_predict
from theateriq.tmdb_join import enrich_movies


def time_based_split(ratings: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    cutoff = ratings["timestamp"].quantile(1.0 - test_fraction)
    test_mask = ratings["timestamp"] >= cutoff
    # ensure every user has some train history when possible
    train = ratings.loc[~test_mask].copy()
    test = ratings.loc[test_mask].copy()
    return train, test, float(cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TheaterIQ pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to ml-100k directory (default: sibling ml-100k or THEATERIQ_ML100K_DIR)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument("--screens", type=int, default=4, help="Number of screens for schedule sketch")
    parser.add_argument(
        "--tmdb-sample",
        type=int,
        default=0,
        help="If >0 and TMDB_API_KEY set, enrich first N movies (slow); 0 skips API",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or default_data_dir()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(data_dir)
    users = load_users(data_dir)
    items = load_items(data_dir)

    users_seg = add_segments(users)
    ratings = ratings.merge(users_seg[["user_id", "segment"]], on="user_id", how="inner")
    ratings = add_slot_calendar(ratings)

    train_r, test_r, cutoff_ts = time_based_split(ratings)
    train_r.to_csv(out_dir / "ratings_train_sample.csv", index=False)
    test_r.to_csv(out_dir / "ratings_test_sample.csv", index=False)

    items_enriched = enrich_movies(items, max_movies=args.tmdb_sample)

    als = train_als(train_r, n_factors=32, reg=0.1, n_iters=25)
    seg_vec = export_segment_vectors(als, users_seg.merge(train_r[["user_id"]].drop_duplicates(), on="user_id"))
    seg_vec.to_csv(out_dir / "segment_als_profiles.csv", index=False)

    train_frame, feature_names = build_supervised_frame(
        train_r, users_seg, items_enriched, als, seg_vec
    )
    ranker = train_ranker(train_frame, feature_names)

    test_frame, _ = build_supervised_frame(test_r, users_seg, items_enriched, als, seg_vec)
    test_frame = test_frame[test_frame["item_id"].isin(als.item_id_to_idx.keys())]
    if len(test_frame) > 0:
        y = test_frame["label_positive"].to_numpy()
        pred = ranker_predict(ranker, test_frame)
        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(y, pred)
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    slate = synthetic_weekly_slate(train_r, items_enriched, n_titles=14, seed=42)
    segments_sorted = sorted(users_seg["segment"].unique())
    profiles = default_slot_profiles()
    grid = match_score_grid(
        ranker,
        als,
        seg_vec,
        items_enriched,
        slate,
        segments_sorted,
        profiles,
    )
    grid.to_csv(out_dir / "match_score_grid.csv", index=False)

    slot_order = [f"slot_{s}" for s in SLOT_NAMES]
    schedule = greedy_schedule(
        grid,
        screens=args.screens,
        slot_order=slot_order,
        archetype="suburban_8plex",
    )
    schedule.to_csv(out_dir / "weekly_schedule_sketch.csv", index=False)

    summary = {
        "data_dir": str(data_dir.resolve()),
        "n_train_ratings": int(len(train_r)),
        "n_test_ratings": int(len(test_r)),
        "time_cutoff_timestamp": cutoff_ts,
        "als_factors": als.user_factors.shape[1],
        "stage2_auc_holdout": auc,
        "slate_item_ids": slate,
        "screens": args.screens,
        "slot_order": slot_order,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("Wrote outputs to", out_dir.resolve())
    print("Holdout AUC (Stage 2, proxy label rating>=4):", round(auc, 4) if auc == auc else auc)


if __name__ == "__main__":
    main()
