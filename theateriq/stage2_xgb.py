"""Stage 2: XGBoost on ALS + content + slot/calendar (+ optional TMDB)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from theateriq.data import GENRE_COLS
from theateriq.features import SLOT_NAMES
from theateriq.stage1_als import AlsModel
from theateriq.tmdb_join import TMDB_STAGE2_FEATURES

META_COLS = ["cal_month", "cal_dow", "cal_is_weekend"]


@dataclass
class RankerModel:
    booster: object
    feature_names: list[str]


def _item_factor_frame(model: AlsModel, items: pd.DataFrame) -> pd.DataFrame:
    k = model.item_factors.shape[1]
    rows = []
    for _, row in items.iterrows():
        mid = int(row["item_id"])
        rid = {"item_id": mid}
        if mid in model.item_id_to_idx:
            ii = model.item_id_to_idx[mid]
            vec = model.item_factors[ii]
            for j in range(k):
                rid[f"item_als_{j}"] = float(vec[j])
        else:
            for j in range(k):
                rid[f"item_als_{j}"] = 0.0
        rows.append(rid)
    return pd.DataFrame(rows)


def build_supervised_frame(
    ratings: pd.DataFrame,
    users_seg: pd.DataFrame,
    items: pd.DataFrame,
    model: AlsModel,
    segment_vectors: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """One row per rating with features and binary label (rating >= 4)."""
    k = model.user_factors.shape[1]
    seg_cols = [f"seg_als_{j}" for j in range(k)]
    u = users_seg[["user_id", "segment"]].drop_duplicates()
    r = ratings.merge(u, on="user_id", how="inner")
    r = r.merge(segment_vectors, on="segment", how="left")
    for c in seg_cols:
        if c not in r.columns:
            r[c] = 0.0
        r[c] = r[c].fillna(0.0)

    item_f = _item_factor_frame(model, items[["item_id"]].drop_duplicates())
    r = r.merge(item_f, on="item_id", how="left")
    for j in range(k):
        c = f"item_als_{j}"
        r[c] = r[c].fillna(0.0)

    items_df = items.copy()
    for c in TMDB_STAGE2_FEATURES:
        if c not in items_df.columns:
            items_df[c] = 0.0
    tmdb_cols = list(TMDB_STAGE2_FEATURES)
    r = r.merge(items_df[["item_id"] + GENRE_COLS + tmdb_cols], on="item_id", how="left")
    for g in GENRE_COLS + tmdb_cols:
        r[g] = r[g].fillna(0)

    slot_cols = [f"slot_{s}" for s in SLOT_NAMES]
    for c in slot_cols:
        if c not in r.columns:
            r[c] = 0
    for m in META_COLS:
        if m not in r.columns:
            r[m] = 0

    feature_names = (
        seg_cols
        + [f"item_als_{j}" for j in range(k)]
        + GENRE_COLS
        + tmdb_cols
        + slot_cols
        + META_COLS
    )
    r["label_positive"] = (r["rating"] >= 4).astype(int)
    return r, feature_names


def train_ranker(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    seed: int = 42,
) -> RankerModel:
    import xgboost as xgb

    X = train_frame[feature_names].to_numpy(dtype=np.float32)
    y = train_frame["label_positive"].to_numpy()
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "seed": seed,
    }
    booster = xgb.train(params, dtrain, num_boost_round=80, verbose_eval=False)
    return RankerModel(booster=booster, feature_names=feature_names)


def predict_proba(ranker: RankerModel, frame: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    X = frame[ranker.feature_names].to_numpy(dtype=np.float32)
    d = xgb.DMatrix(X, feature_names=ranker.feature_names)
    return ranker.booster.predict(d)


def match_score_grid(
    ranker: RankerModel,
    model: AlsModel,
    segment_vectors: pd.DataFrame,
    items_enriched: pd.DataFrame,
    slate: list[int],
    segments: list[str],
    slot_profiles: list[dict[str, object]],
) -> pd.DataFrame:
    """
    slot_profiles: list of dicts with keys slot_* one-hots, cal_month, cal_dow, cal_is_weekend
    """
    fn = ranker.feature_names
    k = model.user_factors.shape[1]
    seg_cols = [f"seg_als_{j}" for j in range(k)]
    meta_rows: list[dict] = []
    feat_rows: list[dict] = []

    slate_items = items_enriched[items_enriched["item_id"].isin(slate)].copy()
    item_f = _item_factor_frame(model, slate_items[["item_id"]].drop_duplicates())
    slate_items = slate_items.drop(columns=[c for c in slate_items.columns if c.startswith("item_als_")], errors="ignore")
    slate_items = slate_items.merge(item_f, on="item_id")

    tmdb_cols = [c for c in TMDB_STAGE2_FEATURES if c in fn]

    for seg in segments:
        sv = segment_vectors[segment_vectors["segment"] == seg]
        if sv.empty:
            seg_vec = {c: 0.0 for c in seg_cols}
        else:
            seg_vec = {c: float(sv.iloc[0].get(c, 0.0) or 0.0) for c in seg_cols}
        for slot in slot_profiles:
            for _, movie in slate_items.iterrows():
                feat: dict[str, float] = {}
                for name in fn:
                    if name in seg_vec:
                        feat[name] = seg_vec[name]
                    elif name.startswith("item_als_"):
                        feat[name] = float(movie.get(name, 0.0) or 0.0)
                    elif name in GENRE_COLS or name in tmdb_cols:
                        feat[name] = float(movie.get(name, 0.0) or 0.0)
                    elif name.startswith("slot_"):
                        v = slot.get(name, 0)
                        feat[name] = float(int(v) if v is not None else 0)
                    elif name in META_COLS:
                        v = slot.get(name, 0)
                        feat[name] = float(v)
                    else:
                        feat[name] = 0.0
                feat_rows.append(feat)
                slot_key = next(
                    (f"slot_{s}" for s in SLOT_NAMES if int(slot.get(f"slot_{s}", 0)) == 1),
                    "slot_unknown",
                )
                meta_rows.append(
                    {
                        "segment": seg,
                        "item_id": int(movie["item_id"]),
                        "title": str(movie.get("title", "")),
                        "slot_key": slot_key,
                    }
                )

    feat_df = pd.DataFrame(feat_rows, columns=fn)
    p = predict_proba(ranker, feat_df)
    out = pd.DataFrame(meta_rows)
    out["match_score_0_100"] = np.clip(p * 100.0, 0.0, 100.0)
    return out


def default_slot_profiles() -> list[dict[str, object]]:
    """Synthetic fixed slots for grid scoring (no timestamp)."""
    profiles = []
    base = {"cal_month": 6, "cal_dow": 4, "cal_is_weekend": 1}
    for name in ("fri_prime", "sat_matinee", "sat_prime", "sun_matinee", "weekday_other"):
        p = {f"slot_{s}": 1 if s == name else 0 for s in ("fri_prime", "sat_matinee", "sat_prime", "sun_matinee", "weekday_other")}
        p.update(base)
        if name == "sat_matinee":
            p["cal_dow"] = 5
        elif name == "sat_prime":
            p["cal_dow"] = 5
        elif name == "sun_matinee":
            p["cal_dow"] = 6
        elif name == "weekday_other":
            p["cal_dow"] = 2
            p["cal_is_weekend"] = 0
        profiles.append(p)
    return profiles
