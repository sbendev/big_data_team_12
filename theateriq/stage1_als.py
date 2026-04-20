"""Explicit-feedback ALS (MovieLens ratings) → user and item latent factors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlsModel:
    user_factors: np.ndarray  # (n_users, k)
    item_factors: np.ndarray  # (n_items, k)
    user_id_to_idx: dict[int, int]
    item_id_to_idx: dict[int, int]
    idx_to_user_id: dict[int, int]
    idx_to_item_id: dict[int, int]
    global_mean: float


def train_als(
    ratings: pd.DataFrame,
    *,
    n_factors: int = 32,
    reg: float = 0.1,
    n_iters: int = 20,
    random_state: int = 42,
) -> AlsModel:
    rng = np.random.default_rng(random_state)
    u_ids = ratings["user_id"].astype(int).unique()
    i_ids = ratings["item_id"].astype(int).unique()
    user_id_to_idx = {int(u): i for i, u in enumerate(sorted(u_ids))}
    item_id_to_idx = {int(i): j for j, i in enumerate(sorted(i_ids))}
    idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    n_u, n_i = len(user_id_to_idx), len(item_id_to_idx)

    global_mean = float(ratings["rating"].mean())
    user_items: dict[int, list[tuple[int, float]]] = defaultdict(list)
    item_users: dict[int, list[tuple[int, float]]] = defaultdict(list)

    u_idx_arr = ratings["user_id"].map(user_id_to_idx).astype(np.int32).to_numpy()
    i_idx_arr = ratings["item_id"].map(item_id_to_idx).astype(np.int32).to_numpy()
    r_arr = (ratings["rating"].astype(np.float64) - global_mean).to_numpy()

    for ui, ii, r in zip(u_idx_arr, i_idx_arr, r_arr, strict=False):
        user_items[int(ui)].append((int(ii), float(r)))
        item_users[int(ii)].append((int(ui), float(r)))

    U = rng.normal(0, 0.1, size=(n_u, n_factors))
    V = rng.normal(0, 0.1, size=(n_i, n_factors))
    eye = np.eye(n_factors, dtype=np.float64) * reg

    for _ in range(n_iters):
        for ui in range(n_u):
            pairs = user_items.get(ui, [])
            if not pairs:
                continue
            idxs = np.array([p[0] for p in pairs], dtype=np.int32)
            rs = np.array([p[1] for p in pairs], dtype=np.float64)
            V_sub = V[idxs]
            a = V_sub.T @ V_sub + eye
            b = V_sub.T @ rs
            U[ui] = np.linalg.solve(a, b)
        for ij in range(n_i):
            pairs = item_users.get(ij, [])
            if not pairs:
                continue
            user_idx = np.array([p[0] for p in pairs], dtype=np.int32)
            rs = np.array([p[1] for p in pairs], dtype=np.float64)
            U_sub = U[user_idx]
            a = U_sub.T @ U_sub + eye
            b = U_sub.T @ rs
            V[ij] = np.linalg.solve(a, b)

    return AlsModel(
        user_factors=U,
        item_factors=V,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        idx_to_user_id=idx_to_user_id,
        idx_to_item_id=idx_to_item_id,
        global_mean=global_mean,
    )


def predict_rating(model: AlsModel, user_id: int, item_id: int) -> float:
    if user_id not in model.user_id_to_idx or item_id not in model.item_id_to_idx:
        return model.global_mean
    ui = model.user_id_to_idx[user_id]
    ii = model.item_id_to_idx[item_id]
    return float(model.user_factors[ui] @ model.item_factors[ii] + model.global_mean)


def segment_item_affinity(
    model: AlsModel,
    users_with_segments: pd.DataFrame,
    item_id: int,
) -> dict[str, float]:
    """Mean predicted centered affinity (rating - global_mean) by segment."""
    if item_id not in model.item_id_to_idx:
        return {}
    ii = model.item_id_to_idx[item_id]
    v = model.item_factors[ii]
    seg_scores: dict[str, list[float]] = {}
    for _, row in users_with_segments.iterrows():
        uid = int(row["user_id"])
        if uid not in model.user_id_to_idx:
            continue
        ui = model.user_id_to_idx[uid]
        s = float(model.user_factors[ui] @ v)
        seg = str(row["segment"])
        seg_scores.setdefault(seg, []).append(s)
    return {s: float(np.mean(vals)) for s, vals in seg_scores.items()}


def export_segment_vectors(
    model: AlsModel,
    users_with_segments: pd.DataFrame,
) -> pd.DataFrame:
    """Average user latent factor per segment (for Stage 2 features)."""
    k = model.user_factors.shape[1]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for _, row in users_with_segments.iterrows():
        uid = int(row["user_id"])
        if uid not in model.user_id_to_idx:
            continue
        seg = str(row["segment"])
        ui = model.user_id_to_idx[uid]
        vec = model.user_factors[ui]
        if seg not in sums:
            sums[seg] = np.zeros(k, dtype=np.float64)
            counts[seg] = 0
        sums[seg] += vec
        counts[seg] += 1
    rows = []
    for seg, s in sums.items():
        c = counts[seg]
        mean_vec = s / max(c, 1)
        row = {"segment": seg, "segment_user_count": c}
        for j in range(k):
            row[f"seg_als_{j}"] = float(mean_vec[j])
        rows.append(row)
    return pd.DataFrame(rows)
