# TheatreIQ — Model Rationale & Design Decisions

## 1. Overall Approach

### The Problem
We wanted to help Minneapolis independent theaters (Trylon, Riverview, Lagoon, Heights, Drafthouse) make smarter programming and marketing decisions. The core question: **which films should a theater schedule, for which audience, and when?**

The fundamental constraint was data scarcity. Independent theaters do not publish box office records. We had:
- Real historical showing schedules (~17,400 rows across 5 venues, 2021–2026)
- No attendance or revenue data
- No individual ticket purchaser data
- 750 unique films on the combined slate

### Our Solution: A Two-Stage Pipeline

```
Film Times.csv ──► TMDB Enrich ──► Synthetic Performance ──┐
MovieLens 100K ──► ALS ──► Segment Vectors ────────────────┤
                                                            ▼
                                              XGBoost Match Score Grid
                                                            ▼
                                   Greedy Schedule + Missed Opportunity Analysis
```

**Stage 1 — ALS on MovieLens 100K:** Learn *which types of films resonate with which demographic groups* from 100,000 real ratings by 943 users. Output: 32-dimensional latent "taste vectors" for 8 demographic segments.

**Stage 2 — XGBoost Ranker:** Combine those taste vectors with film metadata (genre flags, TMDB popularity, runtime, budget) and time-slot features to predict whether a given film is a strong match for a given segment at a given showing time. Output: a match score from 0–100 for every (film, segment, slot) combination.

**Key design constraint — no individual tracking:** All modeling happens at the *segment* level (age band × gender), not the individual user level. This is both a practical necessity (no ticketing data) and a privacy-appropriate choice for a community-oriented venue.

---

## 2. Why ALS?

### What ALS Does
Alternating Least Squares (ALS) is a matrix factorization method. Given a sparse user-item ratings matrix, it learns two dense matrices — user latent factors and item latent factors — such that their dot product approximates the original ratings. Each latent dimension captures a meaningful but unobserved taste axis (e.g., "appreciates slow cinema," "prefers genre action").

### Why It Was the Right Choice Here

**Collaborative filtering captures taste, not just genre.**
A genre flag tells you "this film is a Drama." ALS tells you "users who loved this film also loved these other films" — it captures the *latent character* of a film as perceived by real audiences. For niche arthouse programming, this distinction matters: two Dramas can serve completely different audiences.

**It solves the demographic signal problem directly.**
MovieLens includes user demographics (age, gender). We aggregate individual user factors up to 8 demographic segments. These segment vectors become the demographic taste features in Stage 2. Without ALS, we would have no principled way to express "what does a 35–54 year-old woman tend to prefer?" in quantitative terms.

**It handles sparsity well.**
MovieLens 100K is a sparse matrix (~6% density). ALS with L2 regularization (`reg=0.1`) is specifically designed for sparse collaborative filtering — it avoids overfitting to users with few ratings.

**Interpretable hyperparameters, well-understood behavior.**
With `k=32` factors, `20` iterations, and `reg=0.1`, ALS is stable and deterministic (given `random_state=42`). The solution is a classical least-squares solve per user/item row — no black-box optimization, no learning rate tuning.

### ALS vs. Alternatives

| Alternative | Why We Didn't Use It |
|---|---|
| SVD / SVD++ | Requires dense matrix or imputation; ALS handles observed-only ratings more naturally |
| Neural Collaborative Filtering (NCF) | Needs more data and compute; harder to export as simple segment vectors for Stage 2 |
| Cosine similarity / item-KNN | Only captures item–item similarity; can't produce demographic segment vectors |
| Matrix factorization with implicit feedback | Our data is explicit (1–5 star ratings), so explicit ALS is more appropriate |

### Cold-Start Resolution
Real theater films (Playtime, Fitzcarraldo, etc.) don't exist in MovieLens. We bridge this gap with a priority hierarchy:
1. **Exact title match** in MovieLens → use its actual item factors
2. **Genre cosine similarity** → weighted average of top-3 nearest MovieLens items by genre overlap
3. **Drama fallback** → mean of all MovieLens Drama items (arthouse prior)

Result: 52 exact matches, 657 genre bridges, 41 fallbacks — 94.5% resolved non-trivially.

---

## 3. Why XGBoost?

### What XGBoost Does Here
XGBoost is a gradient-boosted decision tree ensemble. We use it as a **binary classifier**: given a (user-segment, film, slot) triple, predict whether that combination is a strong match (label = 1 if the MovieLens rating would be ≥ 4 stars).

At inference time, the predicted probability becomes the **match score (0–100)** we use for scheduling and marketing decisions.

### Why It Was the Right Choice Here

**It naturally handles heterogeneous features.**
Our 97-feature vector mixes four very different data types:
- 64 ALS latent floats (segment vectors + item vectors) — dense, continuous, abstract
- 19 genre binary flags — sparse, categorical
- 6 TMDB numeric features (popularity, vote average, runtime, log-budget, log-revenue) — continuous, different scales
- 8 slot/calendar features (time slot one-hots, day-of-week, month, is-weekend) — categorical

Decision trees handle all of these natively without normalization or embedding layers. A linear model would require careful feature engineering to capture interactions (e.g., "this segment likes Drama *specifically on weekend evenings*").

**It captures non-linear interactions.**
The relationship between demographic taste and film fit is not linear. A 35–54 M segment might score Documentary highly in a weekday slot but not a Friday prime slot. XGBoost's tree structure captures these conditional interactions automatically.

**Interpretable feature importance.**
We can see which features drive predictions (top features by information gain: `item_als_2`, `item_als_29`, `Documentary`, `Drama`). This lets us audit whether the model is relying on meaningful signals.

**Fast training and inference.**
80,000 training rows, 97 features, 80 boosting rounds with `max_depth=6` — trains in seconds. Scoring 750 films × 8 segments × 5 slots = 30,000 predictions is also near-instant.

### XGBoost vs. Alternatives

| Alternative | Why We Didn't Use It |
|---|---|
| Logistic Regression | Can't capture non-linear feature interactions without manual engineering |
| Neural Network (MLP) | Harder to interpret, needs more data to outperform trees on tabular data |
| Random Forest | XGBoost typically outperforms on tabular data; also slower at inference |
| LightGBM | Equally valid choice; XGBoost was selected for transparency and familiarity |
| Pure ALS scoring | Can't incorporate TMDB metadata, genre flags, or slot/calendar features |

### Why This Is a Two-Stage Architecture (Not One)
We deliberately separate the collaborative signal (ALS, Stage 1) from the content/context signal (XGBoost, Stage 2). This is the industry-standard pattern for recommendation systems for two reasons:

1. **Modularity:** The ALS model can be retrained on new rating data without touching the XGBoost ranker, and vice versa.
2. **Cold-start handling:** ALS fails on films with no ratings. By bridging film factors through genre similarity, Stage 2 can score any film — even one released after the ALS training cutoff.

---

## 4. Why ROC-AUC as the Evaluation Metric?

### What ROC-AUC Measures
ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures a model's ability to **rank** positives above negatives across all decision thresholds. An AUC of 0.5 = random; 1.0 = perfect.

Our holdout result: **AUC = 0.696** on the MovieLens time-based test set.

### Why AUC Is the Right Metric Here

**We care about ranking, not absolute rating prediction.**
The end output is a ranked list of films per slot per segment — we pick the top-k. Whether the model predicts a "match score" of 72 vs. 78 for the top film matters less than whether it correctly ranks it above a score-40 film. AUC directly measures this ranking quality.

**The label is binary.**
We converted the 1–5 star scale to a binary label (≥ 4 = positive, < 4 = negative). With a binary label, RMSE is less meaningful than AUC. AUC also handles the class imbalance gracefully (positive rate = 55.1% in training) without requiring threshold tuning.

**AUC is threshold-agnostic.**
We do not know in advance what match score threshold to use for "high confidence" vs. "promotional support." AUC evaluates the model across all possible thresholds simultaneously, making it a more honest summary of model quality.

### Why Not RMSE?

RMSE would require predicting the actual rating on a 1–5 scale. Our task is *classification* (is this a good match?) not *regression* (how many stars would this user give?). Using RMSE on a binary-labeled XGBoost would be an objective mismatch.

### Why a Time-Based Split?

We split the MovieLens ratings at the 80th percentile timestamp (train = earlier 80%, test = later 20%). A random split would allow information from future ratings to leak into training through items and users seen in both splits — artificially inflating AUC. The time-based split simulates real deployment: the model is trained on historical data and evaluated on future behavior.

### Interpreting AUC = 0.696

This is a reasonable result given the constraints:
- MovieLens users and our theater's actual audience are **different populations** (MovieLens skews toward movie enthusiasts broadly; Minneapolis arthouse patrons are a narrower segment)
- We use **segment-averaged** factors, not individual user factors, which reduces signal precision
- The feature set is intentionally minimal (no cast/director/review-text features)

A well-tuned production system with matched training data would target AUC > 0.80.

---

## 5. What Would We Do in the Next Iteration?

### Highest-Impact Improvements

**1. Replace synthetic performance data with real attendance records.**
The single biggest limitation of the current system is that occupancy rates are simulated using a noise model, not measured. Even anonymized ticket scan counts from a box office system would transform Stage 2 training — we could optimize directly for revenue or seats filled rather than a proxy from MovieLens ratings.

**2. Train on theater-native ratings or survey data.**
MovieLens users are a reasonable demographic proxy but an imperfect one. A short post-film rating kiosk ("How much did you enjoy tonight's film?") or opt-in mobile survey at the ticket window would generate audience-specific training data within 1–2 seasons.

**3. Add cast, director, and critic review features.**
Currently, content features are limited to genre flags and TMDB aggregate stats. Adding director embeddings (derived from their filmography ALS vectors), cast affinity scores, and Rotten Tomatoes/Letterboxd ratings would significantly improve cold-start quality for new releases and obscure catalog titles.

**4. Extend the cold-start bridge with text embeddings.**
The genre cosine similarity bridge is functional but crude. A sentence embedding of the film's synopsis (via a small language model) would produce a richer similarity signal — better distinguishing "slow art-house drama" from "prestige Oscar drama" even when both are tagged `Drama`.

**5. Move from binary classification to a direct revenue optimization objective.**
XGBoost currently minimizes binary cross-entropy (predict rating ≥ 4). A better objective for a theater is: maximize expected revenue per screen per week. With real attendance data, we could train an LTV-weighted ranker (LambdaRank or LambdaMART) that accounts for ticket price, expected occupancy, and ancillary spend (concessions).

**6. Incorporate seasonality and local event context.**
The current slot/calendar features are generic (day of week, month). Minneapolis-specific signals — University of Minnesota academic calendar, Minnesota State Fair, major sporting events, weather patterns — would capture audience availability that a generic calendar misses.

**7. Evaluate with Precision@K and NDCG instead of (or alongside) AUC.**
For a scheduler recommending the top 2–4 films per screen per week, Precision@5 and NDCG@5 are more directly interpretable than AUC. They measure: "of the 5 films we recommend most highly, how many would actually be hits?"

### Lower-Priority but Valuable

- **A/B test the schedule recommendations** against programmer intuition over a full season to validate model lift
- **Personalized marketing triggers:** when a high-match film is booked, automatically flag which segment to target and in which slot (the slot × segment preference matrix already supports this)
- **Venue-specific model fine-tuning:** Trylon and Drafthouse have very different audience compositions; fine-tuning segment weights per venue would improve recommendation specificity

---

## Summary

| Decision | Choice | Core Reason |
|---|---|---|
| Stage 1 algorithm | ALS (explicit feedback) | Best method for extracting demographic taste signals from sparse ratings data |
| Stage 2 algorithm | XGBoost (binary classifier) | Handles heterogeneous features, captures non-linear interactions, interpretable |
| Training data source | MovieLens 100K | Only available source of real user-film ratings with demographic metadata |
| Performance proxy | Synthetic occupancy | No real attendance data exists; structured noise model is transparent about its limits |
| Evaluation metric | ROC-AUC | Ranking metric aligned with the task; threshold-agnostic; handles binary labels |
| Train/test split | Time-based 80/20 | Prevents temporal leakage; simulates real deployment conditions |
| Segment granularity | 8 groups (age band × gender) | Matches MovieLens demographic fields; coarse enough to have statistical weight |
| Scheduling strategy | Greedy top-k by match score | Simple, explainable, and fast — appropriate for a first-pass system |
