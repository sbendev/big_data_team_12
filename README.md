# TheatreIQ — Movie Recommender & Scheduling System

TheatreIQ is a two-stage machine learning recommender that generates **Movie Match Scores** and **weekly screening schedules** for independent movie theaters. Given a theater's historical programming and demographic audience profile, it ranks films by predicted audience fit and greedily assigns them to screens and time slots.

---

## How It Works

```
Film Times.csv ──► TMDB Enrich ──► Synthetic Performance ──┐
MovieLens 100K ──► ALS ──► Segment Vectors ────────────────┤
                                                            ▼
                                              XGBoost Match Score Grid
                                                            ▼
                                     Greedy Schedule + Missed Opportunity Analysis
```

**Stage 1 — ALS Collaborative Filtering**  
Trains on MovieLens 100K explicit ratings to learn latent taste vectors for 8 demographic segments (age × gender bins). Films not in MovieLens are "bridged" via title match or genre cosine similarity.

**Stage 2 — XGBoost Ranker**  
Scores every (segment × film × showtime slot) triple using 74 features: segment ALS vectors, item ALS vectors, 19 genre flags, 6 TMDB metadata features, and 8 slot/calendar features. Output is a Match Score from 0–100.

**Scheduler**  
Greedy assignment fills each (screen × slot) cell with the highest-scoring available film, with optional archetype boosts for theater type. Films are labeled `high_confidence` (≥75), `promotional_support` (50–74), or `consider_dropping` (<50).

---

## Data

### Film Times (`Film Times.csv`)
Historical and synthetic showtime data scraped from Twin Cities theaters.

| Venue | Showings |
|---|---|
| Drafthouse | 10,346 |
| Heights Theater | 5,605 |
| Lagoon | 665 |
| Trylon Cinema | 425 |
| Riverview Theater | 414 |
| **Total** | **17,455** |

Date range: Feb 2021 – May 2026 (includes synthetic future programming).  
Columns: `Date`, `Day`, `Showtime`, `Film Title`, `Film Year`, `Venue`

### MovieLens 100K (`ml-100k/`)
100,000 explicit ratings used for ALS training. Place the `ml-100k/` directory in the project root.

### TMDB (optional)
Set `TMDB_API_KEY` in a `.env` file to enrich films with popularity, ratings, runtime, budget, and genre metadata. The pipeline runs without a key (genre features zero-filled with a Drama prior).

---

## Venue Configuration

| Venue | Capacity | Screens |
|---|---|---|
| Trylon Cinema | 175 seats | 2 |
| Riverview Theater | 250 seats | 3 |
| Lagoon | 350 seats | 4 |
| Heights Theater | 500 seats | 2 |
| Drafthouse | 150 seats | 8 |

---

## Project Structure

```
theateriq/
├── data.py                 # MovieLens 100K loaders
├── stage1_als.py           # ALS collaborative filtering (k=32)
├── stage2_xgb.py           # XGBoost ranker (74 features)
├── features.py             # Slot & calendar feature engineering
├── segments.py             # 8 demographic segments + theater archetypes
├── tmdb_join.py            # TMDB metadata enrichment
├── synthetic_performance.py # Occupancy & revenue simulation
├── scheduler.py            # Greedy screen-slot assignment
├── slate.py                # Synthetic weekly slate construction
└── run_pipeline.py         # End-to-end orchestration script
TheatreIQ_FirstPass.ipynb   # Full pipeline notebook
Film Times.csv              # Showtime dataset (17,455 rows, 5 venues)
docs/twin_cities_theaters.csv  # Theater metadata reference
```

---

## Running the Pipeline

1. Install dependencies: `pip install numpy pandas scikit-learn xgboost requests python-dotenv`
2. Place `ml-100k/` in the project root
3. (Optional) Add `TMDB_API_KEY=<your_key>` to `.env`
4. Open and run `TheatreIQ_FirstPass.ipynb` top-to-bottom

Outputs are written to `.cache/` (TMDB enrichment) and `outputs/` (schedule artifacts).

---

## Recent Updates

- **Expanded dataset**: Film Times grew from 1,524 → 17,455 rows with synthetic data covering all five venues through May 2026
- **New venue — Drafthouse**: Added to scheduling pipeline (8 screens, 150-seat capacity) and synthetic performance model
- **Improved TMDB enrichment**: Title cleaning now strips embedded year suffixes (e.g. `"Playtime (1967)"` → `"Playtime"`) and Drafthouse event annotations (e.g. `"Tenet (2020) — REVIVAL"` → `"Tenet"`) before querying the API; a gap-fill pass propagates data from duplicate clean-title entries and retries remaining misses
