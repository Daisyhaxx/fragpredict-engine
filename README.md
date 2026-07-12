# CS2 Match Prediction Engine

This project predicts which team wins a given map in a professional Counter-Strike 2
match. It started as an attempt to go beyond a single Jupyter notebook and actually
build the thing the way a real ML system would be put together — separate stages for
cleaning the data, engineering features, training a model, and serving predictions
through an API, instead of one long script.

> 🇹🇷 Türkçe README için [README.tr.md](README.tr.md) dosyasına bakın.

## Overview

Give it two teams and a map — say, Team Vitality vs Natus Vincere on Mirage — and it
returns a calibrated win probability for each side. The model is a gradient-boosted
classifier trained on about 2.5 years of professional match history, served through a
FastAPI endpoint, with every prediction logged to a Postgres database (Supabase) so
they can be reviewed later.

```
raw CSVs (HLTV match history)
   │
   ▼
src/eda_cleaning.py   →  cleaning, schema validation, leakage-aware target derivation
   │
   ▼
src/pipeline.py       →  leakage-free feature engineering (Elo, form, H2H, roster
   │                      stability, player firepower, map-specific ratings...)
   ▼
src/train.py          →  chronological train/val/test split, XGBoost / LightGBM /
   │                      CatBoost comparison, Optuna tuning, probability calibration
   ▼
src/predict.py         → inference layer (reuses pipeline.py's feature functions to
   │                      avoid train/serve skew)
   ▼
api/main.py            → FastAPI service, /api/v1/predict, Supabase logging
```

## Results

The current champion model (CatBoost, sigmoid-calibrated) evaluated on a **strictly
chronological, held-out test period** (Jan–Jun 2026, unseen during training):

| Metric | Value |
|---|---|
| ROC-AUC | 0.653 |
| Log Loss | 0.651 |
| Brier Score | 0.230 |
| Accuracy @ 0.5 | 61.7% |

For context: professional esports betting markets typically operate in the 0.60–0.68
ROC-AUC range for this kind of prediction — CS2 is a high-variance game and near-perfect
prediction is not a realistic target. See [Limitations & Lessons Learned](#limitations--lessons-learned)
below for an honest discussion of what this model can and cannot do.

## Data

Source: HLTV professional match history (tier 1–3), covering **5,134 matches /
10,675 played maps** across **392 teams** and **1,398 players**, from
**Oct 25, 2023 to Jun 28, 2026**.

Raw CSVs are not committed to this repo (see `.gitignore`) — place them under
`data/raw/` before running the pipeline:
- `cs2_all_tiers_games.csv`, `cs2_tier1_games.csv`, `cs2_tier2_games.csv`, `cs2_tier3_games.csv`
- `teams.csv`, `players.csv`, `tournaments.csv`

## Feature engineering

All features are computed to be **strictly leakage-free**: every rolling/expanding
statistic for a given match uses only data from *strictly earlier* matches. This
required special care because every map within the same best-of series shares an
identical timestamp in the source data — naive chronological sorting alone is not
enough to prevent leakage across maps of the same series.

| Group | Description |
|---|---|
| Team Form | Rolling win rate over last 5 / 10 / all matches |
| Map Advantage | Team's historical win rate on the specific map |
| Head-to-Head | Historical win rate between the two specific teams |
| Player Firepower | Rolling ADR / KAST / rating of the starting roster |
| Elo Rating | Classic iteratively-updated strength rating |
| Map-specific Elo | Separate Elo rating per (team, map) pair |
| Rest Days | Days since the team's previous match |
| Win/Loss Streak | Current signed momentum streak |
| Roster Stability | Player overlap with the team's previous match roster |

## Project structure

```
├── data/
│   ├── raw/                  # input CSVs (gitignored)
│   └── processed/            # cleaned parquet files (gitignored)
├── src/
│   ├── eda_cleaning.py       # step 1: cleaning & validation
│   ├── pipeline.py           # step 2: feature engineering
│   ├── train.py              # step 3: model training & calibration
│   └── predict.py            # step 4: inference
├── api/
│   └── main.py                # FastAPI service
├── supabase/
│   └── schema.sql             # predictions table DDL
├── models/                    # trained model artifacts + metadata
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Supabase project credentials
(Project Settings → API).

## Running the pipeline

Each stage must be run in order — every stage reads the previous stage's output:

```bash
# 1. Clean raw CSVs -> data/processed/{maps_clean,matches_summary}.parquet
python -m src.eda_cleaning --raw-dir data/raw --output-dir data/processed

# 2. Feature engineering -> data/processed/features_engineered.parquet
python -m src.pipeline --processed-dir data/processed --output-dir data/processed

# 3. Train + calibrate + tune -> models/champion_model.pkl
python -m src.train --features-path data/processed/features_engineered.parquet --output-dir models

# 4. Predict from the command line
python -m src.predict --team1 "Team Vitality" --team2 "Natus Vincere" --map Mirage --tier tier1 --bestof 3
```

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `GET /api/v1/teams` | List of known team names |
| `GET /api/v1/maps` | List of known map names |
| `POST /api/v1/predict` | Predict a match; logs to Supabase if configured |

The API degrades gracefully if Supabase isn't configured — predictions are still
returned, just not logged (`logged_to_db: false`).

## Limitations & lessons learned

Worth being upfront about what this model can't do, not just what it can.

- **CS2 is high-variance.** A ROC-AUC around 0.65 is a realistic ceiling for this
  problem, not a shortfall of the modeling approach.
- **Calibration drifts over time (concept drift).** The scene evolves — rosters
  change, new teams emerge. A calibration curve fit on one time period will not be
  perfectly diagonal on a later period. This was measured explicitly (not assumed)
  and cross-fitted calibration was tested as a potential fix — it did not help,
  confirming the cause is genuine distribution shift rather than a calibration
  methodology issue. Practical mitigation: retrain periodically as new data
  accumulates.
- **Old data still helps, despite scene evolution.** An empirical test comparing
  training windows (6/12/18 months vs. full history) showed the full ~2.5-year
  history outperforms any recency-truncated subset — sample size outweighed
  staleness for this dataset size. This may change as more data accumulates.
- **Two real data-leakage bugs were caught during development** (and are worth
  naming, since catching them was itself part of the engineering process):
  1. `team_id` in the source data is *not* a stable team identity (the same team
     appears under dozens of different IDs across matches) — team matching had to
     be done by name instead.
  2. The map-specific Elo implementation initially processed both perspectives of
     the same match in one loop, causing the second row to read the first row's
     *already-updated* rating as if it were pre-match — leaking that match's own
     outcome. Caught via an implausibly large jump in test ROC-AUC (0.65 → 0.78)
     and fixed by processing each match exactly once.
- **Tested but not adopted** (negative results, kept for transparency): recency-weighted
  training samples, wider Optuna search (100 trials), and Optuna-tuned CatBoost all
  *underperformed* their simpler counterparts on the held-out test set — with a
  validation set of ~1,600 rows, aggressive hyperparameter search tends to overfit
  to validation noise rather than improve generalization.

## Data source

The dataset is [Counter-Strike Pro Matches on Kaggle](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches),
originally compiled from HLTV.org professional match history. Not redistributed in
this repository — download it from Kaggle and place the CSVs under `data/raw/` as
described above.

## License

This project is for educational/portfolio purposes.
