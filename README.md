# CS2 Match Prediction Engine

This project predicts which team wins a given map in a professional Counter-Strike 2
match. It started as an attempt to go beyond a single Jupyter notebook and actually
build the thing the way a real ML system would be put together — separate stages for
cleaning the data, engineering features, training a model, and serving predictions
through an API, instead of one long script.

**Live demo:** [fragpredict.duckdns.org](https://fragpredict.duckdns.org): pick two teams and a
map to get a win probability.

> 🇹🇷 Türkçe README için [README.tr.md](README.tr.md) dosyasına bakın.

## Overview

Give it two teams and a map — say, Team Vitality vs Natus Vincere on Mirage — and it
returns a calibrated win probability for each side. The model is a logistic regression
(it matched or beat XGBoost, LightGBM and CatBoost on a held-out validation period)
trained on about 2.5 years of professional match history, served through a
FastAPI endpoint, with predictions logged to a Postgres database (Supabase) so they
can be reviewed later.

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
src/train.py          →  chronological train/val/test split, logistic regression /
   │                      XGBoost / LightGBM / CatBoost comparison (champion picked on
   │                      validation), Optuna tuning, probability calibration
   ▼
src/predict.py         → inference layer (reuses pipeline.py's feature functions to
   │                      avoid train/serve skew)
   ▼
api/main.py            → FastAPI service, /api/v1/predict, Supabase logging
```

## Results

The current champion model (logistic regression, sigmoid-calibrated) evaluated on a
**strictly chronological, held-out test period** (May–Oct 2026, unseen during training).
The champion is picked on the validation period, so the test period played no part in
choosing it:

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.649 | 0.624 – 0.674 |
| Log Loss | 0.658 | 0.646 – 0.671 |
| Brier Score | 0.233 | 0.227 – 0.239 |
| Accuracy @ 0.5 | 60.2% | 58.0% – 62.6% |

Confidence intervals come from a bootstrap over the 922 test matches (maps of the same
match are resampled together).

Predictions are **order-invariant**: "A vs B" and "B vs A" always get exactly
complementary probabilities (see [team order bias](#limitations--lessons-learned)).

For context: CS2 is a high-variance game and near-perfect prediction is not a realistic
target. See [Limitations & Lessons Learned](#limitations--lessons-learned)
below for an honest discussion of what this model can and cannot do.

## Data

Source: HLTV professional match history (tier 1–3), covering **5,903 matches /
12,235 played maps** across **483 teams** and **1,566 players**, from
**Oct 25, 2023 to Oct 24, 2026**.

The raw CSVs are included under `data/raw/` (CC BY-SA 4.0, see [Data source](#data-source)).
To refresh the data, download the latest version from Kaggle and replace these files:
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
│   ├── raw/                  # input CSVs from Kaggle
│   └── processed/            # cleaned parquet files (the API reads two of them)
├── src/
│   ├── eda_cleaning.py       # step 1: cleaning & validation
│   ├── pipeline.py           # step 2: feature engineering
│   ├── train.py              # step 3: model training & calibration
│   ├── predict.py            # step 4: inference
│   └── symmetry.py           # team-order invariance (shared by train & predict)
├── api/
│   └── main.py                # FastAPI service
├── web/
│   └── index.html             # single-page UI served at /
├── deploy/
│   ├── huggingface/
│   │   └── README.md          # Hugging Face Space config (front matter)
│   └── supabase/
│       └── predictions_rls.sql # insert-only access to the predictions table
├── models/                    # trained model artifacts + metadata
├── Dockerfile                 # API + UI image (serve-only dependencies)
├── requirements.txt           # full: pipeline + training + API
├── requirements-serve.txt     # API only
└── env.example
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `env.example` to `.env` and fill in your Supabase project URL and publishable key
(Project Settings → API). Logging is optional; see [Prediction logging](#prediction-logging).

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

Web UI at `http://localhost:8000/`, interactive docs at `http://localhost:8000/docs`.

| Endpoint | Description |
|---|---|
| `GET /` | Web UI (`web/index.html`) |
| `GET /health` | Service health check |
| `GET /api/v1/teams` | List of known team names |
| `GET /api/v1/maps` | Active map pool (maps played in the last 60 days) |
| `GET /api/v1/model` | Model name, data range and held-out test metrics |
| `POST /api/v1/predict` | Predict a match; logs to Supabase if configured |

The API degrades gracefully if Supabase isn't configured — predictions are still
returned, just not logged (`logged_to_db: false`). Unknown team names and tiers are
rejected with `400` (the CLI still allows cold-start predictions for unseen teams).

## Deployment

The `Dockerfile` builds a serve-only image: it copies just the champion model and the
two parquet files the predictor reads (`maps_clean`, `matches_summary`). Startup takes
~30 s while the snapshot tables are built; memory use is ~300 MB.

### Live demo (VPS)

The [live demo](https://fragpredict.duckdns.org) runs on a small VPS (2 vCPU, 4 GB RAM)
as a Docker Compose service. The container publishes no ports. A Caddy reverse proxy on
the same Docker network forwards the domain to it and gets the HTTPS certificate from
Let's Encrypt automatically.

```yaml
# docker-compose.yml, next to the Dockerfile on the server
services:
  fragpredict:
    build: .
    restart: unless-stopped
    env_file: .env
    networks:
      - proxy

networks:
  proxy:
    external: true
    name: <caddy-network>   # the Docker network Caddy runs on
```

```
# Caddyfile
fragpredict.duckdns.org {
	reverse_proxy fragpredict:7860
}
```

To deploy a new version (for example after retraining), push it to `main` and run this
on the server:

```bash
cd /opt/fragpredict && git pull && docker compose up -d --build
```

### Hugging Face Spaces

The same image also runs as a Docker Space:

```bash
pip install -U huggingface_hub
hf auth login
hf repos create <user>/fragpredict --type space --sdk docker

hf upload <user>/fragpredict . . --repo-type space \
  --include "Dockerfile" --include "requirements-serve.txt" \
  --include "api/*.py" --include "src/*.py" --include "web/*" \
  --include "models/champion_model.pkl" --include "models/model_metadata.json" \
  --include "data/processed/maps_clean.parquet" --include "data/processed/matches_summary.parquet"
hf upload <user>/fragpredict deploy/huggingface/README.md README.md --repo-type space
```

After retraining, rerun the first `hf upload` to redeploy.

### Prediction logging

Prediction logging is optional. To turn it on, first run
`deploy/supabase/predictions_rls.sql` in the Supabase SQL Editor. It gives the
publishable key (`sb_publishable_…`) insert-only access to the `predictions` table: no
reads, updates or deletes. Then set `SUPABASE_URL` and `SUPABASE_KEY` (the publishable
key): on a server, in the `.env` file next to `docker-compose.yml`; on Hugging Face, as
**Secrets** in the Space settings. Don't use Space Variables, because those are
public. Even if the key leaked, all it could do is insert rows, which the public API
already does.

Two safeguards in the API protect the table, and neither uses IP addresses:
- **Write budget:** at most 30 rows a minute and 3,000 a day.
- **Duplicate check:** the same match and map, in either team order, isn't logged again
  within 10 minutes.

When a write is skipped, the prediction is still returned; only the log entry is
dropped.

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
- **Team order bias (found and fixed).** The team1/team2 order in the source data
  isn't random — team1 wins ~53% of maps and usually has the higher Elo. A model
  trained only in that order learned a ~5-point bonus for "whoever is listed first",
  which has no basis in the data: once the features are accounted for, the position
  effect is ≈0. Since API/UI users choose the order themselves, the same match got
  different predictions depending on order (the favourite flipped in ~15% of test
  matches). The fix (`src/symmetry.py`): each split is augmented with mirrored rows
  (teams swapped, label flipped) *after* splitting, and every prediction averages both
  orders, so P(A beats B) + P(B beats A) = 1 exactly. The model's own inconsistency
  before averaging dropped from 6.1 to 1.9 points, and test log loss improved from
  0.6614 (old model, random user order) to 0.6576. Fixing the order canonically
  instead (alphabetical, or higher Elo first) also removes the order dependence but
  wasn't more accurate.
- **The choice between models was noise, so the simplest one won.** The champion used
  to be picked by test-set log loss and then scored on that same test set — a mild
  selection bias (on average it flattered the reported log loss by 0.0008, at most
  0.0025). It's now picked on the validation period. Re-running the comparison with
  10 seeds showed the gradient-boosted models trading places from seed to seed
  (XGBoost won validation in 7 runs, CatBoost in 3), with every pairwise difference
  inside its bootstrap CI. Logistic regression had the best validation log loss in
  all 10 runs, is deterministic and is as good as the best tree model on test, so it
  is the champion: with the current features, gradient boosting finds no extra
  non-linear signal. The selection rule stays model-agnostic, so a tree model takes
  over if future features give it something to find.
  Side effect: logistic regression uses only the numeric features, so tier and
  best-of don't change a prediction (under order invariance they could only matter
  through interactions). The map still matters through map-specific Elo, win rate and
  experience.
- **Tested but not adopted** (negative results, kept for transparency): recency-weighted
  training samples, wider Optuna search (100 trials), and Optuna-tuned CatBoost all
  *underperformed* their simpler counterparts on the held-out test set — with a
  validation set of ~1,600 rows, aggressive hyperparameter search tends to overfit
  to validation noise rather than improve generalization.

## Data source

The dataset is [Counter-Strike Pro Matches on Kaggle](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches)
by ektarr, originally compiled from HLTV.org professional match history and licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). A copy of the CSVs
is included under `data/raw/`; the Kaggle page has the latest version.

This repository and the live demo also ship processed files derived from it
(`maps_clean.parquet`, `matches_summary.parquet` and `features_engineered.parquet`:
cleaned and restructured per match and map). Under the ShareAlike term, those files are
licensed CC BY-SA 4.0 as well.

## License

This project is for educational/portfolio purposes.
