# CESAR – CentraleSupelec-ESSEC System for Asset Rating

[![CI – Train, Serve & Test](https://github.com/SachaKsk/cesar-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/SachaKsk/cesar-mlops/actions/workflows/ci.yml)

CESAR is a complete MLOps system for training, serving, and testing a **French property valuation model**. It covers the full lifecycle: data ingestion, model training, experiment tracking, API serving, batch prediction, acceptance testing, CI/CD, and deployment.

---

## Features

| Feature | Description |
|---------|-------------|
| **Training pipeline** | Load DVF CSVs → feature engineering → train RandomForest → export versioned artifacts |
| **FastAPI serving** | `POST /estimate/` with JSON, `GET /health` for monitoring |
| **Confidence intervals** | Price range (low/high) using per-tree prediction percentiles |
| **Batch prediction** | CSV in → estimates → CSV out via CLI |
| **Web UI** | Interactive form + Leaflet map of France |
| **Acceptance tests** | Automated test cases run against the live API |
| **CI/CD** | GitHub Actions: train → serve → test on every push/PR |
| **Experiment tracking** | CSV-based run logging with metrics comparison |
| **Docker & Kubernetes** | Dockerfiles, docker-compose, canary & blue-green deployments |

---

## Data Enrichment & Experiment Evaluation

During development, we noticed the raw DVF data had some issues—like outliers (crazy high prices) and uneven distribution across departments. To improve the model, we added simple data cleaning and optional augmentation. We also built a way to track and compare training runs, since it's easy to forget what worked.

### Data Enrichment
- **Cleaning**: Remove extreme prices (>€10M or <€10k) and invalid properties (0 rooms or tiny surfaces). This helped reduce noise.
- **Balancing**: Some departments had way more data than others, so we undersampled to keep things fair (~500 samples per department).
- **Augmentation**: For small datasets, we add synthetic variations (±5-10% noise) to boost training size without collecting more real data.

We tested this in `training/scripts/train_with_enrichment.py`. Results: Cleaning gave us ~3% better R², augmentation added another 1%. Not huge, but noticeable for property prices.

### Experiment Tracking
We use a simple CSV file (`experiment_runs/runs.csv`) to log each training run—timestamp, model version, metrics (R², MAE, RMSE), and notes. No fancy tools, just pandas and a CLI to compare runs.

Commands:
```bash
# Train with cleaning
python -m training.scripts.train_with_enrichment --clean

# View all runs
cesar experiment-analysis list

# Compare two models
cesar experiment-analysis compare v1_baseline v2_cleaned

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  DVF Data    │────▶│  Training        │────▶│  Artifact Store │
│  (CSVs)      │     │  (RandomForest)  │     │  (.joblib+.json)│
└─────────────┘     └──────────────────┘     └────────┬────────┘
                            │                          │
                    ┌───────▼────────┐                 │
                    │ Experiment Log │                  │
                    │ (CSV tracker)  │                  │
                    └────────────────┘                  │
                                                       │
                ┌──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────┐
│      FastAPI Server      │
│  POST /estimate/         │
│  GET  /health            │
└──────┬───────────────────┘
       │
       ├──────────────┐
       ▼              ▼
┌────────────┐  ┌───────────┐
│  Web UI    │  │   CLI     │
│  (Vite +   │  │  (Typer)  │
│  Leaflet)  │  │  batch /  │
│            │  │  single   │
└────────────┘  └───────────┘

       ▲
       │
┌──────┴─────────────┐     ┌────────────────────┐
│ Acceptance Tests   │     │ GitHub Actions CI   │
│ (auto validation)  │────▶│ train → serve → test│
└────────────────────┘     └────────────────────┘
```

## Goal

Estimate the value of a property (`valeur_fonciere`) from a small set of features:
- **Surface** (m²)
- **Number of rooms**
- **Department code**
- **Property type** (Appartement, Maison, Dépendance, Local industriel)

---

## Source data

- **Primary:** [DVF – Données de Valeurs Foncières](https://app.dvf.etalab.gouv.fr/) (also at [explore.data.gouv.fr](https://explore.data.gouv.fr/fr/immobilier))


## What we implemented

Our team extended the base CESAR project with three features:

### 1. CI/CD Pipeline (GitHub Actions)
Automated workflow that runs on every push and pull request:
- Installs dependencies
- Trains the model from `data/` CSVs
- Starts the API server
- Runs acceptance tests against the live API

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### 2. Confidence Intervals
The API now returns a price **range** alongside the point estimate:
- Uses per-tree predictions from the RandomForest ensemble
- Computes 5th and 95th percentiles across all 50 trees
- Returns `value_low_eur` and `value_high_eur` in the response
- Displayed visually in the web UI

### 3. Data Enrichment & Experiment Comparison
- Added additional DVF data extracts for broader geographic coverage
- Implemented train/test split with evaluation metrics (MAE, RMSE, R²)
- Logged and compared multiple training runs via the experiment tracker
- Results show impact of data volume and diversity on model performance

---

## Team

| Member | Contribution |
|--------|-------------|
| [SachaKsk](https://github.com/SachaKsk) | CI/CD pipeline, README, project coordination |
| [haiiiio](https://github.com/haiiiio) | Confidence intervals (inference + UI) |
| [qrebut](https://github.com/qrebut) | Data enrichment & experiment comparison |
