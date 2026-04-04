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

---

## Goal

Estimate the value of a property (`valeur_fonciere`) from a small set of features:
- **Surface** (m²)
- **Number of rooms**
- **Department code**
- **Property type** (Appartement, Maison, Dépendance, Local industriel)

---

## Source data

- **Primary:** [DVF – Données de Valeurs Foncières](https://app.dvf.etalab.gouv.fr/) (also at [explore.data.gouv.fr](https://explore.data.gouv.fr/fr/immobilier))
- Additional data from other sources or synthetic data can be used.

---

## How to run

### Prerequisites

- Python 3.11+
- Node 20+ (for the UI)

### 1. Install

Dependencies are declared in `pyproject.toml` (the modern replacement for `requirements.txt`). Use `pip install -e .` so the package is installed in editable mode: code changes are picked up without reinstalling.

```bash
pip install -e .
cd runtime/rating_ui && npm ci
```

### 2. Train

**Simple way (recommended):** Put your CSV file(s) in the `data/` folder. Each CSV must have the same columns and at least: `surface_reelle_bati`, `nombre_pieces_principales`, `code_departement`, `type_local`, `valeur_fonciere`. Then:

```bash
python -m training.scripts.train_from_minimal_csv
```

The script loads every `*.csv` in `data/`, checks that all files have the same columns (and the required ones), combines the rows, and trains one model. If a file has missing or different columns, it raises a clear error.

**Single file (advanced):** To train from one CSV at a custom path, use `train_from_csv_and_export(csv_path, artifact_dir, ...)` from `training.asset_rating_model.train_and_export`.

This writes `artifact_storage/model_<version>.joblib` and `artifact_storage/contract_<version>.json`. Symlink or set `CESAR_MODEL_PATH` and `CESAR_CONTRACT_PATH` to these files.

### 3. Batch CLI

```bash
export CESAR_MODEL_PATH=artifact_storage/model_20250101120000.joblib
export CESAR_CONTRACT_PATH=artifact_storage/contract_20250101120000.json
cesar batch run --input input.csv --output output.csv
```

### 4. Single-record CLI

```bash
cesar predict-one run --surface 50 --pieces 3 --departement 75 --type Appartement
# From JSON file: cesar predict-one run --json one_record.json
# JSON output:  cesar predict-one run --surface 50 --pieces 3 --departement 75 --type Appartement --json-out
```

Use `--model` / `--contract` or set `CESAR_MODEL_PATH` and `CESAR_CONTRACT_PATH`. Valid `--type` values: `Appartement`, `Maison`, `Dépendance`, `Local industriel. commercial ou assimilé`.

### 5. API

```bash
uvicorn runtime.prediction_api.app:app --reload --host 0.0.0.0 --port 8000
```

Set `CESAR_MODEL_PATH` and `CESAR_CONTRACT_PATH` in the environment.

### 6. UI

```bash
cd runtime/rating_ui && npm run dev
```

Open the dev URL (e.g. http://localhost:5173). Set `window.CESAR_API_BASE = 'http://localhost:8000'` in the browser console if the API is on another origin.

### 7. Acceptance tests

Test cases are defined in Python in `model_acceptance_tests/test_cases.py`. Edit that file to add or change cases. With the API running:

```bash
export CESAR_API_URL=http://localhost:8000
cesar acceptance-tests run
```

### 8. Version comparison

Use `comparison.api_version_comparison.run_comparison` with a `CompareConfig` (two base URLs and a list of inputs). Ideas: implement diff of `estimated_value_eur`, regression criteria, or a report.

### 9. Experiment tracking (simple)

When you train often (different data, parameters, or ideas), it’s easy to forget what you did and which model version was best. **Experiment tracking** means: write down each run (when, what you used, what you got) in one place so you can compare later.

We provide a minimal version: no extra service, no database. One CSV file (e.g. `experiment_runs/runs.csv`) stores one row per training run. You can open it in Excel or read it in Python.

**What to record per run**

- **When** you ran (timestamp).
- **Which model version** was produced (e.g. the version string you wrote to `artifact_storage`).
- **What you used**: e.g. number of training rows, path to the CSV, or a short note (“first DVF subset”, “added feature X”).
- **What you got** (optional): e.g. a metric like mean absolute error on a fixed test set.

**How to use it**

After you train and export a model, call the logger once:

```python
from pathlib import Path
from training.asset_rating_model.train_and_export import train_from_csv_and_export
from training.experiment_log import log_run

csv_path = Path("path/to/dvf_subset.csv")
artifact_dir = Path("artifact_storage")
model_path, contract_path = train_from_csv_and_export(csv_path, artifact_dir)
# Version is in the filename, e.g. model_20250101120000.joblib
version = model_path.stem.replace("model_", "")

log_run(version, train_rows=1000, notes="DVF subset 75")  # use len(df) if you have the dataframe
```

If you compute a metric (e.g. test MAE), pass it as `metrics={"mae": 12000}`. To see past runs: `list_runs()` returns a list of dicts; or open `experiment_runs/runs.csv` in Excel.

**Why this helps**

- You can see which run used which data or settings.
- You can compare metrics across runs and pick the best model version to deploy.
- Later you can switch to a real experiment tracker (e.g. MLflow) that adds plots, parameters, and artifacts; the idea is the same: record what you did and what you got.

---

## Ideas to implement

Organized by **difficulty / scope** (from small to larger).

1. **Easier**
   - Add more acceptance test cases in `model_acceptance_tests/test_cases.py`; optional property-based tests.
   - UI: improve map (e.g. click on department to set `code_departement`); display value range if the API returns `value_low_eur` / `value_high_eur`.
   - Confidence intervals: in `estimate_from_artifact` and the API response, add optional `value_low_eur` / `value_high_eur` (e.g. quantile regression).

2. **Medium**
   - Complete the API version comparison: diff responses, define regression criteria, output a small report (HTML or JSON).
   - CI/CD: e.g. GitHub Actions to build Docker images and run acceptance tests.
   - Deployment: add Kubernetes readiness/liveness probes, HPA, ingress, or GitOps.

3. **Larger**
   - Authentication/authorization on the API.
   - Model registry (e.g. MLflow) instead of a single artifact directory.
   - Database or logging for requests / A/B tests.

---

## Repository layout

Code is grouped by MLOps phase so students can navigate easily.

- **Root:** `prediction_contract/` – shared request/response and on-disk contract (feature names, version). `cli/` – all CLI commands (typer); entrypoint: `cesar`. Logic for each command lives in the matching folder (e.g. `runtime/`, `model_acceptance_tests/`).
- **runtime/** – serving: `prediction_api/` (FastAPI), `batch_prediction/` (read CSV, run estimates, write CSV), `inference/` (load artifact, estimate from model), `rating_ui/` (form, map of France).
- **training/** – `asset_rating_model/` (train and export), `scripts/` (e.g. minimal CSV demo), `experiment_log.py` (simple run logging to a CSV).
- **model_acceptance_tests/** – hardcoded test cases in Python, runner; CLI is in `cli/acceptance_tests.py`.
- **comparison/** – `api_version_comparison/` (config and stub to compare two APIs).
- **deployment/** – Dockerfiles, docker-compose, Kubernetes manifests.
- **artifact_storage/** – directory for model and contract (versioned names); created on first train.

---

## Project ideas for students

Here are concrete ways to extend CESAR. All are doable in a short course; pick one or combine a few depending on your time and interests.

**More and better data**
- Add extra CSVs to `data/` (e.g. other DVF extracts, open data from data.gouv.fr) and retrain. Compare experiment-log metrics across runs to see if more data helps.
- Enrich the training CSV with one new column (e.g. `code_postal` or distance to a city center from open data), add it to the contract and model, and measure the impact. Good way to see the full pipeline: data → training → contract → API.

**Expose the model via MCP (Model Context Protocol)**
- Build a small MCP server that exposes a “get property estimate” tool. Other apps (or an AI assistant in Cursor/Claude) can then call your model without touching the API directly. Great for “my model as a building block” and learning how tools are exposed to LLMs.
- Start from the existing inference code: one function that takes (surface, rooms, department, type) and returns the estimate. Wrap it in an MCP server that declares one tool and calls that function. The MCP docs and examples are minimal; you mainly need to return JSON in the right shape.

**Smarter UI**
- Make the map clickable: click a department on the Leaflet map to set `code_departement` in the form. You only need to add a GeoJSON layer (e.g. French departments) and a click handler that updates the form.
- Show the value range when the API returns `value_low_eur` / `value_high_eur` (if you add confidence intervals to the model). The UI already has a placeholder for “Range”.

**Anomaly detection (“why so cheap / so expensive?”)**
- Add a feature that flags surprising estimates: “This estimate looks unusually high (or low) for the given surface, location, and type.” Users get a nudge to double-check or add more context. There are several ways to implement it; the simplest use the model’s own uncertainty.
- **Option A – Confidence ranges:** If the model outputs a range (`value_low_eur`, `value_high_eur`), you can treat a very wide range as “uncertain” and show a warning, or compare the point estimate to the range (e.g. if the user’s prior belief is outside the range, flag it). Implementing ranges may require changing the model (e.g. quantile regression or prediction intervals) and the contract; the response schema already has optional `value_low_eur` / `value_high_eur`.
- **Option B – Simple rules:** Without changing the model, you can add heuristics (e.g. “price per m² far above/below the department average” using training or external stats) and return a flag in the API (e.g. `anomaly_warning: "high_for_department"`). Good first step before touching the model.
- **Option C – Second model or score:** Train a small classifier or scorer that predicts “is this estimate suspicious?” from the same inputs plus the model’s point estimate (e.g. residual-style or comparison to a baseline). The API then returns both the estimate and a binary or score for “possible anomaly”.

Pick one option and plug it into the API and (optionally) the UI; even a simple rule or a wide-range check makes the system more interpretable.

**API and developer experience**
- Add a `/health` that checks that the model and contract load (e.g. read one file). Helps deployment and monitoring.
- Add one extra endpoint, e.g. GET `/model_info`, that returns the contract version and feature names. Useful for scripts or the UI to show “which model is running”.

**Deployment and ops**
- Run the API and UI with `docker compose`, then try the canary or blue-green Kubernetes manifests. Switch traffic from “blue” to “green” once and watch the rollout. No need to design a full pipeline; the goal is to feel what a zero-downtime switch is.
- Add a simple GitHub Action that runs acceptance tests on every push. One YAML file that installs deps, starts the API (with a test model), and runs `cesar acceptance-tests run`.

**Combining ideas**
- “Data + experiment log”: train on two different data mixes, log each run with `experiment_log`, then compare which version the acceptance tests prefer.
- “MCP + UI”: build the MCP server, then use it from a small script or from Cursor’s MCP client to get estimates; keep the existing UI for demos.
- “API + deployment”: add `/health` and `/model_info`, then add readiness/liveness probes in the Kubernetes deployment so the cluster only sends traffic when the model is loaded.

Pick something that excites you and fits your timeline; even one of these will deepen your understanding of the stack.

---

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
