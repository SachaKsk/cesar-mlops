# CESAR – CentraleSupelec-ESSEC System for Asset Rating

[![CI – Train, Serve & Test](https://github.com/SachaKsk/cesar-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/SachaKsk/cesar-mlops/actions/workflows/ci.yml)

CESAR is a complete MLOps system for training, serving, and testing a **French property valuation model**. It covers the full lifecycle: data ingestion, model training, experiment tracking, API serving, batch prediction, acceptance testing, CI/CD, and deployment.

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
┌─── Docker Compose ───────────────────────────────────┐
│                                                      │
│  ┌────────────────────┐    ┌──────────────────────┐  │
│  │  nginx (:8080)     │    │  FastAPI Server       │  │
│  │                    │    │  POST /estimate/      │  │
│  │  /     → static UI ├───▶  GET  /health         │  │
│  │  /api/ → proxy     │    │  :8000                │  │
│  └────────────────────┘    └──────────────────────┘  │
│                              ▲ volumes: artifacts (ro)│
└──────────────────────────────│────────────────────────┘
                               │
       ┌───────────────────────┤
       │                       │
┌──────┴──────┐    ┌───────────┴────────┐
│   CLI       │    │  Acceptance Tests  │
│  (Typer)    │    │  (auto validation) │
│  batch /    │    └─────────┬──────────┘
│  single     │              │
└─────────────┘    ┌─────────▼──────────┐
                   │ GitHub Actions CI   │
                   │ train → serve → test│
                   └────────────────────┘
```

---

## What we implemented

Our team extended the base CESAR skeleton with three features: a **CI/CD pipeline**, **confidence intervals** on every prediction, and **data enrichment with experiment tracking**. Each is described in detail below.

---

## Containerized Deployment

A valuation model sitting on someone's laptop isn't very useful to anyone. For CESAR to actually work as a product (or even just a reliable demo), the API and UI need to run anywhere, not just on the machine where you trained the model. That's why we containerized everything.

### Docker Compose Setup

The local stack runs two containers orchestrated by `docker-compose.yml`:

- **API container** (`Dockerfile.api`): Built from `python:3.12-slim`, installs only what's needed (FastAPI, uvicorn, scikit-learn, joblib, numpy), copies in `prediction_contract/` and `runtime/`, and starts uvicorn on port 8000. The trained model files live outside the container in `artifact_storage/` and are mounted read-only at `/artifacts`. This means we can swap models without rebuilding the image, which is exactly what you'd want in a real deployment where retraining happens regularly.

- **UI container** (`Dockerfile.ui`): Multi-stage build. First stage uses `node:20-alpine` to `npm install && npm run build` the Vite app. Second stage copies the compiled static files into an `nginx:alpine` image. nginx serves the UI on port 80 (mapped to 8080 on the host) and also acts as a reverse proxy: requests to `/api/` get forwarded to the API container at `http://api:8000/`. This way the browser only talks to one origin and we avoid CORS headaches entirely.

- **Health-gated startup**: The UI container uses `depends_on` with `condition: service_healthy`, so it only starts once the API's healthcheck (`GET /health`, every 10s) passes. No more "UI is up but the API isn't ready yet" race conditions.

One `docker compose up --build` from the repo root and both services are live. From a business perspective, this is the minimum bar for shipping: anyone on the team (or a client) can spin up the full system without installing Python, Node, or any dependencies locally.

---

## Confidence Intervals: How It Works

The base system originally returned a single number: the average prediction across all trees in the RandomForest. That's fine for a quick estimate, but for property valuation it's not enough. A buyer or a bank wants to know how confident we are. "Is it 300k plus or minus 5k, or plus or minus 80k?" Those are very different situations.

### Inference Layer

The model is a `RandomForestRegressor` with 50 trees (`n_estimators=50`, `max_depth=10`). Each tree sees a different bootstrap sample of the training data and learns slightly different patterns. At prediction time, instead of just averaging their outputs, we collect all 50 individual predictions and compute:

- **Point estimate**: the standard mean (what `model.predict()` returns)
- **Lower bound** (`value_low_eur`): 5th percentile across the 50 tree predictions
- **Upper bound** (`value_high_eur`): 95th percentile across the 50 tree predictions

This happens in `runtime/inference/estimate_from_artifact.py`. There's also a fallback: if someone loads a model that doesn't have `estimators_` (i.e. not a tree ensemble), the response just returns the point estimate without bounds. No crash, just graceful degradation.

### API Integration

The response schema (`prediction_contract/response_schema.py`) already had `value_low_eur` and `value_high_eur` as optional fields with `None` defaults. This was a deliberate design choice: adding the interval fields doesn't break any existing client that only reads `estimated_value_eur`. The FastAPI endpoint in `runtime/prediction_api/app.py` just returns whatever `estimate_from_model()` produces, so the intervals flow through automatically with zero changes to the API layer itself.

### UI Display

The web UI shows the confidence interval as plain text alongside the point estimate. When the API returns bounds, `display_estimate.ts` renders both the estimated value and the range ("Range: 250,000 – 350,000 €") directly in the page. If the API doesn't return bounds, only the point estimate is shown—no crash, just less information.

The UI also includes a Leaflet map of France (OpenStreetMap tiles), which currently serves as a geographic reference. The map and the estimate display are independent components, so either can be extended without touching the other.

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
```

---

## What More Could Be Done

CESAR covers the core MLOps loop, but there's room to grow if this were heading toward a real product:

- **MCP model serving** — Wrap the inference function in a Model Context Protocol server so AI assistants can call estimates directly. The inference code is already a single function call, so the wrapper would be thin. This enables conversational interfaces ("What's a 3-room apartment in the 15th worth?") powered by the same model.
- **Anomaly detection** — A wide confidence interval means the model is unsure. Surfacing that as a warning ("high uncertainty—verify with comparable sales") or flagging outlier price-per-m² values would build trust with professional users who need to justify valuations.
- **Observability** — Request logging (timestamp, inputs, output, latency) would enable monitoring dashboards and data-drift detection. Combined with the experiment tracker, this creates a retrain-when-needed feedback loop.
- **Richer visualizations** — The confidence interval is currently plain text. A visual gauge, comparable-sales charts, or a GeoJSON-colored department map overlaid on the existing Leaflet map would communicate uncertainty more intuitively.

---

## Team

| Member | Contribution |
|--------|-------------|
| [SachaKsk](https://github.com/SachaKsk) | CI/CD pipeline, README, project coordination |
| [haiiiio](https://github.com/haiiiio) | Confidence intervals (inference + UI), Docker & deployment |
| [qrebut](https://github.com/qrebut) | Data enrichment & experiment comparison |
