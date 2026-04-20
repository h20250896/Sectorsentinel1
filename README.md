[README.md](https://github.com/user-attachments/files/26893610/README.md)
# SectorSentinel

SectorSentinel is a full-stack early warning dashboard for Indian sector financial stress. It combines synthetic credit, market, macro, and alternative indicators with an ensemble ML stack, SHAP explainability, alerting, and a premium React dashboard experience.

## Monorepo layout

```
sectorsentinel/
  backend/   FastAPI, SQLAlchemy, Alembic, ML pipeline, synthetic data generator
  frontend/  React, Vite, TypeScript, Tailwind, React Query, Zustand, Recharts, D3
```

## What is implemented

- Synthetic quarterly panel generator covering 2018Q1 to 2025Q2 for 11 sectors
- Stress labelling logic with 1-2 quarter forward warning targets
- Feature engineering with lags, YoY deltas, z-scores, momentum, and contagion proxy features
- Ensemble training pipeline with stacking, calibration, SHAP artifacts, and saved model bundle
- Contagion graph generation, alert generation, regulator-brief payloads, and scored panel artifacts
- FastAPI routes for sectors, scores, indicators, contagion, model metadata, alerts, scenario simulation, and reports
- Premium dark-mode dashboard with overview, sector deep dive, contagion map, model insights, simulator, alerts, and regulator brief pages
- Docker Compose stack with Postgres, Redis, backend-init, backend API, and frontend

## Quick start

1. Copy `.env.example` to `.env`.
2. From the repo root run:

```bash
docker compose up --build
```

3. Open:

- Frontend: `http://localhost:5173`
- Backend docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Backend workflow

The backend-init service runs:

```bash
python seed_data/generate_synthetic.py
python -m app.ml.model_trainer
```

This populates:

- `backend/data/sector_panel.csv`
- `backend/data/feature_matrix.csv`
- `backend/models/ensemble_v1.pkl`
- `backend/artifacts/scored_panel.csv`
- `backend/artifacts/model_performance.json`
- `backend/artifacts/local_explanations.json`
- `backend/artifacts/global_feature_importance.json`
- `backend/artifacts/contagion_network.json`
- `backend/artifacts/alerts.json`
- `backend/artifacts/regulator_briefs.json`

## Key API routes

- `GET /health`
- `GET /api/v1/sectors`
- `GET /api/v1/sectors/{sector_id}/score`
- `GET /api/v1/sectors/{sector_id}/history`
- `GET /api/v1/scores/dashboard`
- `GET /api/v1/scores/heatmap`
- `GET /api/v1/indicators/{sector_id}`
- `GET /api/v1/contagion/network`
- `GET /api/v1/model/performance`
- `GET /api/v1/model/feature-importance`
- `GET /api/v1/alerts`
- `POST /api/v1/scenario`
- `GET /api/v1/reports/{sector_id}/brief`

## Frontend notes

- Typography uses Playfair Display, DM Sans, and DM Mono.
- The overview page provides KPI cards, sector cards, heatmap, and alerts.
- Sector detail includes timeline, indicator overlays, SHAP waterfall, contagion lists, and recommendations.
- Contagion map uses a force-directed D3 graph.
- Scenario simulator uses live no-retrain scoring against the latest model bundle.
- Regulator brief includes a print stylesheet for PDF export.

## Local verification performed

- Synthetic data generation succeeded with 330 rows and all requested feature columns.
- Stress-frequency diagnostics show Infrastructure as persistently stressed and IT as only briefly stressed.
- Feature engineering produced a 330 x 154 matrix with zero NaN leakage.
- Model training completed locally and persisted the model bundle and artifact files.
- Latest scored panel now surfaces meaningful RED, AMBER, and GREEN sectors for the dashboard.

## Notes

- The local workspace Python environment did not include FastAPI and some runtime dependencies, so backend runtime verification in this workspace was limited to artifact generation plus Python compile checks. The Dockerfiles install the full dependency set for end-to-end startup.
- The frontend source is fully scaffolded and wired, but a local production build was not executed here because npm dependencies were not installed in the workspace.
