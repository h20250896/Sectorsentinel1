from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.ml.data_loader import (
    ALERTS_PATH,
    LOCAL_EXPLANATIONS_PATH,
    load_json,
    load_scored_panel,
    load_sector_panel,
    sorted_quarters,
)
from app.ml.explainer import build_local_explanations
from app.ml.inference import load_model_bundle, score_features, simulate_scenario
from app.schemas.score import DashboardSummary, HeatmapResponse, ScenarioRequest, ScenarioResponse
from app.services.score_service import build_dashboard_payload, build_heatmap
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/scores", tags=["scores"])
scenario_router = APIRouter(tags=["scenario"])


@router.get("/heatmap", response_model=HeatmapResponse, summary="Get sector heatmap data")
async def get_heatmap(quarter: str | None = Query(None)) -> dict:
    cache_key = f"scores:heatmap:{quarter or 'latest12'}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    scored_panel = load_scored_panel()
    available_quarters = sorted_quarters(scored_panel["quarter"])
    if quarter and quarter in available_quarters:
        quarter_index = available_quarters.index(quarter)
        selected_quarters = available_quarters[max(0, quarter_index - 11) : quarter_index + 1]
    else:
        selected_quarters = available_quarters[-12:]
    payload = build_heatmap(scored_panel, selected_quarters)
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload


@router.get("/dashboard", response_model=DashboardSummary, summary="Get one-call dashboard payload")
async def get_dashboard(quarter: str | None = Query(None)) -> dict:
    cache_key = f"scores:dashboard:{quarter or 'latest'}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    scored_panel = load_scored_panel()
    alerts = load_json(ALERTS_PATH)
    explanations = load_json(LOCAL_EXPLANATIONS_PATH)
    payload = build_dashboard_payload(scored_panel, alerts, explanations, quarter)
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload


@scenario_router.post("/scenario", response_model=ScenarioResponse, summary="Run a no-retrain stress scenario")
async def run_scenario(request: ScenarioRequest) -> dict:
    raw_panel = load_sector_panel()
    scored_panel = load_scored_panel()
    bundle = load_model_bundle()
    engineered, target_quarter = simulate_scenario(bundle, raw_panel, request.sector_id, request.overrides, request.quarter)
    feature_row = engineered.loc[[(request.sector_id, target_quarter)], :]
    probabilities, scores, labels, breakdown = score_features(bundle, feature_row)
    explanations, _ = build_local_explanations(
        features=feature_row[bundle["feature_columns"]],
        metadata=engineered[["sector_name", "sector_color"]],
        random_forest_model=bundle["base_models"]["random_forest"],
        probabilities=probabilities,
        stress_scores=scores,
        stress_labels=labels,
    )
    baseline_frame = scored_panel.loc[
        (scored_panel["sector_id"] == request.sector_id) & (scored_panel["quarter"] == target_quarter)
    ]
    if baseline_frame.empty:
        raise HTTPException(status_code=404, detail="Baseline scenario not found")
    baseline_score = int(baseline_frame.iloc[0]["stress_score"])
    return {
        "sector": request.sector_id,
        "quarter": target_quarter,
        "baseline_score": baseline_score,
        "stress_score": int(scores[0]),
        "stress_probability": round(float(probabilities[0]), 4),
        "stress_label": labels[0],
        "delta": int(scores[0]) - baseline_score,
        "model_breakdown": {key: round(float(value[0]), 4) for key, value in breakdown.items()},
        "shap_attributions": explanations[0]["shap_attributions"],
    }
