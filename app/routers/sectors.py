from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.ml.data_loader import LOCAL_EXPLANATIONS_PATH, load_json, load_scored_panel, sorted_quarters
from app.schemas.score import StressScoreResponse
from app.services.score_service import build_sector_history, build_sector_summaries
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/sectors", tags=["sectors"])


@router.get("", summary="List sectors with latest stress scores")
async def list_sectors() -> list[dict]:
    cache_key = "sectors:list"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    scored_panel = load_scored_panel()
    explanations = load_json(LOCAL_EXPLANATIONS_PATH)
    payload = build_sector_summaries(scored_panel, explanations)
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload


@router.get("/{sector_id}/score", response_model=StressScoreResponse, summary="Get sector score for a quarter")
async def get_sector_score(sector_id: str, quarter: str | None = None) -> dict:
    cache_key = f"sector:score:{sector_id}:{quarter or 'latest'}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    scored_panel = load_scored_panel()
    explanations = load_json(LOCAL_EXPLANATIONS_PATH)
    explanation_lookup = {(item["sector"], item["quarter"]): item for item in explanations}
    sector_frame = scored_panel.loc[scored_panel["sector_id"] == sector_id].sort_values("quarter")
    if sector_frame.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sector: {sector_id}")

    target_quarter = quarter or sector_frame.iloc[-1]["quarter"]
    row_frame = sector_frame.loc[sector_frame["quarter"] == target_quarter]
    if row_frame.empty:
        raise HTTPException(status_code=404, detail=f"No data for quarter {target_quarter}")
    row = row_frame.iloc[0]
    previous_score = int(sector_frame.loc[sector_frame["quarter"] < target_quarter, "stress_score"].iloc[-1]) if (sector_frame["quarter"] < target_quarter).any() else int(row["stress_score"])
    payload = {
        "sector": sector_id,
        "sector_name": row["sector_name"],
        "quarter": target_quarter,
        "stress_probability": round(float(row["stress_probability"]), 4),
        "stress_score": int(row["stress_score"]),
        "stress_label": row["stress_label"],
        "delta_qoq": int(row["stress_score"]) - previous_score,
        "base_value": float(row["base_value"]),
        "model_version": "v1",
        "shap_attributions": explanation_lookup[(sector_id, target_quarter)]["shap_attributions"],
        "model_breakdown": {
            key.replace("_probability", ""): round(float(row[key]), 4)
            for key in row.index
            if key.endswith("_probability")
        },
    }
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload


@router.get("/{sector_id}/history", summary="Get sector history and indicators")
async def get_sector_history(
    sector_id: str,
    from_quarter: str = Query("2018Q1"),
    to_quarter: str = Query("2025Q2"),
) -> dict:
    cache_key = f"sector:history:{sector_id}:{from_quarter}:{to_quarter}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    scored_panel = load_scored_panel()
    sector_frame = scored_panel.loc[scored_panel["sector_id"] == sector_id].sort_values("quarter")
    if sector_frame.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sector: {sector_id}")

    available = sorted_quarters(sector_frame["quarter"])
    payload = {
        "sector_id": sector_id,
        "sector_name": sector_frame.iloc[0]["sector_name"],
        "from_quarter": from_quarter,
        "to_quarter": to_quarter,
        "history": build_sector_history(scored_panel, sector_id, from_quarter, to_quarter),
        "available_quarters": available,
    }
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload
