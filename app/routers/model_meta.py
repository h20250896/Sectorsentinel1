from __future__ import annotations

from fastapi import APIRouter

from app.ml.data_loader import GLOBAL_IMPORTANCE_PATH, MODEL_PERFORMANCE_PATH, load_json
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/performance", summary="Get walk-forward validation metrics")
async def get_model_performance() -> dict:
    cache_key = "model:performance"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached
    payload = load_json(MODEL_PERFORMANCE_PATH)
    await cache.set_json(cache_key, payload, ttl_hours(24))
    return payload


@router.get("/feature-importance", summary="Get global feature importance")
async def get_feature_importance() -> list[dict]:
    cache_key = "model:feature_importance"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached
    payload = load_json(GLOBAL_IMPORTANCE_PATH)
    await cache.set_json(cache_key, payload, ttl_hours(24))
    return payload
