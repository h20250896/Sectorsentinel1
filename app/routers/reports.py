from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.ml.data_loader import REPORTS_PATH, load_json
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{sector_id}/brief", summary="Get regulator brief payload")
async def get_regulator_brief(sector_id: str) -> dict:
    cache_key = f"report:{sector_id}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached
    reports = load_json(REPORTS_PATH)
    if sector_id not in reports:
        raise HTTPException(status_code=404, detail=f"Unknown sector: {sector_id}")
    payload = reports[sector_id]
    await cache.set_json(cache_key, payload, ttl_hours(24))
    return payload
