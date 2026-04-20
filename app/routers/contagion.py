from __future__ import annotations

from fastapi import APIRouter

from app.ml.data_loader import CONTAGION_NETWORK_PATH, load_json
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/contagion", tags=["contagion"])


@router.get("/network", summary="Get Granger contagion network")
async def get_contagion_network() -> dict:
    cache_key = "contagion:network"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached
    payload = load_json(CONTAGION_NETWORK_PATH)
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload
