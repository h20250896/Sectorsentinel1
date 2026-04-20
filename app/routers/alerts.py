from __future__ import annotations

from fastapi import APIRouter, Query

from app.ml.data_loader import ALERTS_PATH, load_json
from app.schemas.alert import PaginatedAlertsResponse
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=PaginatedAlertsResponse, summary="List active alerts")
async def list_alerts(
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=100),
    severity: str | None = None,
    sector_id: str | None = None,
    alert_type: str | None = Query(None, alias="type"),
) -> dict:
    cache_key = f"alerts:{page}:{limit}:{severity}:{sector_id}:{alert_type}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    alerts = load_json(ALERTS_PATH)
    filtered = alerts
    if severity:
        filtered = [alert for alert in filtered if alert["severity"] == severity]
    if sector_id:
        filtered = [alert for alert in filtered if alert["sector_id"] == sector_id]
    if alert_type:
        filtered = [alert for alert in filtered if alert["type"] == alert_type]

    start = (page - 1) * limit
    items = filtered[start : start + limit]
    payload = {"page": page, "limit": limit, "total": len(filtered), "items": items}
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload
