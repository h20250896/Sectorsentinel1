from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.ml.data_loader import load_scored_panel
from app.utils.cache import cache, ttl_hours

router = APIRouter(prefix="/indicators", tags=["indicators"])


@router.get("/{sector_id}", summary="Get all raw indicators for a sector-quarter")
async def get_indicators(sector_id: str, quarter: str | None = Query(None)) -> dict:
    cache_key = f"indicators:{sector_id}:{quarter or 'latest'}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return cached

    scored_panel = load_scored_panel()
    sector_frame = scored_panel.loc[scored_panel["sector_id"] == sector_id].sort_values("quarter")
    if sector_frame.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sector: {sector_id}")
    target_quarter = quarter or sector_frame.iloc[-1]["quarter"]
    row_frame = sector_frame.loc[sector_frame["quarter"] == target_quarter]
    if row_frame.empty:
        raise HTTPException(status_code=404, detail=f"No indicators for quarter {target_quarter}")
    row = row_frame.iloc[0]
    payload = {
        "sector_id": sector_id,
        "sector_name": row["sector_name"],
        "quarter": target_quarter,
        "indicators": {
            key: row[key]
            for key in row.index
            if key
            not in {
                "sector_id",
                "sector_name",
                "sector_color",
                "quarter",
                "stress_probability",
                "stress_score",
                "stress_label",
                "base_value",
                "stress_label_target",
                "observed_stress_event",
            }
            and not key.endswith("_probability")
        },
    }
    await cache.set_json(cache_key, payload, ttl_hours(1))
    return payload
