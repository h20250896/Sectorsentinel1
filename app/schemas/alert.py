from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type: str
    sector_id: str | None
    sector_name: str
    message: str
    severity: str
    quarter: str
    created_at: datetime
    dismissed: bool = False
    metadata: dict | None = None


class PaginatedAlertsResponse(BaseModel):
    page: int
    limit: int
    total: int
    items: list[AlertResponse]
