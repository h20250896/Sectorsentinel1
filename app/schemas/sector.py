from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SectorTrendPoint(BaseModel):
    quarter: str
    stress_score: int


class SectorSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    color: str
    stress_score: int
    stress_label: str
    trend_4q: list[int]
    top_drivers: list[str]
    delta_qoq: float
    latest_quarter: str


class SectorHistoryPoint(BaseModel):
    quarter: str
    stress_probability: float
    stress_score: int
    stress_label: str
    observed_stress_event: bool
    indicators: dict[str, float | int | str | None]


class SectorHistoryResponse(BaseModel):
    sector_id: str
    sector_name: str
    from_quarter: str
    to_quarter: str
    history: list[SectorHistoryPoint]
