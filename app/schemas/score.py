from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ShapAttribution(BaseModel):
    feature: str
    display_name: str
    value: float
    shap: float
    direction: str


class StressScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sector: str
    sector_name: str
    quarter: str
    stress_probability: float
    stress_score: int
    stress_label: str
    delta_qoq: float
    base_value: float
    model_version: str
    shap_attributions: list[ShapAttribution]
    model_breakdown: dict[str, float] = Field(default_factory=dict)


class HeatmapCell(BaseModel):
    sector_id: str
    sector_name: str
    quarter: str
    stress_score: int
    stress_label: str


class HeatmapResponse(BaseModel):
    quarters: list[str]
    sectors: list[dict[str, str]]
    cells: list[HeatmapCell]


class DashboardSummary(BaseModel):
    current_quarter: str
    previous_quarter: str | None
    model_last_run: str
    red_count: int
    amber_count: int
    aggregate_stress_index: float
    aggregate_delta: float
    active_alerts: int
    sectors: list[dict]
    heatmap: HeatmapResponse
    alerts: list[dict]


class ScenarioRequest(BaseModel):
    sector_id: str
    quarter: str | None = None
    overrides: dict[str, float]


class ScenarioResponse(BaseModel):
    sector: str
    quarter: str
    baseline_score: int
    stress_score: int
    stress_probability: float
    stress_label: str
    delta: float
    model_breakdown: dict[str, float]
    shap_attributions: list[ShapAttribution]
