from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.ml.constants import FEATURE_GROUPS, RAW_FEATURES, SECTORS


PANEL_PATH = settings.data_dir / "sector_panel.csv"
FEATURE_MATRIX_PATH = settings.data_dir / "feature_matrix.csv"
SCORED_PANEL_PATH = settings.artifacts_dir / "scored_panel.csv"
MODEL_PERFORMANCE_PATH = settings.artifacts_dir / "model_performance.json"
GLOBAL_IMPORTANCE_PATH = settings.artifacts_dir / "global_feature_importance.json"
LOCAL_EXPLANATIONS_PATH = settings.artifacts_dir / "local_explanations.json"
CONTAGION_NETWORK_PATH = settings.artifacts_dir / "contagion_network.json"
ALERTS_PATH = settings.artifacts_dir / "alerts.json"
REPORTS_PATH = settings.artifacts_dir / "regulator_briefs.json"


def quarter_to_period(quarter: str) -> pd.Period:
    return pd.Period(quarter, freq="Q")


def sorted_quarters(quarters: list[str] | pd.Series) -> list[str]:
    unique = pd.Index(quarters).dropna().unique().tolist()
    return sorted(unique, key=quarter_to_period)


def load_sector_panel(path: Path | None = None) -> pd.DataFrame:
    panel_path = path or PANEL_PATH
    if not panel_path.exists():
        raise FileNotFoundError(f"Synthetic panel not found at {panel_path}")
    return pd.read_csv(panel_path)


def save_sector_panel(panel: pd.DataFrame, path: Path | None = None) -> Path:
    panel_path = path or PANEL_PATH
    panel.to_csv(panel_path, index=False)
    return panel_path


def load_feature_matrix(path: Path | None = None) -> pd.DataFrame:
    matrix_path = path or FEATURE_MATRIX_PATH
    if not matrix_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {matrix_path}")
    df = pd.read_csv(matrix_path)
    if {"sector_id", "quarter"}.issubset(df.columns):
        df = df.set_index(["sector_id", "quarter"])
    return df


def save_feature_matrix(matrix: pd.DataFrame, path: Path | None = None) -> Path:
    matrix_path = path or FEATURE_MATRIX_PATH
    matrix.reset_index().to_csv(matrix_path, index=False)
    return matrix_path


def load_scored_panel(path: Path | None = None) -> pd.DataFrame:
    scored_path = path or SCORED_PANEL_PATH
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored panel not found at {scored_path}")
    return pd.read_csv(scored_path)


def save_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2, default=str)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def latest_quarter(df: pd.DataFrame) -> str:
    return sorted_quarters(df["quarter"])[-1]


def sector_name_map() -> dict[str, str]:
    return {item["id"]: item["name"] for item in SECTORS}


def sector_color_map() -> dict[str, str]:
    return {item["id"]: item["color"] for item in SECTORS}
