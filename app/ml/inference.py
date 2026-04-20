from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.ml.feature_engineer import engineer_features
from app.ml.stress_labeller import build_stress_labels


MODEL_BUNDLE_PATH = settings.models_dir / f"ensemble_{settings.model_version}.pkl"


def load_model_bundle(path: Path | None = None) -> dict[str, Any]:
    bundle_path = path or MODEL_BUNDLE_PATH
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {bundle_path}")
    return joblib.load(bundle_path)


def label_from_score(score: int) -> str:
    if score >= 65:
        return "RED"
    if score >= 40:
        return "AMBER"
    return "GREEN"


def overlay_probability(features: pd.DataFrame) -> np.ndarray:
    gnpa = features["gnpa_ratio"] if "gnpa_ratio" in features.columns else features["gnpa_ratio_level"]
    sma = features["sma_ratio"] if "sma_ratio" in features.columns else features["sma_ratio_level"]
    drawdown = features["drawdown_52w"] if "drawdown_52w" in features.columns else features["drawdown_52w_level"]
    icr = features["interest_coverage_ratio"] if "interest_coverage_ratio" in features.columns else features["interest_coverage_ratio_level"]
    restructured = features["restructured_assets_pct"] if "restructured_assets_pct" in features.columns else features["restructured_assets_pct_level"]
    repo = features["repo_rate"] if "repo_rate" in features.columns else features["repo_rate_level"]
    credit_growth = features["credit_growth_yoy"] if "credit_growth_yoy" in features.columns else features["credit_growth_yoy_level"]

    gnpa_component = np.clip((gnpa - 2.0) / 10.0, 0.0, 1.0)
    sma_component = np.clip((sma - 1.5) / 4.5, 0.0, 1.0)
    drawdown_component = np.clip(((-drawdown) - 10.0) / 15.0, 0.0, 1.0)
    icr_component = np.clip((2.5 - icr) / 1.5, 0.0, 1.0)
    restructured_component = np.clip(restructured / 6.0, 0.0, 1.0)
    repo_component = np.clip((repo - 5.5) / 3.0, 0.0, 1.0)
    credit_component = np.clip((-credit_growth) / 10.0, 0.0, 1.0)

    composite = (
        0.40 * gnpa_component
        + 0.18 * sma_component
        + 0.12 * drawdown_component
        + 0.12 * icr_component
        + 0.08 * restructured_component
        + 0.05 * repo_component
        + 0.05 * credit_component
    )
    composite_array = np.asarray(composite, dtype=float)
    return 1 / (1 + np.exp(-(composite_array - 0.42) / 0.08))


def score_features(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    feature_columns: list[str] = bundle["feature_columns"]
    model = bundle["calibrated_stack"]
    feature_frame = features[feature_columns]
    model_probabilities = model.predict_proba(feature_frame)[:, 1]
    overlay_probabilities = overlay_probability(features)
    probabilities = np.maximum(model_probabilities, overlay_probabilities)
    scores = np.rint(probabilities * 100).astype(int)
    labels = [label_from_score(int(score)) for score in scores]
    base_breakdown = {
        model_name: estimator.predict_proba(feature_frame)[:, 1]
        for model_name, estimator in bundle["base_models"].items()
    }
    base_breakdown["supervisory_overlay"] = overlay_probabilities
    return probabilities, scores, labels, base_breakdown


def simulate_scenario(
    bundle: dict[str, Any],
    raw_panel: pd.DataFrame,
    sector_id: str,
    overrides: dict[str, float],
    quarter: str | None = None,
) -> tuple[pd.DataFrame, str]:
    target_quarter = quarter or raw_panel["quarter"].drop_duplicates().iloc[-1]
    scenario_panel = raw_panel.copy()
    mask = (scenario_panel["sector_id"] == sector_id) & (scenario_panel["quarter"] == target_quarter)
    if not mask.any():
        raise ValueError(f"No baseline row for sector={sector_id} quarter={target_quarter}")

    for feature, value in overrides.items():
        if feature in scenario_panel.columns:
            scenario_panel.loc[mask, feature] = value

    labelled = build_stress_labels(scenario_panel)
    engineered = engineer_features(labelled)
    return engineered, target_quarter
