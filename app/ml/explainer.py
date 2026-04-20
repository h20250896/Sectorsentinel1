from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore[assignment]


def display_name_for_feature(feature: str) -> str:
    name = feature.replace("_", " ").replace("qoq", "QoQ").replace("yoy", "YoY").title()
    replacements = {
        "Gnpa": "GNPA",
        "Npa": "NPA",
        "Gdp": "GDP",
        "Iip": "IIP",
        "Pmi": "PMI",
        "Fii": "FII",
        "Qoq": "QoQ",
        "Yoy": "YoY",
        "Icr": "ICR",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return name


def compute_rf_shap(random_forest_model: Any, features: pd.DataFrame) -> tuple[np.ndarray, float]:
    if shap is None:  # pragma: no cover
        pseudo = np.zeros_like(features.to_numpy(dtype=float))
        return pseudo, 0.5

    explainer = shap.TreeExplainer(random_forest_model)
    shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[-1], dtype=float)
        expected_value = explainer.expected_value[-1]
    else:
        shap_matrix = np.asarray(shap_values, dtype=float)
        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, -1]
        expected = explainer.expected_value
        expected_value = float(expected[-1] if isinstance(expected, (list, np.ndarray)) else expected)

    return shap_matrix, float(expected_value)


def top_attributions(feature_row: pd.Series, shap_row: np.ndarray, top_n: int = 5) -> list[dict[str, float | str]]:
    payload = []
    abs_order = np.argsort(np.abs(shap_row))[::-1][:top_n]
    for idx in abs_order:
        feature = feature_row.index[idx]
        shap_value = float(shap_row[idx])
        payload.append(
            {
                "feature": feature,
                "display_name": display_name_for_feature(feature),
                "value": round(float(feature_row.iloc[idx]), 4),
                "shap": round(shap_value, 4),
                "direction": "positive" if shap_value >= 0 else "negative",
            }
        )
    return payload


def infer_feature_category(feature: str) -> str:
    if any(token in feature for token in ["gnpa", "npa", "credit", "coverage", "restructured", "sma"]):
        return "credit"
    if any(token in feature for token in ["index", "drawdown", "book", "vol", "fii"]):
        return "market"
    if any(token in feature for token in ["repo", "inflation", "gdp", "iip", "account", "capacity", "confidence"]):
        return "macro"
    return "alternative"


def build_local_explanations(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    random_forest_model: Any,
    probabilities: np.ndarray,
    stress_scores: np.ndarray,
    stress_labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shap_matrix, base_value = compute_rf_shap(random_forest_model, features)
    explanations: list[dict[str, Any]] = []
    global_rows = []

    for row_idx, ((sector_id, quarter), row) in enumerate(features.iterrows()):
        attributions = top_attributions(row, shap_matrix[row_idx], top_n=5)
        explanations.append(
            {
                "sector": sector_id,
                "quarter": quarter,
                "sector_name": metadata.loc[(sector_id, quarter), "sector_name"],
                "stress_score": int(stress_scores[row_idx]),
                "stress_probability": round(float(probabilities[row_idx]), 4),
                "stress_label": stress_labels[row_idx],
                "base_value": round(base_value, 4),
                "shap_attributions": attributions,
            }
        )

    global_importance = np.mean(np.abs(shap_matrix), axis=0)
    for feature, importance in sorted(zip(features.columns, global_importance, strict=True), key=lambda item: item[1], reverse=True):
        global_rows.append(
            {
                "feature": feature,
                "display_name": display_name_for_feature(feature),
                "mean_abs_shap": round(float(importance), 6),
                "category": infer_feature_category(feature),
            }
        )

    return explanations, global_rows
