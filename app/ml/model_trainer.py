from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.ml.contagion import build_contagion_network
from app.ml.data_loader import (
    ALERTS_PATH,
    CONTAGION_NETWORK_PATH,
    FEATURE_MATRIX_PATH,
    GLOBAL_IMPORTANCE_PATH,
    LOCAL_EXPLANATIONS_PATH,
    MODEL_PERFORMANCE_PATH,
    REPORTS_PATH,
    SCORED_PANEL_PATH,
    load_sector_panel,
    save_feature_matrix,
    save_json,
)
from app.ml.explainer import build_local_explanations
from app.ml.feature_engineer import engineer_features
from app.ml.inference import overlay_probability
from app.ml.stress_labeller import build_stress_labels
from app.ml.validator import aggregate_fold_metrics, evaluate_predictions, generate_walk_forward_splits
from app.services.alert_service import generate_alerts
from app.services.report_service import build_regulator_brief

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]


def build_base_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=3,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    else:  # pragma: no cover
        models["xgboost"] = RandomForestClassifier(n_estimators=150, max_depth=5, class_weight="balanced", random_state=42)

    if LGBMClassifier is not None:
        models["lgbm"] = LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            verbose=-1,
            random_state=42,
        )
    else:  # pragma: no cover
        models["lgbm"] = RandomForestClassifier(n_estimators=180, max_depth=4, class_weight="balanced", random_state=42)

    return models


def build_stacking_estimator() -> StackingClassifier:
    estimators = [(name, clone(model)) for name, model in build_base_models().items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        stack_method="predict_proba",
        passthrough=False,
        cv=3,
        n_jobs=None,
    )


def _prepare_training_frame(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metadata_columns = ["sector_name", "sector_color", "stress_label_target", "observed_stress_event"]
    metadata = feature_matrix[metadata_columns].copy()
    feature_columns = [
        column
        for column in feature_matrix.columns
        if column not in metadata_columns and pd.api.types.is_numeric_dtype(feature_matrix[column])
    ]
    features = feature_matrix[feature_columns].copy()
    return features, metadata, feature_columns


def run_walk_forward_validation(feature_matrix: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    modelling_frame = feature_matrix.reset_index()
    labelled = modelling_frame.dropna(subset=["stress_label_target"]).copy()
    labelled["stress_label_target"] = labelled["stress_label_target"].astype(int)
    splits = generate_walk_forward_splits(labelled["quarter"].tolist())

    folds: list[dict[str, Any]] = []
    latest_fold_payload: dict[str, Any] | None = None

    for fold_number, (train_quarters, validation_quarters) in enumerate(splits, start=1):
        train_mask = labelled["quarter"].isin(train_quarters)
        validation_mask = labelled["quarter"].isin(validation_quarters)
        X_train = labelled.loc[train_mask, feature_columns]
        y_train = labelled.loc[train_mask, "stress_label_target"]
        X_val = labelled.loc[validation_mask, feature_columns]
        y_val = labelled.loc[validation_mask, "stress_label_target"].to_numpy(dtype=int)

        calibrated = CalibratedClassifierCV(estimator=build_stacking_estimator(), method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        probabilities = calibrated.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, probabilities, threshold=0.45)
        fold_payload = {
            "fold": fold_number,
            "train_quarters": train_quarters,
            "validation_quarters": validation_quarters,
            **{metric: metrics[metric] for metric in ["auc_roc", "precision", "recall", "f1", "brier_score", "threshold"]},
        }
        folds.append(fold_payload)
        latest_fold_payload = {
            "fold": fold_number,
            "validation_quarters": validation_quarters,
            **metrics,
        }

    return {
        "model_version": settings.model_version,
        "folds": folds,
        "aggregate": aggregate_fold_metrics(folds),
        "latest_fold": latest_fold_payload,
    }


def train_and_persist(version: str = "v1") -> dict[str, Any]:
    raw_panel = load_sector_panel()
    labelled_panel = build_stress_labels(raw_panel)
    feature_matrix = engineer_features(labelled_panel)
    save_feature_matrix(feature_matrix, FEATURE_MATRIX_PATH)

    features, metadata, feature_columns = _prepare_training_frame(feature_matrix)
    performance = run_walk_forward_validation(feature_matrix, feature_columns)

    training_mask = feature_matrix["stress_label_target"].notna()
    X_train = features.loc[training_mask, feature_columns]
    y_train = feature_matrix.loc[training_mask, "stress_label_target"].astype(int)

    calibrated_stack = CalibratedClassifierCV(estimator=build_stacking_estimator(), method="isotonic", cv=3)
    calibrated_stack.fit(X_train, y_train)

    fitted_base_models: dict[str, Any] = {}
    for name, model in build_base_models().items():
        estimator = clone(model)
        estimator.fit(X_train, y_train)
        fitted_base_models[name] = estimator

    model_probabilities = calibrated_stack.predict_proba(features[feature_columns])[:, 1]
    probabilities = np.maximum(model_probabilities, overlay_probability(features[feature_columns]))
    stress_scores = np.rint(probabilities * 100).astype(int)
    stress_labels = np.where(stress_scores >= 65, "RED", np.where(stress_scores >= 40, "AMBER", "GREEN")).tolist()

    local_explanations, global_importance = build_local_explanations(
        features=features[feature_columns],
        metadata=metadata,
        random_forest_model=fitted_base_models["random_forest"],
        probabilities=probabilities,
        stress_scores=stress_scores,
        stress_labels=stress_labels,
    )

    model_breakdown = {
        name: estimator.predict_proba(features[feature_columns])[:, 1]
        for name, estimator in fitted_base_models.items()
    }
    model_breakdown["supervisory_overlay"] = overlay_probability(features[feature_columns])

    scored_panel = labelled_panel.copy()
    scored_panel["stress_probability"] = probabilities.round(6)
    scored_panel["stress_score"] = stress_scores
    scored_panel["stress_label"] = stress_labels
    for name, values in model_breakdown.items():
        scored_panel[f"{name}_probability"] = values.round(6)
    explanation_lookup = {(item["sector"], item["quarter"]): item for item in local_explanations}
    scored_panel["base_value"] = scored_panel.apply(
        lambda row: explanation_lookup[(row["sector_id"], row["quarter"])]["base_value"],
        axis=1,
    )
    scored_panel.to_csv(SCORED_PANEL_PATH, index=False)

    network = build_contagion_network(scored_panel)
    alerts = generate_alerts(scored_panel, network)
    briefs = {
        sector_id: build_regulator_brief(sector_id, scored_panel, local_explanations, network, performance)
        for sector_id in scored_panel["sector_id"].drop_duplicates().tolist()
    }

    bundle = {
        "version": version,
        "feature_columns": feature_columns,
        "calibrated_stack": calibrated_stack,
        "base_models": fitted_base_models,
        "performance": performance,
    }
    model_path = settings.models_dir / f"ensemble_{version}.pkl"
    joblib.dump(bundle, model_path)

    save_json(performance, MODEL_PERFORMANCE_PATH)
    save_json(global_importance[:15], GLOBAL_IMPORTANCE_PATH)
    save_json(local_explanations, LOCAL_EXPLANATIONS_PATH)
    save_json(network, CONTAGION_NETWORK_PATH)
    save_json(alerts, ALERTS_PATH)
    save_json(briefs, REPORTS_PATH)

    return {
        "model_path": str(model_path),
        "scored_panel_path": str(SCORED_PANEL_PATH),
        "performance_path": str(MODEL_PERFORMANCE_PATH),
    }


def main() -> None:
    result = train_and_persist(settings.model_version)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
