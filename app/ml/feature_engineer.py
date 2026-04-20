from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.constants import RAW_FEATURES


def _safe_yoy(current: pd.Series, lagged: pd.Series) -> pd.Series:
    denominator = lagged.replace(0, np.nan).abs()
    return ((current - lagged) / denominator) * 100


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["sector_id", "quarter"]).copy()
    numeric_features = [feature for feature in RAW_FEATURES if feature in df.columns]
    derived_columns: dict[str, pd.Series] = {}

    for feature in numeric_features:
        derived_columns[f"{feature}_level"] = df[feature]
        grouped = df.groupby("sector_id", sort=False)[feature]
        lag_1 = grouped.shift(1)
        lag_2 = grouped.shift(2)
        lag_4 = grouped.shift(4)
        derived_columns[f"{feature}_lag1"] = lag_1
        derived_columns[f"{feature}_lag2"] = lag_2
        derived_columns[f"{feature}_lag4"] = lag_4
        derived_columns[f"{feature}_yoy_change"] = _safe_yoy(df[feature], lag_4)

    derived_columns["repo_rate_x_credit"] = df["credit_growth_yoy"] * df["repo_rate"]
    derived_columns["index_return_momentum_2q"] = df.groupby("sector_id", sort=False)["index_return_1q"].diff(2)
    derived_columns["credit_growth_momentum_2q"] = df.groupby("sector_id", sort=False)["credit_growth_yoy"].diff(2)

    rolling_mean = (
        df.groupby("sector_id", sort=False)["gnpa_ratio"]
        .rolling(window=4, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        df.groupby("sector_id", sort=False)["gnpa_ratio"]
        .rolling(window=4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .replace(0, np.nan)
    )
    derived_columns["gnpa_ratio_rolling_zscore_4q"] = (df["gnpa_ratio"] - rolling_mean) / rolling_std

    total_gnpa = df.groupby("quarter")["gnpa_ratio"].transform("sum")
    sector_counts = df.groupby("quarter")["sector_id"].transform("count").replace(1, np.nan)
    derived_columns["cross_sector_avg_gnpa"] = (total_gnpa - df["gnpa_ratio"]) / (sector_counts - 1)
    derived_columns["quarter_index"] = df.groupby("sector_id", sort=False).cumcount()

    df = pd.concat([df, pd.DataFrame(derived_columns)], axis=1)

    feature_df = df.set_index(["sector_id", "quarter"]).sort_index()
    metadata_columns = ["sector_name", "sector_color", "stress_label_target", "observed_stress_event"]
    for column in metadata_columns:
        if column not in feature_df.columns:
            feature_df[column] = np.nan

    numeric_columns = feature_df.select_dtypes(include=["number", "bool"]).columns
    feature_df[numeric_columns] = feature_df[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feature_df
