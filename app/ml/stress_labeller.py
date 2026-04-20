from __future__ import annotations

import numpy as np
import pandas as pd


def _current_observed_stress(group: pd.DataFrame) -> pd.Series:
    gnpa_jump = group["gnpa_ratio"].diff().fillna(0) >= 1.5
    drawdown_breach = group["drawdown_52w"] <= -20
    credit_crunch = group["credit_growth_yoy"] < 0
    return (gnpa_jump | drawdown_breach | credit_crunch).astype(bool)


def build_stress_labels(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["sector_id", "quarter"]).copy()
    labels: list[pd.DataFrame] = []

    for _, group in df.groupby("sector_id", sort=False):
        group = group.copy()
        future_label = []
        for idx in range(len(group)):
            if idx >= len(group) - 2:
                future_label.append(np.nan)
                continue
            current_gnpa = group.iloc[idx]["gnpa_ratio"]
            future = group.iloc[idx + 1 : idx + 3]
            gnpa_jump = ((future["gnpa_ratio"] - current_gnpa) >= 1.5).any()
            drawdown_breach = (future["drawdown_52w"] <= -20).any()
            credit_crunch = (future["credit_growth_yoy"] < 0).any()
            future_label.append(int(gnpa_jump or drawdown_breach or credit_crunch))

        group["stress_label_target"] = future_label
        group["observed_stress_event"] = _current_observed_stress(group)
        labels.append(group)

    return pd.concat(labels, ignore_index=True)


def stress_frequency_by_sector(panel_with_labels: pd.DataFrame) -> pd.DataFrame:
    df = panel_with_labels.dropna(subset=["stress_label_target"]).copy()
    summary = df.groupby(["sector_id", "sector_name"], as_index=False).agg(
        stress_quarters=("stress_label_target", "sum"),
        labelled_quarters=("stress_label_target", "count"),
    )
    summary["stress_share"] = summary["stress_quarters"] / summary["labelled_quarters"]
    return summary.sort_values("stress_share", ascending=False)
