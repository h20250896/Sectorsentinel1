from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.data_loader import latest_quarter, sorted_quarters


def _resolve_dashboard_scope(scored_panel: pd.DataFrame, target_quarter: str | None) -> tuple[pd.DataFrame, list[str], str]:
    ordered_quarters = sorted_quarters(scored_panel["quarter"])
    resolved_quarter = target_quarter if target_quarter in ordered_quarters else ordered_quarters[-1]
    quarter_index = ordered_quarters.index(resolved_quarter)
    selected_quarters = ordered_quarters[: quarter_index + 1]
    scoped_panel = scored_panel.loc[scored_panel["quarter"].isin(selected_quarters)].copy()
    return scoped_panel, selected_quarters, resolved_quarter


def build_sector_summaries(
    scored_panel: pd.DataFrame,
    explanations: list[dict[str, Any]],
    target_quarter: str | None = None,
) -> list[dict[str, Any]]:
    explanation_lookup = {(item["sector"], item["quarter"]): item for item in explanations}
    scoped_panel, _, resolved_quarter = _resolve_dashboard_scope(scored_panel, target_quarter)
    summaries = []
    for sector_id, group in scoped_panel.groupby("sector_id", sort=False):
        group = group.sort_values("quarter")
        row = group.iloc[-1]
        previous_score = int(group.iloc[-2]["stress_score"]) if len(group) > 1 else int(row["stress_score"])
        delta = int(row["stress_score"]) - previous_score
        explanation = explanation_lookup.get((sector_id, resolved_quarter), {"shap_attributions": []})
        summaries.append(
            {
                "id": sector_id,
                "name": row["sector_name"],
                "color": row["sector_color"],
                "stress_score": int(row["stress_score"]),
                "stress_label": row["stress_label"],
                "trend_4q": group.tail(4)["stress_score"].astype(int).tolist(),
                "trend_8q": group.tail(8)["stress_score"].astype(int).tolist(),
                "top_drivers": [_badge_for_driver(item) for item in explanation.get("shap_attributions", [])[:2]],
                "delta_qoq": delta,
                "latest_quarter": resolved_quarter,
            }
        )
    return sorted(summaries, key=lambda item: item["stress_score"], reverse=True)


def build_heatmap(scored_panel: pd.DataFrame, quarters: list[str] | None = None) -> dict[str, Any]:
    selected_quarters = quarters or sorted_quarters(scored_panel["quarter"])[-12:]
    filtered = scored_panel.loc[scored_panel["quarter"].isin(selected_quarters)].copy()
    sectors = (
        filtered[["sector_id", "sector_name", "sector_color"]]
        .drop_duplicates()
        .sort_values("sector_id")
        .to_dict(orient="records")
    )
    cells = [
        {
            "sector_id": row["sector_id"],
            "sector_name": row["sector_name"],
            "quarter": row["quarter"],
            "stress_score": int(row["stress_score"]),
            "stress_label": row["stress_label"],
        }
        for _, row in filtered.sort_values(["sector_id", "quarter"]).iterrows()
    ]
    return {"quarters": selected_quarters, "sectors": sectors, "cells": cells}


def build_dashboard_payload(
    scored_panel: pd.DataFrame,
    alerts: list[dict],
    explanations: list[dict],
    target_quarter: str | None = None,
) -> dict[str, Any]:
    scoped_panel, ordered_quarters, current_quarter = _resolve_dashboard_scope(scored_panel, target_quarter)
    previous_quarter = ordered_quarters[-2] if len(ordered_quarters) > 1 else None
    latest_df = scoped_panel.loc[scoped_panel["quarter"] == current_quarter].copy()
    previous_df = scoped_panel.loc[scoped_panel["quarter"] == previous_quarter].copy() if previous_quarter else pd.DataFrame()
    aggregate = float(latest_df["stress_score"].mean())
    previous_aggregate = float(previous_df["stress_score"].mean()) if not previous_df.empty else aggregate
    recent_alert_quarters = set(ordered_quarters[-2:])
    visible_alerts = [alert for alert in alerts if alert["quarter"] in recent_alert_quarters]

    return {
        "current_quarter": current_quarter,
        "previous_quarter": previous_quarter,
        "model_last_run": current_quarter,
        "red_count": int((latest_df["stress_label"] == "RED").sum()),
        "amber_count": int((latest_df["stress_label"] == "AMBER").sum()),
        "aggregate_stress_index": round(aggregate, 2),
        "aggregate_delta": round(aggregate - previous_aggregate, 2),
        "active_alerts": len(visible_alerts),
        "sectors": build_sector_summaries(scoped_panel, explanations, current_quarter),
        "heatmap": build_heatmap(scoped_panel, ordered_quarters[-12:]),
        "alerts": visible_alerts[:10],
    }


def build_sector_history(scored_panel: pd.DataFrame, sector_id: str, from_quarter: str, to_quarter: str) -> list[dict[str, Any]]:
    history = scored_panel.loc[
        (scored_panel["sector_id"] == sector_id)
        & (scored_panel["quarter"] >= from_quarter)
        & (scored_panel["quarter"] <= to_quarter)
    ].sort_values("quarter")

    history_rows = []
    for _, row in history.iterrows():
        indicators = {
            column: row[column]
            for column in history.columns
            if column
            not in {
                "sector_id",
                "sector_name",
                "sector_color",
                "quarter",
                "stress_probability",
                "stress_score",
                "stress_label",
                "observed_stress_event",
                "stress_label_target",
            }
        }
        history_rows.append(
            {
                "quarter": row["quarter"],
                "stress_probability": round(float(row["stress_probability"]), 4),
                "stress_score": int(row["stress_score"]),
                "stress_label": row["stress_label"],
                "observed_stress_event": bool(row["observed_stress_event"]),
                "indicators": indicators,
            }
        )
    return history_rows


def _badge_for_driver(driver: dict[str, Any]) -> str:
    arrow = "up" if driver["direction"] == "positive" else "down"
    return f"{arrow} {driver['display_name']}"
