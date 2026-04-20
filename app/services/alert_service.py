from __future__ import annotations

from datetime import datetime

import pandas as pd


def _severity_for(alert_type: str) -> str:
    mapping = {
        "RED_ENTRY": "critical",
        "AMBER_ENTRY": "warning",
        "RAPID_RISE": "warning",
        "CONTAGION_RISK": "critical",
        "TREND_REVERSAL": "info",
    }
    return mapping.get(alert_type, "info")


def _build_alert(alert_type: str, sector_id: str, sector_name: str, quarter: str, message: str) -> dict:
    return {
        "id": f"{alert_type.lower()}-{sector_id}-{quarter}",
        "type": alert_type,
        "sector_id": sector_id,
        "sector_name": sector_name,
        "message": message,
        "severity": _severity_for(alert_type),
        "quarter": quarter,
        "created_at": datetime.utcnow().isoformat(),
        "dismissed": False,
    }


def generate_alerts(scored_panel: pd.DataFrame, network: dict) -> list[dict]:
    alerts: list[dict] = []
    latest_quarter = scored_panel["quarter"].drop_duplicates().iloc[-1]

    for sector_id, group in scored_panel.sort_values(["sector_id", "quarter"]).groupby("sector_id", sort=False):
        group = group.reset_index(drop=True)
        sector_name = group.loc[0, "sector_name"]
        deltas = group["stress_score"].diff().fillna(0)

        for idx in range(1, len(group)):
            current = group.loc[idx]
            previous = group.loc[idx - 1]
            delta = float(deltas.iloc[idx])

            if previous["stress_label"] == "AMBER" and current["stress_label"] == "RED":
                alerts.append(_build_alert("RED_ENTRY", sector_id, sector_name, current["quarter"], f"{sector_name} has crossed from AMBER into RED at {int(current['stress_score'])}."))
            if previous["stress_label"] == "GREEN" and current["stress_label"] == "AMBER":
                alerts.append(_build_alert("AMBER_ENTRY", sector_id, sector_name, current["quarter"], f"{sector_name} has moved into AMBER territory at {int(current['stress_score'])}."))
            if delta >= 12:
                alerts.append(_build_alert("RAPID_RISE", sector_id, sector_name, current["quarter"], f"{sector_name} stress score rose by {int(delta)} points quarter-on-quarter."))
            if idx >= 3:
                prior_deltas = deltas.iloc[idx - 3 : idx]
                if (prior_deltas > 0).all() and delta <= -10:
                    alerts.append(_build_alert("TREND_REVERSAL", sector_id, sector_name, current["quarter"], f"{sector_name} reversed sharply after three consecutive quarterly increases."))

    latest_scores = scored_panel.loc[scored_panel["quarter"] == latest_quarter, ["sector_id", "sector_name", "stress_label", "stress_score"]]
    score_lookup = latest_scores.set_index("sector_id").to_dict(orient="index")
    for edge in network.get("edges", []):
        source = score_lookup.get(edge["source"])
        target = score_lookup.get(edge["target"])
        if not source or not target:
            continue
        if source["stress_label"] == "RED" and target["stress_label"] == "GREEN" and edge["strength"] == "strong":
            alerts.append(
                _build_alert(
                    "CONTAGION_RISK",
                    edge["target"],
                    target["sector_name"],
                    latest_quarter,
                    f"{target['sector_name']} is exposed to stress spillover from {source['sector_name']} with a {edge['lag']}-quarter lag.",
                )
            )

    alerts.sort(key=lambda item: (item["quarter"], item["severity"] != "critical"), reverse=True)
    return alerts
