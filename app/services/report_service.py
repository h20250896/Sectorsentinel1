from __future__ import annotations

from typing import Any

import pandas as pd


def build_regulator_brief(
    sector_id: str,
    scored_panel: pd.DataFrame,
    explanations: list[dict[str, Any]],
    network: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    history = scored_panel.loc[scored_panel["sector_id"] == sector_id].copy()
    if history.empty:
        raise ValueError(f"No data for sector {sector_id}")

    latest = history.iloc[-1]
    explanation = next(item for item in explanations if item["sector"] == sector_id and item["quarter"] == latest["quarter"])

    incoming = [edge for edge in network["edges"] if edge["target"] == sector_id]
    outgoing = [edge for edge in network["edges"] if edge["source"] == sector_id]
    recommendation_map = {
        "RED": ["Cap sector exposure", "Increase provisioning buffers", "Escalate to watchlist committee"],
        "AMBER": ["Enhance monitoring cadence", "Issue early warning memo", "Tighten underwriting terms"],
        "GREEN": ["Maintain normal monitoring", "Track emerging macro signals", "Preserve baseline exposure limits"],
    }

    top_driver_names = ", ".join(item["display_name"] for item in explanation["shap_attributions"][:3])
    summary = (
        f"{latest['sector_name']} is currently rated {latest['stress_label']} with a stress score of {int(latest['stress_score'])}. "
        f"The model is primarily driven by {top_driver_names}. "
        f"Current conditions suggest a {round(float(latest['stress_probability']) * 100, 1)}% near-term distress probability."
    )

    indicators = {
        "GNPA Ratio": latest["gnpa_ratio"],
        "Credit Growth YoY": latest["credit_growth_yoy"],
        "52-week Drawdown": latest["drawdown_52w"],
        "Interest Coverage Ratio": latest["interest_coverage_ratio"],
        "Repo Rate": latest["repo_rate"],
        "GDP Growth YoY": latest["gdp_growth_yoy"],
    }

    return {
        "sector_id": sector_id,
        "sector_name": latest["sector_name"],
        "quarter": latest["quarter"],
        "stress_score": int(latest["stress_score"]),
        "stress_label": latest["stress_label"],
        "executive_summary": summary,
        "key_indicators": indicators,
        "score_history": history[["quarter", "stress_score", "stress_label"]].to_dict(orient="records"),
        "top_drivers": explanation["shap_attributions"],
        "contagion_exposure": {"incoming": incoming, "outgoing": outgoing},
        "recommendations": recommendation_map[latest["stress_label"]],
        "model_version": performance.get("model_version", "v1"),
        "disclaimer": "Synthetic data and model outputs are for demonstration purposes only.",
    }
