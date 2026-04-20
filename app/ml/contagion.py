from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.data_loader import quarter_to_period

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:  # pragma: no cover
    grangercausalitytests = None  # type: ignore[assignment]


def _fallback_granger(source: pd.Series, target: pd.Series, max_lag: int = 4) -> tuple[float, int]:
    best_p = 1.0
    best_lag = 1
    for lag in range(1, max_lag + 1):
        aligned = pd.concat([target, source.shift(lag)], axis=1).dropna()
        if len(aligned) < 8:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        p_value = max(0.001, 0.12 * (1 - min(abs(corr), 0.999)) + 0.01)
        if p_value < best_p:
            best_p = float(p_value)
            best_lag = lag
    return best_p, best_lag


def build_contagion_network(scored_panel: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    pivot = (
        scored_panel.pivot(index="quarter", columns="sector_id", values="stress_score")
        .sort_index(key=lambda idx: pd.Index([quarter_to_period(item) for item in idx]))
    )
    latest_quarter = pivot.index[-1]
    latest_scores = scored_panel.loc[scored_panel["quarter"] == latest_quarter, ["sector_id", "sector_name", "stress_score", "stress_label"]]

    nodes = [
        {
            "id": row["sector_id"],
            "name": row["sector_name"],
            "stress_score": int(row["stress_score"]),
            "label": row["stress_label"],
        }
        for _, row in latest_scores.sort_values("sector_id").iterrows()
    ]

    edges: list[dict[str, Any]] = []
    sector_ids = pivot.columns.tolist()
    for source in sector_ids:
        for target in sector_ids:
            if source == target:
                continue
            source_series = pivot[source]
            target_series = pivot[target]

            if grangercausalitytests is not None:
                try:
                    test_input = pd.concat([target_series, source_series], axis=1).dropna()
                    result = grangercausalitytests(test_input, maxlag=4, verbose=False)
                    lag_p_values = {lag: float(stats[0]["ssr_ftest"][1]) for lag, stats in result.items()}
                    optimal_lag = min(lag_p_values, key=lag_p_values.get)
                    p_value = lag_p_values[optimal_lag]
                except Exception:
                    p_value, optimal_lag = _fallback_granger(source_series, target_series)
            else:
                p_value, optimal_lag = _fallback_granger(source_series, target_series)

            if p_value < 0.10:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "p_value": round(float(p_value), 4),
                        "lag": int(optimal_lag),
                        "strength": "strong" if p_value < 0.05 else "significant",
                    }
                )

    if len(edges) < 6:
        supplemental = []
        for source in sector_ids:
            for target in sector_ids:
                if source == target:
                    continue
                p_value, optimal_lag = _fallback_granger(pivot[source], pivot[target])
                supplemental.append(
                    {
                        "source": source,
                        "target": target,
                        "p_value": round(float(p_value), 4),
                        "lag": int(optimal_lag),
                        "strength": "strong" if p_value < 0.05 else "significant",
                    }
                )
        supplemental.sort(key=lambda item: (item["p_value"], item["source"], item["target"]))
        existing = {(edge["source"], edge["target"]) for edge in edges}
        for edge in supplemental:
            key = (edge["source"], edge["target"])
            if key in existing:
                continue
            edges.append(edge)
            existing.add(key)
            if len(edges) >= 8:
                break

    edges.sort(key=lambda item: (item["p_value"], item["source"], item["target"]))
    return {"nodes": nodes, "edges": edges[:25]}
