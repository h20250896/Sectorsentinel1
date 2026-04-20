from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.constants import RAW_FEATURES, SECTORS
from app.ml.data_loader import save_sector_panel


RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class SectorProfile:
    sector_id: str
    base_gnpa: float
    gnpa_span: float
    base_credit_growth: float
    growth_sensitivity: float
    base_icr: float
    icr_sensitivity: float
    base_pb: float
    base_vol: float
    index_base: float
    macro_sensitivity: float
    episodes: tuple[tuple[int, int, float], ...]


SECTOR_PROFILES: dict[str, SectorProfile] = {
    "banking": SectorProfile("banking", 5.6, 6.8, 10.5, 12.0, 2.2, 1.0, 1.4, 24.0, 15000, 0.8, ((2, 7, 0.55), (18, 23, 0.35))),
    "nbfc": SectorProfile("nbfc", 6.8, 7.8, 12.5, 14.5, 1.9, 1.15, 1.7, 28.0, 12000, 0.95, ((2, 9, 0.7), (18, 21, 0.45))),
    "real_estate": SectorProfile("real_estate", 7.4, 8.2, 9.5, 16.0, 1.8, 1.25, 1.1, 26.0, 9000, 1.05, ((5, 12, 0.82), (20, 26, 0.72))),
    "infrastructure": SectorProfile("infrastructure", 8.6, 9.8, 8.7, 18.5, 1.6, 1.3, 0.95, 27.0, 8000, 1.1, ((0, 8, 0.88), (15, 26, 0.92))),
    "auto": SectorProfile("auto", 4.1, 5.1, 11.8, 11.0, 2.8, 0.95, 2.1, 29.0, 13500, 0.85, ((6, 10, 0.45), (18, 20, 0.35))),
    "it": SectorProfile("it", 0.6, 1.0, 12.6, 4.0, 5.2, 0.35, 4.6, 18.0, 22000, 0.35, ((9, 10, 0.72),)),
    "metals": SectorProfile("metals", 4.8, 5.6, 10.4, 13.0, 2.5, 1.0, 1.5, 31.0, 11000, 1.0, ((7, 11, 0.58), (16, 20, 0.62))),
    "power": SectorProfile("power", 5.4, 7.4, 9.1, 14.0, 2.0, 1.05, 1.2, 25.0, 9500, 0.95, ((4, 8, 0.48), (16, 24, 0.7))),
    "pharma": SectorProfile("pharma", 2.0, 2.8, 10.9, 6.5, 3.6, 0.55, 3.1, 20.0, 14500, 0.5, ((8, 10, 0.18), (22, 23, 0.16))),
    "fmcg": SectorProfile("fmcg", 1.6, 2.2, 9.4, 5.5, 4.1, 0.45, 4.0, 17.0, 17500, 0.42, ((12, 13, 0.16),)),
    "telecom": SectorProfile("telecom", 4.2, 5.9, 8.8, 12.5, 2.2, 1.0, 1.4, 27.0, 10500, 0.9, ((5, 8, 0.5), (19, 21, 0.36))),
}

CONTAGION_RULES: tuple[tuple[str, str, int, float], ...] = (
    ("banking", "nbfc", 1, 0.14),
    ("nbfc", "real_estate", 2, 0.2),
    ("infrastructure", "power", 2, 0.18),
    ("metals", "auto", 1, 0.12),
    ("banking", "real_estate", 2, 0.1),
    ("telecom", "banking", 1, 0.08),
)


def _quarter_strings() -> list[str]:
    return [str(period) for period in pd.period_range("2018Q1", "2025Q2", freq="Q")]


def _generate_macro_panel(quarters: list[str]) -> pd.DataFrame:
    timeline = np.arange(len(quarters))
    repo_rate = 5.25 + 0.9 * np.sin(timeline / 3.1) + 0.22 * timeline / len(quarters)
    repo_rate += np.where(timeline >= 18, 0.45, 0.0)
    cpi_inflation = 4.7 + 0.7 * np.sin(timeline / 2.4 + 0.8) + np.where(timeline >= 17, 0.6, 0.0)
    gdp_growth_yoy = 6.4 + 1.1 * np.sin(timeline / 2.7) - np.where((timeline >= 8) & (timeline <= 11), 1.4, 0.0)
    iip_growth = 5.1 + 2.0 * np.sin(timeline / 1.9 + 0.4) - np.where((timeline >= 8) & (timeline <= 10), 3.2, 0.0)
    cad = 1.8 + 0.6 * np.sin(timeline / 4.0 + 1.2) + np.where(timeline >= 16, 0.35, 0.0)
    capacity = 71.5 + 3.2 * np.sin(timeline / 2.8 + 0.3) - np.where((timeline >= 8) & (timeline <= 11), 2.4, 0.0)
    business_conf = 100 + 5.8 * np.sin(timeline / 2.5 + 0.9) - np.where((timeline >= 7) & (timeline <= 11), 8.0, 0.0)
    pmi_mfg = 52.4 + 1.8 * np.sin(timeline / 2.1) - np.where((timeline >= 8) & (timeline <= 11), 4.2, 0.0)
    pmi_srv = 53.0 + 1.5 * np.sin(timeline / 2.3 + 0.5) - np.where((timeline >= 8) & (timeline <= 11), 3.6, 0.0)
    gst_growth = 9.2 + 3.4 * np.sin(timeline / 1.8 + 0.2) - np.where((timeline >= 8) & (timeline <= 11), 10.5, 0.0)

    return pd.DataFrame(
        {
            "quarter": quarters,
            "repo_rate": np.round(repo_rate + RNG.normal(0, 0.08, len(quarters)), 2),
            "cpi_inflation": np.round(cpi_inflation + RNG.normal(0, 0.18, len(quarters)), 2),
            "gdp_growth_yoy": np.round(gdp_growth_yoy + RNG.normal(0, 0.22, len(quarters)), 2),
            "iip_growth": np.round(iip_growth + RNG.normal(0, 0.45, len(quarters)), 2),
            "current_account_deficit_pct_gdp": np.round(cad + RNG.normal(0, 0.06, len(quarters)), 2),
            "capacity_utilisation": np.round(capacity + RNG.normal(0, 0.35, len(quarters)), 2),
            "business_confidence_index": np.round(business_conf + RNG.normal(0, 0.9, len(quarters)), 2),
            "gst_eway_bill_growth": np.round(gst_growth + RNG.normal(0, 0.65, len(quarters)), 2),
            "pmi_manufacturing": np.round(pmi_mfg + RNG.normal(0, 0.18, len(quarters)), 2),
            "pmi_services": np.round(pmi_srv + RNG.normal(0, 0.2, len(quarters)), 2),
        }
    )


def _build_stress_profiles(quarters: list[str], macro: pd.DataFrame) -> dict[str, np.ndarray]:
    macro_headwind = (
        (macro["repo_rate"] - macro["repo_rate"].mean()) / macro["repo_rate"].std()
        + (macro["cpi_inflation"] - macro["cpi_inflation"].mean()) / macro["cpi_inflation"].std()
        - (macro["gdp_growth_yoy"] - macro["gdp_growth_yoy"].mean()) / macro["gdp_growth_yoy"].std()
    ) / 3
    macro_headwind = macro_headwind.to_numpy()
    stress: dict[str, np.ndarray] = {}

    for sector in SECTORS:
        profile = SECTOR_PROFILES[sector["id"]]
        series = np.full(len(quarters), 0.15 + profile.macro_sensitivity * 0.08)
        series += 0.06 * np.sin(np.arange(len(quarters)) / 2.5 + len(sector["id"]) / 4)
        series += profile.macro_sensitivity * 0.12 * macro_headwind
        for start, end, amplitude in profile.episodes:
            width = max(end - start + 1, 1)
            taper = np.sin(np.linspace(0.2, np.pi - 0.2, width))
            series[start : end + 1] += amplitude * taper
        for idx in range(1, len(series)):
            series[idx] = 0.55 * series[idx] + 0.35 * series[idx - 1] + RNG.normal(0, 0.015)
        stress[sector["id"]] = np.clip(series, 0.02, 1.0)

    for source, target, lag, weight in CONTAGION_RULES:
        source_series = stress[source]
        target_series = stress[target].copy()
        for idx in range(lag, len(quarters)):
            target_series[idx] += source_series[idx - lag] * weight
        stress[target] = np.clip(target_series, 0.02, 1.0)
    return stress


def generate_synthetic_panel() -> pd.DataFrame:
    quarters = _quarter_strings()
    macro = _generate_macro_panel(quarters)
    stress_profiles = _build_stress_profiles(quarters, macro)
    rows: list[dict[str, float | str]] = []

    for sector in SECTORS:
        profile = SECTOR_PROFILES[sector["id"]]
        stress = stress_profiles[sector["id"]]
        index_level = profile.index_base
        trailing_returns: list[float] = []
        previous_gnpa = profile.base_gnpa

        for idx, quarter in enumerate(quarters):
            macro_row = macro.iloc[idx]
            stress_t = float(stress[idx])
            tailwind = float(max(macro_row["gdp_growth_yoy"] - 5.0, 0.0))
            headwind = float(max(macro_row["repo_rate"] - 5.75, 0.0))
            gnpa = profile.base_gnpa + profile.gnpa_span * stress_t + 0.35 * max(stress_t - (stress[idx - 1] if idx else stress_t), 0)
            gnpa += RNG.normal(0, 0.18)
            if sector["id"] == "it":
                gnpa = min(1.9, gnpa)
            if sector["id"] == "infrastructure":
                gnpa = min(18.2, gnpa)
            gnpa = max(0.4, gnpa)

            net_npa = max(0.2, gnpa * (0.52 + 0.07 * RNG.random()))
            sma_ratio = max(0.3, 1.1 + gnpa * 0.42 + stress_t * 1.6 + RNG.normal(0, 0.12))
            credit_growth = profile.base_credit_growth - profile.growth_sensitivity * stress_t - 0.8 * headwind + 0.35 * tailwind
            credit_growth += RNG.normal(0, 0.65)
            credit_to_gdp_gap = 1.8 * np.sin(idx / 2.7 + profile.base_gnpa / 4) - 2.4 * stress_t + RNG.normal(0, 0.35)
            interest_coverage = max(0.6, profile.base_icr - profile.icr_sensitivity * stress_t - 0.08 * headwind + 0.04 * tailwind)
            interest_coverage += RNG.normal(0, 0.08)
            restructured_assets = max(0.1, 0.25 + 3.0 * stress_t + RNG.normal(0, 0.12))

            index_return_1q = 3.4 + 1.8 * np.sin(idx / 1.7 + profile.base_pb) - 18.0 * stress_t + 0.65 * tailwind
            index_return_1q += RNG.normal(0, 1.4)
            if sector["id"] == "it" and idx in {8, 9, 10}:
                index_return_1q -= 8.5
            index_level = max(1800.0, index_level * (1 + index_return_1q / 100))
            trailing_returns.append(index_return_1q)
            index_return_4q = float(np.sum(trailing_returns[-4:]))

            drawdown = -4.0 - 26.0 * stress_t - 0.8 * max(previous_gnpa - gnpa, 0) + RNG.normal(0, 1.8)
            if sector["id"] == "it" and idx in {8, 9, 10}:
                drawdown -= 13.0
            drawdown = min(-0.5, drawdown)
            price_to_book = max(0.45, profile.base_pb - 1.25 * stress_t + RNG.normal(0, 0.08))
            implied_vol = max(12.0, profile.base_vol + 13.0 * stress_t + RNG.normal(0, 1.2))
            fii_flow = 1800 - 5200 * stress_t + 320 * tailwind - 150 * headwind + RNG.normal(0, 420)

            gst_growth = macro_row["gst_eway_bill_growth"] - 5.4 * stress_t + RNG.normal(0, 0.5)
            if sector["id"] in {"fmcg", "pharma"}:
                gst_growth += 1.1
            if sector["id"] in {"real_estate", "infrastructure"}:
                gst_growth -= 1.3

            rows.append(
                {
                    "sector_id": sector["id"],
                    "sector_name": sector["name"],
                    "sector_color": sector["color"],
                    "quarter": quarter,
                    "gnpa_ratio": round(gnpa, 2),
                    "net_npa_ratio": round(min(net_npa, gnpa - 0.1), 2),
                    "sma_ratio": round(sma_ratio, 2),
                    "credit_growth_yoy": round(credit_growth, 2),
                    "credit_to_gdp_gap": round(credit_to_gdp_gap, 2),
                    "interest_coverage_ratio": round(interest_coverage, 2),
                    "restructured_assets_pct": round(restructured_assets, 2),
                    "index_return_1q": round(index_return_1q, 2),
                    "index_return_4q": round(index_return_4q, 2),
                    "drawdown_52w": round(drawdown, 2),
                    "price_to_book": round(price_to_book, 2),
                    "implied_vol": round(implied_vol, 2),
                    "fii_flow_qoq": round(fii_flow, 2),
                    "index_level": round(index_level, 2),
                    "repo_rate": float(macro_row["repo_rate"]),
                    "cpi_inflation": float(macro_row["cpi_inflation"]),
                    "gdp_growth_yoy": float(macro_row["gdp_growth_yoy"]),
                    "iip_growth": float(macro_row["iip_growth"]),
                    "current_account_deficit_pct_gdp": float(macro_row["current_account_deficit_pct_gdp"]),
                    "capacity_utilisation": float(macro_row["capacity_utilisation"]),
                    "business_confidence_index": float(macro_row["business_confidence_index"]),
                    "gst_eway_bill_growth": round(gst_growth, 2),
                    "pmi_manufacturing": float(macro_row["pmi_manufacturing"]),
                    "pmi_services": float(macro_row["pmi_services"]),
                }
            )
            previous_gnpa = gnpa

    panel = pd.DataFrame(rows).sort_values(["sector_id", "quarter"]).reset_index(drop=True)
    assert len(panel) == 330, "Synthetic dataset must contain 330 rows."
    expected_columns = {"sector_id", "quarter", *RAW_FEATURES}
    missing = expected_columns.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing synthetic columns: {sorted(missing)}")
    return panel


def main() -> None:
    panel = generate_synthetic_panel()
    path = save_sector_panel(panel)
    print(f"Saved synthetic panel to {path}")
    print(f"Rows: {len(panel)}")
    print(f"Columns: {len(panel.columns)}")
    print(panel.groupby('sector_id').size().to_string())


if __name__ == "__main__":
    main()
