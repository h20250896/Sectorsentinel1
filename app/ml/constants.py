from __future__ import annotations

SECTORS = [
    {"id": "banking", "name": "Banking & Finance", "color": "#3b82f6"},
    {"id": "nbfc", "name": "NBFC", "color": "#f59e0b"},
    {"id": "real_estate", "name": "Real Estate", "color": "#ef4444"},
    {"id": "infrastructure", "name": "Infrastructure", "color": "#f97316"},
    {"id": "auto", "name": "Auto & Auto-Anc.", "color": "#14b8a6"},
    {"id": "it", "name": "IT & Technology", "color": "#22c55e"},
    {"id": "metals", "name": "Metals & Mining", "color": "#94a3b8"},
    {"id": "power", "name": "Power & Energy", "color": "#a78bfa"},
    {"id": "pharma", "name": "Pharma & Healthcare", "color": "#ec4899"},
    {"id": "fmcg", "name": "FMCG", "color": "#06b6d4"},
    {"id": "telecom", "name": "Telecom", "color": "#84cc16"},
]

RAW_FEATURES = [
    "gnpa_ratio",
    "net_npa_ratio",
    "sma_ratio",
    "credit_growth_yoy",
    "credit_to_gdp_gap",
    "interest_coverage_ratio",
    "restructured_assets_pct",
    "index_return_1q",
    "index_return_4q",
    "drawdown_52w",
    "price_to_book",
    "implied_vol",
    "fii_flow_qoq",
    "index_level",
    "repo_rate",
    "cpi_inflation",
    "gdp_growth_yoy",
    "iip_growth",
    "current_account_deficit_pct_gdp",
    "capacity_utilisation",
    "business_confidence_index",
    "gst_eway_bill_growth",
    "pmi_manufacturing",
    "pmi_services",
]

FEATURE_GROUPS = {
    "credit": {
        "label": "Credit",
        "features": [
            "gnpa_ratio",
            "net_npa_ratio",
            "sma_ratio",
            "credit_growth_yoy",
            "credit_to_gdp_gap",
            "interest_coverage_ratio",
            "restructured_assets_pct",
        ],
    },
    "market": {
        "label": "Market",
        "features": [
            "index_return_1q",
            "index_return_4q",
            "drawdown_52w",
            "price_to_book",
            "implied_vol",
            "fii_flow_qoq",
            "index_level",
        ],
    },
    "macro": {
        "label": "Macro",
        "features": [
            "repo_rate",
            "cpi_inflation",
            "gdp_growth_yoy",
            "iip_growth",
            "current_account_deficit_pct_gdp",
            "capacity_utilisation",
            "business_confidence_index",
        ],
    },
    "alternative": {
        "label": "Alternative",
        "features": [
            "gst_eway_bill_growth",
            "pmi_manufacturing",
            "pmi_services",
        ],
    },
}
