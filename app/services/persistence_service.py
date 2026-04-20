from __future__ import annotations

from sqlalchemy import func, select

from app.database import AsyncSessionLocal
from app.ml.constants import SECTORS
from app.ml.data_loader import ALERTS_PATH, LOCAL_EXPLANATIONS_PATH, load_json, load_scored_panel
from app.models.alert import Alert
from app.models.indicator import IndicatorSnapshot
from app.models.sector import Sector
from app.models.stress_score import StressScore


async def seed_database_from_artifacts() -> None:
    async with AsyncSessionLocal() as session:
        sector_count = await session.scalar(select(func.count(Sector.id)))
        if not sector_count:
            session.add_all([Sector(id=item["id"], name=item["name"], color=item["color"]) for item in SECTORS])
            await session.commit()

        score_count = await session.scalar(select(func.count(StressScore.id)))
        indicator_count = await session.scalar(select(func.count(IndicatorSnapshot.id)))
        alert_count = await session.scalar(select(func.count(Alert.id)))

        scored_panel = load_scored_panel()
        explanations = load_json(LOCAL_EXPLANATIONS_PATH)
        explanation_lookup = {(item["sector"], item["quarter"]): item["shap_attributions"] for item in explanations}

        if not score_count:
          session.add_all(
              [
                  StressScore(
                      sector_id=row["sector_id"],
                      quarter=row["quarter"],
                      stress_probability=float(row["stress_probability"]),
                      stress_score=int(row["stress_score"]),
                      stress_label=row["stress_label"],
                      quarter_on_quarter_delta=None,
                      base_value=float(row["base_value"]),
                      top_drivers=explanation_lookup.get((row["sector_id"], row["quarter"]), []),
                      model_version="v1",
                  )
                  for _, row in scored_panel.iterrows()
              ]
          )
          await session.commit()

        if not indicator_count:
            metadata_columns = {
                "sector_id",
                "sector_name",
                "sector_color",
                "quarter",
                "stress_probability",
                "stress_score",
                "stress_label",
                "base_value",
                "stress_label_target",
                "observed_stress_event",
            }
            session.add_all(
                [
                    IndicatorSnapshot(
                        sector_id=row["sector_id"],
                        quarter=row["quarter"],
                        payload={
                            key: row[key]
                            for key in row.index
                            if key not in metadata_columns and not key.endswith("_probability")
                        },
                    )
                    for _, row in scored_panel.iterrows()
                ]
            )
            await session.commit()

        if not alert_count:
            alerts = load_json(ALERTS_PATH)
            session.add_all(
                [
                    Alert(
                        id=alert["id"],
                        type=alert["type"],
                        sector_id=alert["sector_id"],
                        sector_name=alert["sector_name"],
                        message=alert["message"],
                        severity=alert["severity"],
                        quarter=alert["quarter"],
                        dismissed=bool(alert.get("dismissed", False)),
                    )
                    for alert in alerts
                ]
            )
            await session.commit()
