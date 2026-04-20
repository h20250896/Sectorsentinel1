from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class StressScore(Base):
    __tablename__ = "stress_scores"
    __table_args__ = (UniqueConstraint("sector_id", "quarter", name="uq_stress_score_sector_quarter"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sector_id: Mapped[str] = mapped_column(ForeignKey("sectors.id", ondelete="CASCADE"), nullable=False, index=True)
    quarter: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    stress_probability: Mapped[float] = mapped_column(Float, nullable=False)
    stress_score: Mapped[int] = mapped_column(Integer, nullable=False)
    stress_label: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    quarter_on_quarter_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_drivers: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    sector = relationship("Sector", back_populates="scores")
