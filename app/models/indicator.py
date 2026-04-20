from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class IndicatorSnapshot(Base):
    __tablename__ = "indicator_snapshots"
    __table_args__ = (UniqueConstraint("sector_id", "quarter", name="uq_indicator_sector_quarter"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sector_id: Mapped[str] = mapped_column(ForeignKey("sectors.id", ondelete="CASCADE"), nullable=False, index=True)
    quarter: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    sector = relationship("Sector", back_populates="indicators")
