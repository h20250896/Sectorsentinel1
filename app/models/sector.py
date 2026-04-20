from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Sector(Base):
    __tablename__ = "sectors"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    color: Mapped[str] = mapped_column(String(12), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    scores = relationship("StressScore", back_populates="sector", cascade="all, delete-orphan")
    indicators = relationship("IndicatorSnapshot", back_populates="sector", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="sector", cascade="all, delete-orphan")
