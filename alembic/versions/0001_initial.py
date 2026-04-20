"""Initial SectorSentinel schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sectors",
        sa.Column("id", sa.String(length=50), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("color", sa.String(length=12), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.String(length=100), primary_key=True),
        sa.Column("type", sa.String(length=40), nullable=False),
        sa.Column("sector_id", sa.String(length=50), sa.ForeignKey("sectors.id", ondelete="SET NULL")),
        sa.Column("sector_name", sa.String(length=120), nullable=False),
        sa.Column("message", sa.String(length=500), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("quarter", sa.String(length=8), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("dismissed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_alerts_type", "alerts", ["type"])
    op.create_index("ix_alerts_sector_id", "alerts", ["sector_id"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])
    op.create_index("ix_alerts_quarter", "alerts", ["quarter"])

    op.create_table(
        "indicator_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sector_id", sa.String(length=50), sa.ForeignKey("sectors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("quarter", sa.String(length=8), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("sector_id", "quarter", name="uq_indicator_sector_quarter"),
    )
    op.create_index("ix_indicator_snapshots_sector_id", "indicator_snapshots", ["sector_id"])
    op.create_index("ix_indicator_snapshots_quarter", "indicator_snapshots", ["quarter"])

    op.create_table(
        "stress_scores",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sector_id", sa.String(length=50), sa.ForeignKey("sectors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("quarter", sa.String(length=8), nullable=False),
        sa.Column("stress_probability", sa.Float(), nullable=False),
        sa.Column("stress_score", sa.Integer(), nullable=False),
        sa.Column("stress_label", sa.String(length=12), nullable=False),
        sa.Column("quarter_on_quarter_delta", sa.Float(), nullable=True),
        sa.Column("base_value", sa.Float(), nullable=True),
        sa.Column("top_drivers", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("sector_id", "quarter", name="uq_stress_score_sector_quarter"),
    )
    op.create_index("ix_stress_scores_sector_id", "stress_scores", ["sector_id"])
    op.create_index("ix_stress_scores_quarter", "stress_scores", ["quarter"])
    op.create_index("ix_stress_scores_stress_label", "stress_scores", ["stress_label"])


def downgrade() -> None:
    op.drop_index("ix_stress_scores_stress_label", table_name="stress_scores")
    op.drop_index("ix_stress_scores_quarter", table_name="stress_scores")
    op.drop_index("ix_stress_scores_sector_id", table_name="stress_scores")
    op.drop_table("stress_scores")
    op.drop_index("ix_indicator_snapshots_quarter", table_name="indicator_snapshots")
    op.drop_index("ix_indicator_snapshots_sector_id", table_name="indicator_snapshots")
    op.drop_table("indicator_snapshots")
    op.drop_index("ix_alerts_quarter", table_name="alerts")
    op.drop_index("ix_alerts_severity", table_name="alerts")
    op.drop_index("ix_alerts_sector_id", table_name="alerts")
    op.drop_index("ix_alerts_type", table_name="alerts")
    op.drop_table("alerts")
    op.drop_table("sectors")
