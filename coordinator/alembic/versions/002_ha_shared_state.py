"""HA shared state tables: reputation, incentives, geo, rate limits, locks.

Revision ID: 002_ha_shared_state
Revises: 001_initial
Create Date: 2026-07-18
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_ha_shared_state"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "client_reputation",
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("rounds_participated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rounds_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rounds_dropped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updates_submitted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updates_accepted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updates_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_latency_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_samples", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_seen", sa.Float(), nullable=True),
        sa.Column("last_seen", sa.Float(), nullable=True),
        sa.Column("client_rounds", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("client_id"),
    )

    op.create_table(
        "client_incentives",
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("total_tokens_earned", sa.Float(), nullable=False, server_default="0"),
        sa.Column("tokens_spent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("speed_bonuses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("consistency_bonuses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rewards_received", sa.JSON(), nullable=True),
        sa.Column("consecutive_completions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_completion_time", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("client_id"),
    )

    op.create_table(
        "geo_presence",
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lng", sa.Float(), nullable=True),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("country", sa.String(length=128), nullable=True),
        sa.Column("last_seen", sa.Float(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("client_id"),
    )

    op.create_table(
        "rate_limit_buckets",
        sa.Column("bucket_key", sa.String(length=512), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("timestamps", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("bucket_key"),
    )

    op.create_table(
        "ha_locks",
        sa.Column("lock_name", sa.String(length=255), nullable=False),
        sa.Column("holder", sa.String(length=255), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("lock_name"),
    )


def downgrade() -> None:
    op.drop_table("ha_locks")
    op.drop_table("rate_limit_buckets")
    op.drop_table("geo_presence")
    op.drop_table("client_incentives")
    op.drop_table("client_reputation")
