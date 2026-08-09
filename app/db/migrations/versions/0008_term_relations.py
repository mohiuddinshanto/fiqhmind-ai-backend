"""Term relations: flat fiqh lexicon graph table (Phase 9 retrieval expansion).

Revision ID: 0008_term_relations
Revises: 0007_indexing
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_term_relations"
down_revision: str | None = "0007_indexing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.text("CURRENT_TIMESTAMP")

RELATION_TYPES = "'synonym', 'antonym', 'hyponym', 'broader', 'narrower', 'related'"


def upgrade() -> None:
    op.create_table(
        "term_relations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("primary_term", sa.String(length=255), nullable=False),
        sa.Column("related_term", sa.String(length=255), nullable=False),
        sa.Column("relation_type", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.CheckConstraint(
            f"relation_type IN ({RELATION_TYPES})",
            name="ck_term_relations_relation_type",
        ),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_term_relations_confidence"),
        sa.UniqueConstraint(
            "primary_term", "related_term", "relation_type", name="uq_term_relations_edge"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_term_relations"),
    )
    op.create_index("ix_term_relations_primary_term", "term_relations", ["primary_term"])
    op.create_index("ix_term_relations_related_term", "term_relations", ["related_term"])


def downgrade() -> None:
    op.drop_table("term_relations")
