"""Layout: page_blocks layout columns + region/kind checks (Phase 5).

Revision ID: 0004_layout
Revises: 0003_extraction
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_layout"
down_revision: str | None = "0003_extraction"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PAGE_BLOCK_REGIONS = "region IN ('main', 'footnote', 'margin', 'header', 'footer', 'unknown')"


def upgrade() -> None:
    with op.batch_alter_table("page_blocks") as batch_op:
        batch_op.add_column(
            sa.Column("region", sa.String(length=20), nullable=False, server_default="unknown")
        )
        batch_op.add_column(
            sa.Column("reading_order", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("confidence", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(
            sa.Column("classification_reason", sa.String(length=500), nullable=True)
        )
        batch_op.create_check_constraint("ck_page_blocks_region", _PAGE_BLOCK_REGIONS)

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.drop_constraint("ck_chunks_region", type_="check")
        batch_op.create_check_constraint("ck_chunks_region", _PAGE_BLOCK_REGIONS)

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_kind",
            "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout')",
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_kind", "kind IN ('initial', 'reindex', 'ocr', 'extraction')"
        )

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.drop_constraint("ck_chunks_region", type_="check")
        batch_op.create_check_constraint(
            "ck_chunks_region", "region IN ('main', 'footnote', 'margin', 'header', 'footer')"
        )

    with op.batch_alter_table("page_blocks") as batch_op:
        batch_op.drop_constraint("ck_page_blocks_region", type_="check")
        batch_op.drop_column("classification_reason")
        batch_op.drop_column("confidence")
        batch_op.drop_column("reading_order")
        batch_op.drop_column("region")
