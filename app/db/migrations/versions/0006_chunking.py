"""Chunking: chunks job/context columns + nullable corpus refs + 'chunking' kind (Phase 7).

Revision ID: 0006_chunking
Revises: 0005_metadata
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_chunking"
down_revision: str | None = "0005_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORPHAN_KIND_CHECK = "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout', 'metadata')"
_CHUNKING_KIND_CHECK = (
    "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout', 'metadata', 'chunking')"
)


def upgrade() -> None:
    with op.batch_alter_table("chunks") as batch_op:
        batch_op.add_column(sa.Column("job_id", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("context_heading", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("order_index", sa.Integer(), nullable=False, server_default="0")
        )
        # Corpus attachment is a later phase: chunks are written before
        # books/volumes/pages are materialized, so these refs become nullable.
        batch_op.alter_column("book_id", existing_type=sa.String(length=32), nullable=True)
        batch_op.alter_column("edition_id", existing_type=sa.String(length=32), nullable=True)
        batch_op.alter_column("volume_id", existing_type=sa.String(length=32), nullable=True)
        batch_op.create_foreign_key(
            "fk_chunks_job_id_ingestion_jobs", "ingestion_jobs", ["job_id"], ["id"]
        )
        batch_op.create_index("ix_chunks_job_id", ["job_id"])

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint("ck_ingestion_jobs_kind", _CHUNKING_KIND_CHECK)


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint("ck_ingestion_jobs_kind", _ORPHAN_KIND_CHECK)

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.drop_index("ix_chunks_job_id")
        batch_op.drop_constraint("fk_chunks_job_id_ingestion_jobs", type_="foreignkey")
        batch_op.alter_column("volume_id", existing_type=sa.String(length=32), nullable=False)
        batch_op.alter_column("edition_id", existing_type=sa.String(length=32), nullable=False)
        batch_op.alter_column("book_id", existing_type=sa.String(length=32), nullable=False)
        batch_op.drop_column("order_index")
        batch_op.drop_column("context_heading")
        batch_op.drop_column("job_id")
