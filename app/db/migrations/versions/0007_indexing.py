"""Indexing: add 'indexing' to ingestion_jobs.kind CHECK (Phase 8).

Revision ID: 0007_indexing
Revises: 0006_chunking
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

from alembic import op

revision: str = "0007_indexing"
down_revision: str | None = "0006_chunking"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PRE_INDEXING_KIND_CHECK = (
    "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout', 'metadata', 'chunking')"
)
_INDEXING_KIND_CHECK = (
    "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout', 'metadata', "
    "'chunking', 'indexing')"
)


def upgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint("ck_ingestion_jobs_kind", _INDEXING_KIND_CHECK)


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint("ck_ingestion_jobs_kind", _PRE_INDEXING_KIND_CHECK)
