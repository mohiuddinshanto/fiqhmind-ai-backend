"""Metadata extraction tables + 'metadata' job kind (Phase 6).

Revision ID: 0005_metadata
Revises: 0004_layout
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_metadata"
down_revision: str | None = "0004_layout"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "metadata_documents",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("job_id", sa.String(length=32), nullable=False),
        sa.Column("upload_id", sa.String(length=32), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("numbering_system", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["job_id"], ["ingestion_jobs.id"]),
        sa.ForeignKeyConstraint(["upload_id"], ["uploads.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", name="uq_metadata_documents_job"),
    )
    op.create_index("ix_metadata_documents_job_id", "metadata_documents", ["job_id"], unique=False)
    op.create_index(
        "ix_metadata_documents_upload_id", "metadata_documents", ["upload_id"], unique=False
    )

    op.create_table(
        "metadata_fields",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("document_id", sa.String(length=32), nullable=False),
        sa.Column("field", sa.String(length=50), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_metadata_fields_confidence"),
        sa.ForeignKeyConstraint(["document_id"], ["metadata_documents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "field", name="uq_metadata_fields_document_field"),
    )
    op.create_index(
        "ix_metadata_fields_document_id", "metadata_fields", ["document_id"], unique=False
    )

    op.create_table(
        "page_metadata",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("document_id", sa.String(length=32), nullable=False),
        sa.Column("pdf_page", sa.Integer(), nullable=False),
        sa.Column("printed_page", sa.String(length=255), nullable=False),
        sa.Column("printed_page_numeric", sa.Integer(), nullable=True),
        sa.Column("numbering_system", sa.String(length=20), nullable=False),
        sa.Column("page_number_uncertain", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("kitab", sa.String(length=255), nullable=True),
        sa.Column("bab", sa.String(length=255), nullable=True),
        sa.Column("fasl", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_page_metadata_confidence"),
        sa.ForeignKeyConstraint(["document_id"], ["metadata_documents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "pdf_page", name="uq_page_metadata_document_page"),
    )
    op.create_index("ix_page_metadata_document_id", "page_metadata", ["document_id"], unique=False)

    op.create_table(
        "metadata_structures",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("document_id", sa.String(length=32), nullable=False),
        sa.Column("level", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "level IN ('kitab', 'bab', 'fasl', 'topic')", name="ck_metadata_structures_level"
        ),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_metadata_structures_confidence"),
        sa.ForeignKeyConstraint(["document_id"], ["metadata_documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_metadata_structures_document_id", "metadata_structures", ["document_id"], unique=False
    )
    op.create_index(
        "ix_metadata_structures_doc_level_start",
        "metadata_structures",
        ["document_id", "level", "page_start"],
        unique=False,
    )

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_kind",
            "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout', 'metadata')",
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_kind",
            "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout')",
        )

    op.drop_index("ix_metadata_structures_doc_level_start", table_name="metadata_structures")
    op.drop_index("ix_metadata_structures_document_id", table_name="metadata_structures")
    op.drop_table("metadata_structures")
    op.drop_index("ix_page_metadata_document_id", table_name="page_metadata")
    op.drop_table("page_metadata")
    op.drop_index("ix_metadata_fields_document_id", table_name="metadata_fields")
    op.drop_table("metadata_fields")
    op.drop_index("ix_metadata_documents_upload_id", table_name="metadata_documents")
    op.drop_index("ix_metadata_documents_job_id", table_name="metadata_documents")
    op.drop_table("metadata_documents")
