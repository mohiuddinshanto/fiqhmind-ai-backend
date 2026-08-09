"""Uploads: PDF upload records, upload logs, and ingestion_jobs upload linkage.

Revision ID: 0002_uploads
Revises: 0001_initial
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_uploads"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.text("CURRENT_TIMESTAMP")

INGESTION_JOB_STATUSES = (
    "'uploaded', 'queued', 'processing', 'sanitizing', 'extracting', 'ocr', "
    "'ocr_correcting', 'structuring', 'chunking', 'embedding', 'indexed', "
    "'completed', 'failed'"
)


def upgrade() -> None:
    op.create_table(
        "uploads",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("size", sa.Integer(), nullable=True),
        sa.Column("mime", sa.String(length=100), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("storage_path", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("received_bytes", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('uploading', 'uploaded', 'queued', 'processing', 'completed', 'failed')",
            name="ck_uploads_status",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_uploads"),
    )
    op.create_index("ix_uploads_sha256", "uploads", ["sha256"], unique=True)
    op.create_index("ix_uploads_status", "uploads", ["status"])

    op.create_table(
        "upload_logs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("upload_id", sa.String(length=32), nullable=False),
        sa.Column("event", sa.String(length=50), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["upload_id"], ["uploads.id"], name="fk_upload_logs_upload_id_uploads"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_upload_logs"),
    )
    op.create_index("ix_upload_logs_upload_id", "upload_logs", ["upload_id"])

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.add_column(sa.Column("upload_id", sa.String(length=32), nullable=True))
        batch_op.create_foreign_key(
            "fk_ingestion_jobs_upload_id_uploads", "uploads", ["upload_id"], ["id"]
        )
        batch_op.alter_column("book_id", existing_type=sa.String(length=32), nullable=True)
        batch_op.create_index("ix_ingestion_jobs_upload_id", ["upload_id"])
        batch_op.drop_constraint("ck_ingestion_jobs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_status", f"status IN ({INGESTION_JOB_STATUSES})"
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_status",
            "status IN ('uploaded', 'sanitizing', 'extracting', 'ocr', 'ocr_correcting', "
            "'structuring', 'chunking', 'embedding', 'indexed', 'failed')",
        )
        batch_op.drop_index("ix_ingestion_jobs_upload_id")
        batch_op.drop_constraint("fk_ingestion_jobs_upload_id_uploads", type_="foreignkey")
        batch_op.drop_column("upload_id")
        batch_op.alter_column("book_id", existing_type=sa.String(length=32), nullable=False)

    op.drop_table("upload_logs")
    op.drop_table("uploads")
