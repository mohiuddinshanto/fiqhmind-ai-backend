"""Extraction: PDF page/block/span/image/drawing persistence (Phase 4).

Revision ID: 0003_extraction
Revises: 0002_uploads
Create Date: 2026-08-06

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_extraction"
down_revision: str | None = "0002_uploads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "page_extractions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("job_id", sa.String(length=32), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("rotation", sa.Integer(), nullable=False),
        sa.Column("has_text", sa.Boolean(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("block_count", sa.Integer(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("drawing_count", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["ingestion_jobs.id"], name="fk_page_extractions_job_id_ingestion_jobs"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_page_extractions"),
        sa.UniqueConstraint("job_id", "page_number", name="uq_page_extractions_job_page"),
    )
    op.create_index("ix_page_extractions_job_id", "page_extractions", ["job_id"])

    op.create_table(
        "page_blocks",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("page_extraction_id", sa.String(length=32), nullable=False),
        sa.Column("block_index", sa.Integer(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("font", sa.String(length=255), nullable=True),
        sa.Column("font_size", sa.Float(), nullable=True),
        sa.Column("span_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["page_extraction_id"],
            ["page_extractions.id"],
            name="fk_page_blocks_page_extraction_id_page_extractions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_page_blocks"),
        sa.UniqueConstraint("page_extraction_id", "block_index", name="uq_page_blocks_page_index"),
    )
    op.create_index("ix_page_blocks_page_extraction_id", "page_blocks", ["page_extraction_id"])

    op.create_table(
        "page_spans",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("block_id", sa.String(length=32), nullable=False),
        sa.Column("span_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("font", sa.String(length=255), nullable=False),
        sa.Column("font_size", sa.Float(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("flags", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["block_id"], ["page_blocks.id"], name="fk_page_spans_block_id_page_blocks"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_page_spans"),
        sa.UniqueConstraint("block_id", "span_index", name="uq_page_spans_block_index"),
    )
    op.create_index("ix_page_spans_block_id", "page_spans", ["block_id"])

    op.create_table(
        "page_images",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("page_extraction_id", sa.String(length=32), nullable=False),
        sa.Column("image_index", sa.Integer(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("xref", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["page_extraction_id"],
            ["page_extractions.id"],
            name="fk_page_images_page_extraction_id_page_extractions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_page_images"),
        sa.UniqueConstraint("page_extraction_id", "image_index", name="uq_page_images_page_index"),
    )
    op.create_index("ix_page_images_page_extraction_id", "page_images", ["page_extraction_id"])

    op.create_table(
        "page_drawings",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("page_extraction_id", sa.String(length=32), nullable=False),
        sa.Column("drawing_index", sa.Integer(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("kind", sa.String(length=10), nullable=True),
        sa.Column("stroke_width", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["page_extraction_id"],
            ["page_extractions.id"],
            name="fk_page_drawings_page_extraction_id_page_extractions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_page_drawings"),
        sa.UniqueConstraint(
            "page_extraction_id", "drawing_index", name="uq_page_drawings_page_index"
        ),
    )
    op.create_index("ix_page_drawings_page_extraction_id", "page_drawings", ["page_extraction_id"])

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_kind", "kind IN ('initial', 'reindex', 'ocr', 'extraction')"
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_kind", "kind IN ('initial', 'reindex', 'ocr')"
        )

    op.drop_table("page_drawings")
    op.drop_table("page_images")
    op.drop_table("page_spans")
    op.drop_table("page_blocks")
    op.drop_table("page_extractions")
