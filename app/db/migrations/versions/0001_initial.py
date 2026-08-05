"""Initial schema: users, corpus (books -> editions -> volumes -> pages -> chunks), ingestion, user state, audit.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-05

"""
# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "roles",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_roles"),
    )
    op.create_index("ix_roles_name", "roles", ["name"], unique=True)

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("role_id", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], name="fk_users_role_id_roles"),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role_id", "users", ["role_id"])

    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("refresh_token_hash", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_sessions_user_id_users"),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
    )
    op.create_index("ix_sessions_refresh_token_hash", "sessions", ["refresh_token_hash"])
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "books",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("title_transliteration", sa.String(length=255), nullable=True),
        sa.Column("author", sa.String(length=255), nullable=False),
        sa.Column("language", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "status IN ('draft', 'publishing', 'published')",
            name="ck_books_status",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_books"),
        sa.UniqueConstraint("title", "author", name="uq_books_title_author"),
    )
    op.create_index("ix_books_title", "books", ["title"])

    op.create_table(
        "editions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("book_id", sa.String(length=32), nullable=False),
        sa.Column("publisher", sa.String(length=255), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("muhaqqiq", sa.String(length=255), nullable=True),
        sa.Column("edition_number", sa.Integer(), nullable=False),
        sa.Column("isbn", sa.String(length=32), nullable=True),
        sa.Column("language", sa.String(length=10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], name="fk_editions_book_id_books"),
        sa.PrimaryKeyConstraint("id", name="pk_editions"),
        sa.UniqueConstraint("book_id", "edition_number", name="uq_editions_book_number"),
    )
    op.create_index("ix_editions_book_id", "editions", ["book_id"])
    op.create_index("ix_editions_isbn", "editions", ["isbn"])

    op.create_table(
        "volumes",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("edition_id", sa.String(length=32), nullable=False),
        sa.Column("volume_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("printed_page_start", sa.Integer(), nullable=True),
        sa.Column("printed_page_end", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["edition_id"], ["editions.id"], name="fk_volumes_edition_id_editions"),
        sa.PrimaryKeyConstraint("id", name="pk_volumes"),
        sa.UniqueConstraint("edition_id", "volume_number", name="uq_volumes_edition_number"),
    )
    op.create_index("ix_volumes_edition_id", "volumes", ["edition_id"])

    op.create_table(
        "pages",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("volume_id", sa.String(length=32), nullable=False),
        sa.Column("printed_page", sa.Integer(), nullable=False),
        sa.Column("pdf_page", sa.Integer(), nullable=False),
        sa.Column("arabic_char_count", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("page_number_uncertain", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'extracted', 'review', 'verified')",
            name="ck_pages_status",
        ),
        sa.ForeignKeyConstraint(["volume_id"], ["volumes.id"], name="fk_pages_volume_id_volumes"),
        sa.PrimaryKeyConstraint("id", name="pk_pages"),
        sa.UniqueConstraint("volume_id", "printed_page", name="uq_pages_volume_page"),
    )
    op.create_index("ix_pages_volume_id", "pages", ["volume_id"])

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("book_id", sa.String(length=32), nullable=False),
        sa.Column("edition_id", sa.String(length=32), nullable=False),
        sa.Column("volume_id", sa.String(length=32), nullable=False),
        sa.Column("page_start_id", sa.String(length=32), nullable=True),
        sa.Column("page_end_id", sa.String(length=32), nullable=True),
        sa.Column("printed_page_start", sa.Integer(), nullable=True),
        sa.Column("printed_page_end", sa.Integer(), nullable=True),
        sa.Column("pdf_page_start", sa.Integer(), nullable=True),
        sa.Column("pdf_page_end", sa.Integer(), nullable=True),
        sa.Column("kitab", sa.String(length=255), nullable=True),
        sa.Column("bab", sa.String(length=255), nullable=True),
        sa.Column("fasl", sa.String(length=255), nullable=True),
        sa.Column("topic", sa.String(length=255), nullable=True),
        sa.Column("region", sa.String(length=20), nullable=False),
        sa.Column("lang", sa.String(length=10), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.CheckConstraint(
            "region IN ('main', 'footnote', 'margin', 'header', 'footer')",
            name="ck_chunks_region",
        ),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], name="fk_chunks_book_id_books"),
        sa.ForeignKeyConstraint(["edition_id"], ["editions.id"], name="fk_chunks_edition_id_editions"),
        sa.ForeignKeyConstraint(["page_end_id"], ["pages.id"], name="fk_chunks_page_end_id_pages"),
        sa.ForeignKeyConstraint(["page_start_id"], ["pages.id"], name="fk_chunks_page_start_id_pages"),
        sa.ForeignKeyConstraint(["volume_id"], ["volumes.id"], name="fk_chunks_volume_id_volumes"),
        sa.PrimaryKeyConstraint("chunk_id", name="pk_chunks"),
    )
    op.create_index("ix_chunks_book_id", "chunks", ["book_id"])
    op.create_index("ix_chunks_edition_id", "chunks", ["edition_id"])
    op.create_index("ix_chunks_page_end_id", "chunks", ["page_end_id"])
    op.create_index("ix_chunks_page_start_id", "chunks", ["page_start_id"])
    op.create_index("ix_chunks_volume_id", "chunks", ["volume_id"])
    op.create_index("ix_chunks_book_region_verified", "chunks", ["book_id", "region", "verified"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("book_id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("current_step", sa.String(length=30), nullable=True),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.CheckConstraint("kind IN ('initial', 'reindex', 'ocr')", name="ck_ingestion_jobs_kind"),
        sa.CheckConstraint(
            "status IN ('uploaded', 'sanitizing', 'extracting', 'ocr', 'ocr_correcting', "
            "'structuring', 'chunking', 'embedding', 'indexed', 'failed')",
            name="ck_ingestion_jobs_status",
        ),
        sa.CheckConstraint("progress_percent BETWEEN 0 AND 100", name="ck_ingestion_jobs_progress_range"),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], name="fk_ingestion_jobs_book_id_books"),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_jobs"),
    )
    op.create_index("ix_ingestion_jobs_book_id", "ingestion_jobs", ["book_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])

    op.create_table(
        "ingestion_errors",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("job_id", sa.String(length=32), nullable=False),
        sa.Column("step", sa.String(length=50), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["ingestion_jobs.id"], name="fk_ingestion_errors_job_id_ingestion_jobs"),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_errors"),
    )
    op.create_index("ix_ingestion_errors_job_id", "ingestion_errors", ["job_id"])

    op.create_table(
        "bookmarks",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("topic", sa.String(length=255), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.chunk_id"], name="fk_bookmarks_chunk_id_chunks"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_bookmarks_user_id_users"),
        sa.PrimaryKeyConstraint("id", name="pk_bookmarks"),
        sa.UniqueConstraint("user_id", "chunk_id", name="uq_bookmarks_user_chunk"),
    )
    op.create_index("ix_bookmarks_user_id", "bookmarks", ["user_id"])

    op.create_table(
        "chat_history",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=True),
        sa.Column("answer_language", sa.String(length=10), nullable=False),
        sa.Column("answer", sa.JSON(), nullable=True),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.String(length=10), nullable=True),
        sa.Column("refusal", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_chat_history_user_id_users"),
        sa.PrimaryKeyConstraint("id", name="pk_chat_history"),
    )
    op.create_index("ix_chat_history_user_id", "chat_history", ["user_id"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=True),
        sa.Column("actor_type", sa.String(length=30), nullable=False),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_audit_log_user_id_users"),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_resource_type", "audit_log", ["resource_type"])
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("chat_history")
    op.drop_table("bookmarks")
    op.drop_table("ingestion_errors")
    op.drop_table("ingestion_jobs")
    op.drop_table("chunks")
    op.drop_table("pages")
    op.drop_table("volumes")
    op.drop_table("editions")
    op.drop_table("books")
    op.drop_table("sessions")
    op.drop_table("users")
    op.drop_table("roles")
