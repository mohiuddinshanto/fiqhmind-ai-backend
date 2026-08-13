"""Eval harness persistence: eval_runs + eval_run_results (Phase 17).

Revision ID: 0009_eval
Revises: 0008_term_relations
Create Date: 2026-08-13

"""

# ruff: noqa: E501  (migration DDL rows are intentionally long)
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_eval"
down_revision: str | None = "0008_term_relations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("git_sha", sa.String(length=40), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("gold_set_version", sa.String(length=20), nullable=True),
        sa.Column("total_items", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("thresholds", sa.JSON(), nullable=True),
        sa.Column("failures", sa.JSON(), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'passed', 'failed', 'error')",
            name="ck_eval_runs_status",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_eval_runs"),
    )

    op.create_table(
        "eval_run_results",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("gold_item_id", sa.String(length=120), nullable=False),
        sa.Column("language", sa.String(length=10), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("difficulty", sa.String(length=20), nullable=False),
        sa.Column("expected_citations", sa.JSON(), nullable=True),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("expect_refusal", sa.Boolean(), nullable=False),
        sa.Column("retrieved_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("answer", sa.JSON(), nullable=True),
        sa.Column("refusal_given", sa.Boolean(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=TIMESTAMP, nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["eval_runs.id"], name="fk_eval_run_results_run"),
        sa.PrimaryKeyConstraint("id", name="pk_eval_run_results"),
    )
    op.create_index("ix_eval_run_results_run_id", "eval_run_results", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_eval_run_results_run_id", table_name="eval_run_results")
    op.drop_table("eval_run_results")
    op.drop_table("eval_runs")
