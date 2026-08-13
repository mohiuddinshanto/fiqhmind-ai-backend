from __future__ import annotations

from sqlalchemy import func, select

from app.db.models import EvalRun, EvalRunResult
from app.db.repositories.base import RepositoryBase


class EvalRunRepository(RepositoryBase[EvalRun]):
    """Persistence for Phase 17 eval runs (aggregate snapshot + per-item rows)."""

    model = EvalRun

    def create_run(
        self,
        *,
        label: str = "manual",
        git_sha: str | None = None,
        gold_set_version: str | None = None,
        total_items: int = 0,
    ) -> EvalRun:
        return self.create(
            EvalRun(
                label=label,
                git_sha=git_sha,
                status="running",
                gold_set_version=gold_set_version,
                total_items=total_items,
            )
        )

    def complete_run(
        self,
        run: EvalRun,
        *,
        metrics: dict,
        thresholds: dict,
        failures: list[str],
    ) -> EvalRun:
        run.metrics = metrics
        run.thresholds = thresholds
        run.failures = failures
        run.status = "failed" if failures else "passed"
        run.completed_at = func.now()
        return self.update(run)

    def add_result(self, run_id: str, result: EvalRunResult) -> EvalRunResult:
        result.run_id = run_id
        self._session.add(result)
        self._session.commit()
        self._session.refresh(result)
        return result

    def mark_error(self, run: EvalRun, *, error: str) -> EvalRun:
        run.status = "error"
        run.failures = [error]
        run.completed_at = func.now()
        return self.update(run)

    def latest_runs(self, limit: int = 10) -> list[EvalRun]:
        return list(
            self._session.scalars(
                select(EvalRun).order_by(EvalRun.started_at.desc()).limit(limit)
            )
        )

    def result_for(self, run_id: str, gold_item_id: str, language: str) -> EvalRunResult | None:
        return self._session.scalar(
            select(EvalRunResult).where(
                EvalRunResult.run_id == run_id,
                EvalRunResult.gold_item_id == gold_item_id,
                EvalRunResult.language == language,
            )
        )
