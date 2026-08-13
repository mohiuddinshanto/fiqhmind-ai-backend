"""Phase 17 eval runner (ARCHITECTURE §17.3): replay gold items through the
live retrieval + generation pipeline and persist results to Postgres.

Not a mocked shortcut: every gold item (in each of its three language
variants) is replayed through the actual `RetrievalRunner.search` and
`GenerationService.generate`, the results are scored with the §17.2 metrics,
persisted (`eval_runs`, `eval_run_results`), and compared against
`backend/eval/thresholds.yaml` for a regression pass/fail verdict.

CLI:
    python -m app.services.eval.runner --gold eval/gold_qa.jsonl --thresholds eval/thresholds.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import structlog
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.postgres import get_session_factory
from app.db.models import Book, EvalRun, EvalRunResult
from app.db.repositories.eval import EvalRunRepository
from app.services.eval import metrics as m
from app.services.eval.gold import GoldItem, load_gold_set
from app.services.eval.judge import get_answer_judge
from app.services.eval.resolve import resolve_expected_chunk_ids
from app.services.generation.providers import get_llm_provider
from app.services.generation.service import GenerationService
from app.services.retrieval import RetrievalRunner

logger = structlog.get_logger(__name__)

DEFAULT_GOLD_PATH = Path(__file__).resolve().parents[3] / "eval" / "gold_qa.jsonl"
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "eval" / "thresholds.yaml"

# Languages to replay per gold item (§17.1: three variants test cross-lingual retrieval).
LANGUAGES = ("bn", "ar", "en")


def _load_thresholds(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"thresholds not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    thresholds = payload.get("metrics", payload)
    return {key: float(value) for key, value in thresholds.items()}


def _default_thresholds() -> dict:
    return {
        "recall@10": 0.8,
        "mrr": 0.6,
        "ndcg@10": 0.7,
        "citation_accuracy": 0.9,
        "groundedness": 0.8,
        "hallucination_rate": 0.05,
        "refusal_precision": 0.8,
        "refusal_recall": 0.8,
    }


def _answer_text(answer) -> str:
    """Extract the prose explanation from a ChatAnswer (or dict)."""
    if answer is None:
        return ""
    if hasattr(answer, "model_dump"):
        answer = answer.model_dump(mode="json")
    explanation = answer.get("explanation", {})
    if isinstance(explanation, dict):
        return explanation.get("html") or explanation.get("text") or ""
    return str(explanation)


def _answer_citations(answer) -> list[dict]:
    """Return the answer's citations as plain dicts (book/volume/page/chunk_id)."""
    if answer is None:
        return []
    if hasattr(answer, "model_dump"):
        answer = answer.model_dump(mode="json")
    citations = answer.get("citations", []) or []
    return [
        {
            "book": citation.get("book"),
            "volume": citation.get("volume"),
            "page": citation.get("page"),
            "chunk_id": citation.get("chunk_id"),
        }
        for citation in citations
        if isinstance(citation, dict)
    ]


def _gold_citation_dicts(item: GoldItem) -> list[dict]:
    return [
        {"book": c.book, "volume": c.volume, "page": c.page} for c in item.expected_citations
    ]


def _resolve_all_items(session: Session, items: list[GoldItem]) -> dict[str, list[str]]:
    """Resolve each item's citations to gold chunk ids once per run."""
    return {item.id: resolve_expected_chunk_ids(item, session) for item in items}


def _ensure_book_rows(session: Session, items: list[GoldItem]) -> None:
    """Best-effort: materialize gold-referenced books so citation resolution works.

    The eval harness runs against whatever the corpus already contains (books
    are created by the ingestion pipeline). This step only logs when a gold
    book is missing, so a gold set can be authored ahead of ingestion.
    """
    titles = {c.book for item in items for c in item.expected_citations if c.book}
    for title in titles:
        existing = session.scalar(select(Book).where(Book.title == title))
        if existing is None:
            logger.warning("gold_book_not_in_corpus", book=title)


class EvalRunner:
    """Replays a gold set through the live pipeline and persists a scored run."""

    def __init__(
        self,
        session: Session,
        retrieval: Any,
        generation: Any,
        *,
        judge: Any = None,
        label: str = "manual",
    ) -> None:
        self._session = session
        self._retrieval = retrieval
        self._generation = generation
        self._judge = judge or get_answer_judge(get_llm_provider())
        self._label = label

    def run(self, items: list[GoldItem], *, thresholds: dict | None = None) -> EvalRun:
        thresholds = thresholds or _default_thresholds()
        gold_ids = resolve_all_items(self._session, items)
        repo = EvalRunRepository(self._session)
        run = repo.create_run(label=self._label, total_items=len(items))

        per_item_metrics: list[dict] = []
        refusal_pairs: list[tuple[bool, bool]] = []
        latencies: list[float] = []
        retrieval_contributors: list[tuple[list[str], set[str]]] = []

        for item in items:
            expected_ids = gold_ids[item.id]
            expected_set = set(expected_ids)
            for language in LANGUAGES:
                question = getattr(item.question, language)
                result = self._replay_one(item, language, question)
                refusal_pairs.append((item.expect_refusal, result["refusal_given"]))
                if result["latency_ms"] is not None:
                    latencies.append(result["latency_ms"])

                item_metrics = {
                    "recall@5": m.recall_at_k(result["retrieved_ids"], expected_set, 5),
                    "recall@10": m.recall_at_k(result["retrieved_ids"], expected_set, 10),
                    "recall@20": m.recall_at_k(result["retrieved_ids"], expected_set, 20),
                    "mrr": m.mrr(result["retrieved_ids"], expected_set),
                    "ndcg@10": m.ndcg_at_k(result["retrieved_ids"], expected_set, 10),
                    "citation_accuracy": m.citation_accuracy(
                        result["citations"], _gold_citation_dicts(item)
                    ),
                    "groundedness": m.groundedness(result["explanation"], result["cited_texts"]),
                    "hallucination_rate": m.hallucination_rate(
                        result["explanation"], result["evidence_texts"]
                    ),
                    "answer_accuracy": self._judge.score(
                        question=question,
                        explanation=result["explanation"],
                        expected_answer=item.expected_answer or "",
                    ),
                }
                per_item_metrics.append(item_metrics)
                retrieval_contributors.append((result["retrieved_ids"], expected_set))

                row = EvalRunResult(
                    gold_item_id=item.id,
                    language=language,
                    question=question,
                    category=item.category,
                    difficulty=item.difficulty,
                    expected_citations=_gold_citation_dicts(item),
                    expected_answer=item.expected_answer,
                    expect_refusal=item.expect_refusal,
                    retrieved_chunk_ids=result["retrieved_ids"],
                    answer=result["answer_dict"],
                    refusal_given=result["refusal_given"],
                    metrics=item_metrics,
                    latency_ms=result["latency_ms"],
                    error=result["error"],
                )
                repo.add_result(run.id, row)

        aggregate = aggregate_metrics(
            per_item_metrics, refusal_pairs, latencies, retrieval_contributors
        )
        metrics_dict = aggregate.to_dict()
        failures = check_thresholds(metrics_dict, thresholds)
        return repo.complete_run(
            run,
            metrics=metrics_dict,
            thresholds=thresholds,
            failures=failures,
        )

    def _replay_one(self, item: GoldItem, language: str, question: str) -> dict:
        """Run live retrieval + generation for one item/language and collect raw artifacts."""
        start = time.perf_counter()
        result: dict[str, Any] = {
            "retrieved_ids": [],
            "evidence_texts": [],
            "cited_texts": [],
            "citations": [],
            "explanation": "",
            "answer_dict": None,
            "refusal_given": False,
            "latency_ms": None,
            "error": None,
        }
        try:
            retrieval = self._retrieval.search(question, top_n=20)
            elapsed = time.perf_counter() - start
            result["latency_ms"] = round(elapsed * 1000, 2)
            result["retrieved_ids"] = [c.chunk_id for c in retrieval.chunks]
            result["evidence_texts"] = [c.text for c in retrieval.chunks]

            answer = self._generation.generate(retrieval, answer_language=language)
            answer_dict = answer.model_dump(mode="json")
            result["explanation"] = _answer_text(answer)
            result["citations"] = _answer_citations(answer)
            result["answer_dict"] = answer_dict
            result["refusal_given"] = answer_dict.get("refusal") is not None

            cited_ids = {
                citation.get("chunk_id")
                for citation in result["citations"]
                if citation.get("chunk_id")
            }
            result["cited_texts"] = [
                c.text for c in retrieval.chunks if c.chunk_id in cited_ids
            ]
        except Exception as exc:  # noqa: BLE001 - one bad item must not sink the run
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("eval_replay_failed", gold_item_id=item.id, language=language)
        return result


def resolve_all_items(session: Session, items: list[GoldItem]) -> dict[str, list[str]]:
    """Public helper: resolve every item's gold chunk ids (injectable for tests)."""
    return _resolve_all_items(session, items)


def aggregate_metrics(
    per_item_metrics: list[dict],
    refusal_pairs: list[tuple[bool, bool]],
    latencies: list[float],
    retrieval_contributors: list[tuple[list[str], set[str]]],
) -> m.AggregateMetrics:
    """Roll item-level scores into the run-level §17.2 aggregates."""
    if not per_item_metrics:
        return m.AggregateMetrics()

    def _mean(key: str) -> float:
        return sum(row[key] for row in per_item_metrics) / len(per_item_metrics)

    matrix = m.refusal_confusion(refusal_pairs)
    latency = m.latency_percentiles(latencies)
    answer_scores = [row["answer_accuracy"] for row in per_item_metrics]
    return m.AggregateMetrics(
        recall_at_5=_mean("recall@5"),
        recall_at_10=_mean("recall@10"),
        recall_at_20=_mean("recall@20"),
        mrr=_mean("mrr"),
        ndcg_at_10=_mean("ndcg@10"),
        citation_accuracy=_mean("citation_accuracy"),
        groundedness=_mean("groundedness"),
        hallucination_rate=_mean("hallucination_rate"),
        refusal_precision=matrix.precision,
        refusal_recall=matrix.recall,
        answer_accuracy=sum(answer_scores) / len(answer_scores),
        latency_ms=latency,
    )


def check_thresholds(metrics_dict: dict, thresholds: dict) -> list[str]:
    """Return a list of threshold violations (empty list means the run passes).

    Higher-is-better metrics must be >= their floor; `hallucination_rate` is a
    ceiling (<=). ARCHITECTURE §17.3 regression gating.
    """
    violations: list[str] = []
    for key, floor in thresholds.items():
        actual = metrics_dict.get(key)
        if actual is None:
            violations.append(f"{key}: missing from metrics")
            continue
        if key == "hallucination_rate":
            if actual > floor:
                violations.append(f"{key}: {actual} > ceiling {floor}")
        elif actual < floor:
            violations.append(f"{key}: {actual} < floor {floor}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="FiqhMind Phase 17 eval runner")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--label", default="manual")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the gold set + thresholds only (no live replay). "
        "Used by the CI regression gate when no indexed corpus/stack is present.",
    )
    args = parser.parse_args()

    items = load_gold_set(args.gold)
    thresholds = _load_thresholds(args.thresholds)

    if args.check:
        for item in items:
            for language in LANGUAGES:
                assert getattr(item.question, language).strip(), f"{item.id} missing {language}"
        logger.info(
            "eval_check_ok",
            gold_items=len(items),
            thresholds=thresholds,
        )
        print(f"Gold set OK: {len(items)} items, {len(thresholds)} threshold metrics")
        return 0

    settings: Settings = get_settings()
    session_factory = get_session_factory()
    with session_factory() as session:
        _ensure_book_rows(session, items)
        from app.core.qdrant import QdrantStore, create_qdrant_client

        store = QdrantStore(create_qdrant_client(settings), collection=settings.qdrant_collection)
        retrieval = RetrievalRunner(session, store)
        generation = GenerationService(settings=settings)
        runner = EvalRunner(session, retrieval, generation, label=args.label)
        run = runner.run(items, thresholds=thresholds)

        logger.info(
            "eval_run_completed",
            run_id=run.id,
            status=run.status,
            metrics=run.metrics,
            failures=run.failures,
        )
        print(f"\nEval run {run.id[:8]} → {run.status}")
        for key, value in (run.metrics or {}).items():
            print(f"  {key}: {value}")
        if run.failures:
            print("\nThreshold violations:")
            for failure in run.failures:
                print(f"  - {failure}")
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
