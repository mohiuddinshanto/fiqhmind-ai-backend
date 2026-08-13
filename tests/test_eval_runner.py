"""Tests for the Phase 17 eval runner (ARCHITECTURE §17.3).

Uses an in-memory SQLite DB and fake retrieval/generation so the runner's
replay + persistence + threshold gating are exercised without Qdrant or a
live LLM. The `resolve_expected_chunk_ids` path is tested against real Book /
Edition / Volume / Chunk rows.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Book, Edition, EvalRun, EvalRunResult, Volume
from app.schemas.chat import ChatAnswer
from app.services.eval.gold import GoldItem
from app.services.eval.runner import EvalRunner, aggregate_metrics, resolve_all_items


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    yield testing_session()
    engine.dispose()


class FakeRetrieval:
    """Returns a fixed chunk list in a fixed order (no vector store)."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    def search(self, query: str, top_n: int = 8):
        class _Result:
            chunks: list

        result = _Result()
        result.chunks = [
            type(
                "Chunk",
                (),
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "book_name": chunk.get("book_name"),
                    "volume": chunk.get("volume"),
                    "printed_page_start": chunk.get("printed_page_start"),
                },
            )()
            for chunk in self.chunks[:top_n]
        ]
        result.evidence_sufficient = True
        return result


class FakeGeneration:
    """Synthesizes an answer from the first retrieved chunk."""

    def generate(self, retrieval, *, answer_language: str = "bn"):
        chunk = retrieval.chunks[0]
        return ChatAnswer.model_validate(
            {
                "answer_language": answer_language,
                "explanation": {"html": chunk.text, "type": "markdown"},
                "arabic_quotes": [],
                "citations": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "book": chunk.book_name,
                        "volume": chunk.volume,
                        "page": str(chunk.printed_page_start) if chunk.printed_page_start else None,
                    }
                ],
                "confidence": {
                    "level": "high",
                    "retrieval_score": 0.5,
                    "source_agreement": "consensus",
                    "rationale": "test",
                },
                "refusal": None,
                "caveats": [],
                "related": [],
            }
        )


class FakeGenerationRefusing:
    """Always refuses, simulating a should-refuse item's correct behavior."""

    def generate(self, retrieval, *, answer_language: str = "bn"):
        return ChatAnswer.model_validate(
            {
                "answer_language": answer_language,
                "explanation": {"html": "I cannot answer this question.", "type": "markdown"},
                "arabic_quotes": [],
                "citations": [],
                "confidence": {
                    "level": "low",
                    "retrieval_score": 0.1,
                    "source_agreement": "unknown",
                    "rationale": "test",
                },
                "refusal": {"reason": "insufficient_evidence", "closest_evidence": []},
                "caveats": [],
                "related": [],
            }
        )


def _gold_item(
    *,
    item_id: str,
    category: str = "single_clear_ruling",
    question: str | None = None,
    expected_answer: str | None = None,
    citations: list[dict] | None = None,
    expect_refusal: bool = False,
) -> GoldItem:
    q = question or "Is water pure?"
    return GoldItem(
        id=item_id,
        category=category,
        difficulty="easy",
        question={"bn": q, "ar": q, "en": q},
        expected_citations=citations or [{"book": "Al-Hidayah", "volume": "1", "page": 5}],
        expected_answer=expected_answer or q,
        expect_refusal=expect_refusal,
    )


def _seed_corpus(session: Session) -> None:
    book = Book(title="Al-Hidayah", author="al-Marghinani", status="published")
    session.add(book)
    session.flush()
    edition = Edition(book_id=book.id, edition_number=1)
    session.add(edition)
    session.flush()
    volume = Volume(edition_id=edition.id, volume_number=1)
    session.add(volume)
    session.flush()
    session.commit()


def test_resolve_expected_chunk_ids_maps_citations(session: Session) -> None:
    from app.db.models import Chunk

    book = Book(title="Al-Mabsut", author="al-Sarakhsi", status="published")
    session.add(book)
    session.flush()
    edition = Edition(book_id=book.id, edition_number=1)
    session.add(edition)
    session.flush()
    volume = Volume(edition_id=edition.id, volume_number=1)
    session.add(volume)
    session.flush()
    chunk = Chunk(
        chunk_id="c1",
        book_id=book.id,
        volume_id=volume.id,
        printed_page_start=5,
        printed_page_end=6,
        raw_text="Water is pure in itself.",
        region="main",
        lang="ar",
    )
    session.add(chunk)
    session.commit()

    item = _gold_item(item_id="x", citations=[{"book": "Al-Mabsut", "volume": "1", "page": 5}])
    ids = resolve_all_items(session, [item])["x"]
    assert ids == ["c1"]


def test_runner_persists_run_and_results(session: Session) -> None:
    _seed_corpus(session)
    retrieval = FakeRetrieval(
        [
            {
                "chunk_id": "gold-chunk",
                "text": "Water is pure in itself and purifying for others.",
                "book_name": "Al-Hidayah",
                "volume": "1",
                "printed_page_start": 5,
            }
        ]
    )
    runner = EvalRunner(session, retrieval, FakeGeneration(), label="test-run")

    items = [_gold_item(item_id="item-1")]
    run = runner.run(items, thresholds={})

    assert isinstance(run, EvalRun)
    assert run.status in ("passed", "failed")
    assert run.label == "test-run"
    assert run.total_items == 1

    results = session.query(EvalRunResult).filter(EvalRunResult.run_id == run.id).all()
    assert len(results) == 3  # bn, ar, en
    for row in results:
        assert row.gold_item_id == "item-1"
        assert row.retrieved_chunk_ids == ["gold-chunk"]
        assert row.answer is not None


def test_runner_marks_refusal_items(session: Session) -> None:
    _seed_corpus(session)
    retrieval = FakeRetrieval(
        [
            {
                "chunk_id": "c1",
                "text": "Some unrelated corpus text.",
                "book_name": "Al-Hidayah",
                "volume": "1",
                "printed_page_start": 5,
            }
        ]
    )
    runner = EvalRunner(session, retrieval, FakeGenerationRefusing(), label="test-run")
    items = [
        _gold_item(
            item_id="refuse-1",
            category="should_refuse",
            citations=[],
            expect_refusal=True,
        )
    ]
    run = runner.run(items, thresholds={})
    results = session.query(EvalRunResult).filter(EvalRunResult.run_id == run.id).all()
    assert len(results) == 3
    assert all(row.refusal_given for row in results)
    assert all(row.expect_refusal for row in results)


def test_threshold_failure_marks_run_failed(session: Session) -> None:
    _seed_corpus(session)
    retrieval = FakeRetrieval(
        [
            {
                "chunk_id": "c1",
                "text": "Water is pure in itself.",
                "book_name": "Al-Hidayah",
                "volume": "1",
                "printed_page_start": 5,
            }
        ]
    )
    runner = EvalRunner(session, retrieval, FakeGeneration(), label="test-run")
    items = [_gold_item(item_id="item-1")]
    run = runner.run(items, thresholds={"recall@10": 0.99})
    assert run.status == "failed"
    assert run.failures
    assert any("recall@10" in failure for failure in run.failures)


def test_runner_handles_replay_error_gracefully(session: Session) -> None:
    _seed_corpus(session)

    class ExplodingRetrieval:
        def search(self, query: str, top_n: int = 8):
            raise RuntimeError("vector store down")

    runner = EvalRunner(session, ExplodingRetrieval(), FakeGeneration(), label="test-run")
    items = [_gold_item(item_id="item-1")]
    run = runner.run(items, thresholds={})
    results = session.query(EvalRunResult).filter(EvalRunResult.run_id == run.id).all()
    assert len(results) == 3
    assert all(row.error is not None for row in results)
    assert all(row.retrieved_chunk_ids == [] for row in results)


def test_aggregate_metrics_mean() -> None:
    per_item = [
        {
            "recall@5": 1.0,
            "recall@10": 1.0,
            "recall@20": 1.0,
            "mrr": 1.0,
            "ndcg@10": 1.0,
            "citation_accuracy": 1.0,
            "groundedness": 1.0,
            "hallucination_rate": 0.0,
            "answer_accuracy": 2,
        },
        {
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 1.0,
            "mrr": 0.0,
            "ndcg@10": 0.0,
            "citation_accuracy": 0.0,
            "groundedness": 0.5,
            "hallucination_rate": 0.5,
            "answer_accuracy": 0,
        },
    ]
    aggregate = aggregate_metrics(per_item, [], [], [])
    assert aggregate.recall_at_5 == 0.5
    assert aggregate.recall_at_10 == 0.5
    assert aggregate.recall_at_20 == 1.0
    assert aggregate.mrr == 0.5
    assert aggregate.groundedness == 0.75
    assert aggregate.hallucination_rate == 0.25
    assert aggregate.answer_accuracy == 1.0
    assert aggregate.refusal_precision == 1.0
    assert aggregate.refusal_recall == 1.0
