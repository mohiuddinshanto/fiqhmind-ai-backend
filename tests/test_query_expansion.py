"""Tests for the Phase 9 query expansion (lexicon + term_relations KG hop)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.base import Base
from app.db.repositories import TermRelationRepository
from app.services.query_expansion import (
    FIQH_SYNONYMS,
    CandidateQuery,
    ExpandedQuery,
    ExpansionRunner,
)


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


@pytest.fixture()
def runner(session: Session) -> ExpansionRunner:
    return ExpansionRunner(session, Settings())


def _kinds(expanded: ExpandedQuery) -> list[str]:
    return [candidate.kind for candidate in expanded.candidates]


def test_expansion_keeps_original_and_canonical(runner: ExpansionRunner) -> None:
    expanded = runner.expand(
        original="ما حكم صلاة المسافر؟",
        canonical_arabic="ما حكم صلاه المسافر؟",
    )

    assert expanded.canonical_arabic == "ما حكم صلاه المسافر؟"
    assert expanded.candidates[0] == CandidateQuery(text="ما حكم صلاة المسافر؟", kind="original")
    assert expanded.candidates[1].kind == "canonical"


def test_synonym_lexicon_expands_matched_terms(runner: ExpansionRunner) -> None:
    expanded = runner.expand(original="صلاه", canonical_arabic="صلاه")

    kinds = _kinds(expanded)
    assert "synonym" in kinds
    synonyms = [c.text for c in expanded.candidates if c.kind == "synonym"]
    assert any("عباده" in text for text in synonyms)


def test_synonym_lexicon_defines_architecture_terms() -> None:
    assert "ثبوت" in FIQH_SYNONYMS and "وجوب" in FIQH_SYNONYMS["ثبوت"]
    assert "ماء" in FIQH_SYNONYMS and "طهور" in FIQH_SYNONYMS["ماء"]


def test_kg_hop_adds_related_term_variants(session: Session, runner: ExpansionRunner) -> None:
    TermRelationRepository(session).seed_fixtures()

    expanded = runner.expand(original="الطلاق", canonical_arabic="الطلاق")

    kg = [c.text for c in expanded.candidates if c.kind == "kg"]
    assert kg, "expected KG-hop candidates"
    assert any("عده" in text for text in kg)


def test_kg_hop_handles_definite_article(session: Session, runner: ExpansionRunner) -> None:
    TermRelationRepository(session).seed_fixtures()

    expanded = runner.expand(original="الزكاة", canonical_arabic="الزكاة")

    kg = [c.text for c in expanded.candidates if c.kind == "kg"]
    assert any("نصاب" in text for text in kg)


def test_expansion_bounds_candidate_count(session: Session) -> None:
    TermRelationRepository(session).seed_fixtures()
    runner = ExpansionRunner(session, Settings(retrieval_max_variants=4))

    expanded = runner.expand(original="حكم الصلاة والزكاة", canonical_arabic="حكم الصلاه والزكاه")

    assert len(expanded.candidates) <= 4


def test_expansion_deduplicates_candidate_texts(runner: ExpansionRunner) -> None:
    expanded = runner.expand(original="صلاة", canonical_arabic="صلاة")

    texts = [c.text for c in expanded.candidates]
    assert len(texts) == len(set(texts))


def test_llm_expansion_disabled_by_default(runner: ExpansionRunner) -> None:
    expanded = runner.expand(original="وضوء", canonical_arabic="وضوء")

    assert "llm" not in _kinds(expanded)


def test_llm_expansion_enabled_without_provider_raises() -> None:
    runner = ExpansionRunner(None, Settings(retrieval_llm_expansion_enabled=True))

    with pytest.raises(NotImplementedError):
        runner.expand(original="وضوء", canonical_arabic="وضوء")


def test_expansion_without_session_skips_kg_hop() -> None:
    runner = ExpansionRunner(None, Settings())

    expanded = runner.expand(original="الطلاق", canonical_arabic="الطلاق")

    assert "kg" not in _kinds(expanded)
