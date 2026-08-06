"""Tests for the Phase 9 term_relations repository (lexicon graph)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.repositories import TermRelationRepository


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
def repo(session: Session) -> TermRelationRepository:
    return TermRelationRepository(session)


def test_upsert_creates_a_directed_edge(repo: TermRelationRepository) -> None:
    edge = repo.upsert(primary_term="طلاق", related_term="عدة", relation_type="related")

    assert edge.id
    assert edge.primary_term == "طلاق"
    assert edge.related_term == "عدة"
    assert edge.relation_type == "related"
    assert edge.confidence == 1.0


def test_upsert_is_idempotent_and_updates_confidence(repo: TermRelationRepository) -> None:
    repo.upsert(primary_term="صلاة", related_term="عبادة", confidence=0.8)
    updated = repo.upsert(primary_term="صلاة", related_term="عبادة", confidence=0.95)

    rows = repo.get_multi()
    assert len(rows) == 1
    assert updated.confidence == pytest.approx(0.95)


def test_duplicate_edge_violates_unique_constraint(session: Session) -> None:
    repo = TermRelationRepository(session)
    repo.upsert(primary_term="زكاة", related_term="نصاب", relation_type="related")
    session.commit()

    repo.upsert(primary_term="زكاة", related_term="نصاب", relation_type="related")
    session.commit()
    assert len(repo.get_multi()) == 1


def test_upsert_rejects_invalid_relation_type(session: Session) -> None:
    repo = TermRelationRepository(session)
    with pytest.raises(IntegrityError):
        repo.upsert(primary_term="a", related_term="b", relation_type="bogus")


def test_related_terms_looks_up_both_directions(repo: TermRelationRepository) -> None:
    repo.upsert(primary_term="طلاق", related_term="عدة", relation_type="related")
    repo.upsert(primary_term="نكاح", related_term="طلاق", relation_type="related")

    neighbors = {
        edge.related_term if edge.primary_term == "طلاق" else edge.primary_term
        for edge in repo.related_terms("طلاق")
    }
    assert {"عدة", "نكاح"} <= neighbors
    assert len(repo.related_terms("طلاق")) == 2


def test_related_terms_returns_empty_for_unknown_term(repo: TermRelationRepository) -> None:
    assert repo.related_terms("لا-يوجد") == []


def test_delete_edge_removes_exactly_one_row(repo: TermRelationRepository) -> None:
    repo.upsert(primary_term="ماء", related_term="طهور")
    assert repo.delete_edge(primary_term="ماء", related_term="طهور", relation_type="synonym")
    assert repo.get_multi() == []
    assert not repo.delete_edge(primary_term="ماء", related_term="طهور", relation_type="synonym")


def test_seed_fixtures_populates_the_lexicon_graph(repo: TermRelationRepository) -> None:
    written = repo.seed_fixtures()

    assert written == 10
    relations = repo.get_multi()
    assert len(relations) == 10

    neighbors = {edge.related_term: edge.relation_type for edge in repo.related_terms("طلاق")}
    assert neighbors.get("عده") == "related"
    assert neighbors.get("نكاح") == "related"

    synonyms = {edge.related_term for edge in repo.related_terms("صلاه")}
    assert "عباده" in synonyms

    zakah = {edge.related_term for edge in repo.related_terms("زكاه")}
    assert {"نصاب", "حول"} <= zakah


def test_seed_fixtures_is_idempotent(repo: TermRelationRepository) -> None:
    repo.seed_fixtures()
    repo.seed_fixtures()
    assert len(repo.get_multi()) == 10
