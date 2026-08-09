from __future__ import annotations

from sqlalchemy import or_, select

from app.db.models import TermRelation
from app.db.repositories.base import RepositoryBase


class TermRelationRepository(RepositoryBase[TermRelation]):
    """Access to the flat fiqh term-relation graph (Phase 9).

    Every method is deliberately auditable: edges are upserted one at a time
    (unique per directed `(primary_term, related_term, relation_type)`), and
    `related_terms` reads the 1-hop neighborhood in *both* directions so a
    symmetric synonym needs only one stored row.
    """

    model = TermRelation

    def related_terms(self, term: str) -> list[TermRelation]:
        """Return the 1-hop neighborhood of `term` (term in either column).

        Edges are ordered by confidence descending; the caller caps the number
        of variants (ARCHITECTURE: "a single indexed SQL join").
        """
        return list(
            self._session.scalars(
                select(TermRelation)
                .where(
                    or_(
                        TermRelation.primary_term == term,
                        TermRelation.related_term == term,
                    )
                )
                .order_by(TermRelation.confidence.desc())
            )
        )

    def upsert(
        self,
        *,
        primary_term: str,
        related_term: str,
        relation_type: str = "synonym",
        confidence: float = 1.0,
    ) -> TermRelation:
        """Insert one directed edge or update it in place (idempotent)."""
        existing = self._session.scalar(
            select(TermRelation).where(
                TermRelation.primary_term == primary_term,
                TermRelation.related_term == related_term,
                TermRelation.relation_type == relation_type,
            )
        )
        if existing is not None:
            existing.confidence = confidence
            return self.update(existing)
        return self.create(
            TermRelation(
                primary_term=primary_term,
                related_term=related_term,
                relation_type=relation_type,
                confidence=confidence,
            )
        )

    def delete_edge(self, *, primary_term: str, related_term: str, relation_type: str) -> bool:
        """Delete one directed edge; returns True when a row was removed."""
        existing = self._session.scalar(
            select(TermRelation).where(
                TermRelation.primary_term == primary_term,
                TermRelation.related_term == related_term,
                TermRelation.relation_type == relation_type,
            )
        )
        if existing is None:
            return False
        self._session.delete(existing)
        self._session.commit()
        return True

    def seed_fixtures(self) -> int:
        """Seed the manually curated fiqh lexicon graph (ARCHITECTURE examples).

        Terms are stored in the canonical matching-copy normalization (same as
        the Phase 9 query preprocessor) so a query token matches an edge
        directly. Synonyms mirror the Phase 9 synonym lexicon (ثبوت/وجوب,
        صلاة/عبادة, ماء/طهور); `related` edges mirror the worked graph examples
        ("طلاق" relates_to "عدة" / "نكاح"; "زكاة" relates_to "نصاب" / "حول").
        Returns the number of rows written. Idempotent: re-running updates in
        place via `upsert`.
        """
        edges = [
            ("ثبوت", "وجوب", "synonym", 1.0),
            ("وجوب", "ثبوت", "synonym", 1.0),
            ("صلاه", "عباده", "synonym", 0.9),
            ("عباده", "صلاه", "synonym", 0.9),
            ("ماء", "طهور", "synonym", 1.0),
            ("طهور", "ماء", "synonym", 1.0),
            ("طلاق", "عده", "related", 1.0),
            ("طلاق", "نكاح", "related", 0.8),
            ("زكاه", "نصاب", "related", 1.0),
            ("زكاه", "حول", "related", 1.0),
        ]
        for primary_term, related_term, relation_type, confidence in edges:
            self.upsert(
                primary_term=primary_term,
                related_term=related_term,
                relation_type=relation_type,
                confidence=confidence,
            )
        return len(edges)
