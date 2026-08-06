"""Query expansion (Phase 9 — Retrieval Pipeline).

ARCHITECTURE §Phase 9 step 4 (Arabic domain): fiqh synonym lexicon, bounded LLM
paraphrases (external — gated off), and a lightweight knowledge-graph hop over
the `term_relations` table. The runner keeps the original + the canonical Arabic
query + bounded variants, at most `retrieval_max_variants` candidate queries.

Candidate kinds: `original` (user's wording), `canonical` (translated Arabic),
`synonym` (lexicon expansion), `kg` (term_relations 1-hop hop), `llm` (external,
off by default). The KG hop is a single indexed SQL join per matched term
(ARCHITECTURE: "a single indexed SQL join — no new infrastructure").
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

import structlog
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.repositories import TermRelationRepository
from app.services.language import has_arabic_script
from app.services.query_preprocessing import normalize_arabic

logger = structlog.get_logger(__name__)

# Fiqh synonym lexicon (ARCHITECTURE examples: ثبوت/وجوب, صلاة/عبادة, ماء/طهور).
# Keys and values are stored in the canonical matching-copy normalization
# (see `normalize_arabic`) so they match query tokens directly.
FIQH_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ثبوت": ("وجوب",),
    "وجوب": ("ثبوت",),
    "صلاه": ("عباده",),
    "عباده": ("صلاه",),
    "ماء": ("طهور",),
    "طهور": ("ماء",),
}


@dataclass(frozen=True)
class CandidateQuery:
    """One query string sent to hybrid search, tagged by origin."""

    text: str
    kind: str  # "original" | "canonical" | "synonym" | "kg" | "llm"


@dataclass(frozen=True)
class ExpandedQuery:
    """The prepared query plus every candidate string for hybrid search."""

    canonical_arabic: str
    candidates: tuple[CandidateQuery, ...] = field(default_factory=tuple)


class Paraphraser(Protocol):
    """Port for the external LLM paraphrase step (ARCHITECTURE step 4)."""

    def paraphrase(self, text: str, *, limit: int = 3) -> list[str]: ...


class _UnconfiguredParaphraser:
    """Placeholder: paraphrasing is external and off until a provider exists."""

    def paraphrase(self, text: str, *, limit: int = 3) -> list[str]:
        raise NotImplementedError(
            "LLM query paraphrasing is an external dependency; enable "
            "retrieval_llm_expansion_enabled only with a configured provider"
        )


class ExpansionRunner:
    """Builds bounded candidate queries from a prepared + translated query."""

    def __init__(
        self,
        session: Session | None = None,
        settings: Settings | None = None,
        paraphraser: Paraphraser | None = None,
    ) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._paraphraser = paraphraser or _UnconfiguredParaphraser()
        self._term_repo = TermRelationRepository(session) if session is not None else None

    def expand(self, *, original: str, canonical_arabic: str) -> ExpandedQuery:
        """Return bounded candidates: original + canonical + variants."""
        # Matching happens in canonical form; `term_relations` and the lexicon
        # store that same normalization, so normalize defensively here.
        canonical_arabic = normalize_arabic(canonical_arabic)
        max_variants = max(int(self._settings.retrieval_max_variants), 1)
        candidates: list[CandidateQuery] = []

        def _add(text: str, kind: str) -> bool:
            text = text.strip()
            if not text or any(candidate.text == text for candidate in candidates):
                return False
            candidates.append(CandidateQuery(text=text, kind=kind))
            return True

        _add(original, "original")
        _add(canonical_arabic, "canonical")

        for variant in self._synonym_variants(canonical_arabic):
            _add(variant, "synonym")
        if self._term_repo is not None:
            for variant in self._kg_variants(canonical_arabic):
                _add(variant, "kg")

        if self._settings.retrieval_llm_expansion_enabled:
            for paraphrase in self._paraphraser.paraphrase(
                canonical_arabic, limit=max_variants
            ):
                _add(paraphrase, "llm")

        return ExpandedQuery(
            canonical_arabic=canonical_arabic,
            candidates=tuple(candidates[:max_variants]),
        )

    def _synonym_variants(self, canonical: str) -> list[str]:
        """One variant per matched lexicon term, first synonym only (bounded)."""
        variants: list[str] = []
        for term, synonyms in FIQH_SYNONYMS.items():
            if term not in canonical:
                continue
            for synonym in synonyms:
                variant = canonical.replace(term, synonym, 1)
                if variant != canonical and variant not in variants:
                    variants.append(variant)
                break
        return variants

    def _kg_variants(self, canonical: str) -> list[str]:
        """1-hop related-term variants from `term_relations` (ARCHITECTURE §9)."""
        if self._term_repo is None:
            return []
        variants: list[str] = []
        seen_neighbors: set[str] = set()
        for probe in self._probe_terms(canonical):
            edges = self._term_repo.related_terms(probe)
            for edge in edges:
                neighbor = (
                    edge.related_term if edge.primary_term == probe else edge.primary_term
                )
                if neighbor in seen_neighbors:
                    continue
                seen_neighbors.add(neighbor)
                variant = canonical.replace(probe, neighbor, 1)
                if variant != canonical:
                    variants.append(variant)
        return variants

    def _probe_terms(self, canonical: str) -> list[str]:
        """Arabic tokens to probe in `term_relations`, plus lexicon keys.

        A token is normalized (diacritics/alef folding) and its bare form is
        also probed so the definite article ال doesn't hide an edge.
        """
        probes: set[str] = set()
        for token in re.findall(r"\S+", canonical):
            if not has_arabic_script(token):
                continue
            bare = normalize_arabic(token)
            probes.add(bare)
            if bare.startswith("ال") and len(bare) > 3:
                probes.add(bare[2:])
        probes.update(term for term in FIQH_SYNONYMS if term in canonical)
        return sorted(probes)
