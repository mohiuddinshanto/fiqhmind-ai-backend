"""Query preprocessing (Phase 9 — Retrieval Pipeline).

ARCHITECTURE §Phase 9 step 1: normalize Unicode (alef/hamza variants), remove
diacritics *for matching* while keeping the original for display, trim, bound
length, and drop profanity/attack patterns.

Every query becomes two copies: `display` (the trimmed, bounded original — what
the user typed) and `canonical` (the normalization used for matching, expansion
and reranking). The canonical copy is NFC-normalized, folds the alef/hamza
variants (أ إ آ ٱ → ا, ؤ ئ → ء, ى → ي, ة → ه) and strips combining marks
(harakat + tatweel). The definite article ال is deliberately *not* stripped
globally: a lossy strip corrupts words like الله / التي, and the sparse lexical
path already handles exact tokens. Rejected queries raise `QueryValidationError`
(empty, over-long, or attack/profanity patterns).
"""

import re
import unicodedata
from dataclasses import dataclass

import structlog

from app.core.config import Settings, get_settings
from app.core.exceptions import QueryValidationError

logger = structlog.get_logger(__name__)

# Alef/hamza variant folding (matching copy only; display keeps the original).
_ALEF_RE = re.compile("[أإآٱ]")
_ALEF_MAQSURA_RE = re.compile("[ى]")
_TA_MARBUTA_RE = re.compile("[ة]")
_HAMZA_RE = re.compile("[ؤئ]")

# Combining marks (harakat, sukun, shadda, maddah, tatweel, Quranic signs).
_COMBINING_RE = re.compile(
    "[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]"
)

# Attack / prompt-injection phrases that make a query unusable (case-folded
# substring checks — deliberately no regex, so this is ReDoS-safe).
_ATTACK_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your instructions",
    "forget your instructions",
    "pretend you are",
    "system prompt",
    "you are now",
    "jailbreak",
)

# Unambiguous crude abuse tokens. Kept minimal — a scholarly filter, not a
# censor; substrings that occur inside innocent words (e.g. "ass" in "assist")
# are deliberately not listed.
_PROFANITY_PATTERNS = (
    "fuck",
    "fucking",
    "shut up",
    "stupid",
    "idiot",
    "الاحمق",
    "غبي",
    "بلا عقل",
)


def strip_diacritics(text: str) -> str:
    """Remove combining marks (harakat etc.) from `text` (canonical copy)."""
    return _COMBINING_RE.sub("", text)


def normalize_arabic(text: str) -> str:
    """Produce the canonical matching copy of an Arabic(-script) string.

    NFC-normalize, fold alef/hamza variants, alef maqsura, taa marbuta, then
    strip combining marks. Deterministic and lossless enough for retrieval
    matching while preserving the original for display elsewhere.
    """
    folded = unicodedata.normalize("NFC", text)
    folded = _ALEF_RE.sub("ا", folded)
    folded = _ALEF_MAQSURA_RE.sub("ي", folded)
    folded = _TA_MARBUTA_RE.sub("ه", folded)
    folded = _HAMZA_RE.sub("ء", folded)
    return strip_diacritics(folded)


@dataclass(frozen=True)
class PreparedQuery:
    """A query that passed preprocessing, with its two copies."""

    original: str
    display: str
    canonical: str


class QueryPreprocessor:
    """Trims, bounds, validates and normalizes a user query (Phase 9 step 1)."""

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        self._max_length = max(int(resolved.query_max_length), 1)

    def prepare(self, text: str) -> PreparedQuery:
        display = text.strip()
        if not display:
            raise QueryValidationError("query must not be empty")
        if len(display) > self._max_length:
            raise QueryValidationError(
                f"query exceeds the {self._max_length}-character limit"
            )
        lowered = display.lower()
        for pattern in _ATTACK_PATTERNS + _PROFANITY_PATTERNS:
            if pattern in lowered:
                logger.warning("query_rejected", reason="profanity_or_attack")
                raise QueryValidationError("query contains disallowed content")
        return PreparedQuery(
            original=text,
            display=display,
            canonical=normalize_arabic(display),
        )
