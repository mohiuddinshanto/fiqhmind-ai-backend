"""Language detection (Phase 9 — Retrieval Pipeline).

ARCHITECTURE §Phase 9 step 2: detect the query language with fastText LID-176 /
a small fine-tuned head — an *external model*. The pipeline therefore depends on
a `LanguageDetector` interface, and `HeuristicLanguageDetector` is the
dependency-free default: script-range classification over the three languages
the product serves (Arabic, Bengali, English). The fastText head plugs into the
same interface in a later phase without touching the retrieval pipeline.
"""

from dataclasses import dataclass
from typing import Protocol

import structlog

from app.core.config import Settings, get_settings
from app.core.exceptions import TranslationError

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Language:
    """A detected language: an ISO-ish code plus a script-based confidence."""

    code: str  # "ar" | "bn" | "en" | "other"
    confidence: float  # fraction of letter characters matching the script


class LanguageDetector(Protocol):
    """Port implemented by every language-detection adapter."""

    def detect(self, text: str) -> Language: ...


def _is_arabic(ch: str) -> bool:
    return "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\u08a0" <= ch <= "\u08ff"


def _is_bengali(ch: str) -> bool:
    return "\u0980" <= ch <= "\u09ff"


def _is_latin(ch: str) -> bool:
    return "A" <= ch <= "Z" or "a" <= ch <= "z" or "\u00c0" <= ch <= "\u00ff"


def has_arabic_script(text: str) -> bool:
    """True when `text` contains at least one Arabic-script character."""
    return any(_is_arabic(ch) for ch in text)


class HeuristicLanguageDetector:
    """Dependency-free script-range classifier (default adapter)."""

    def detect(self, text: str) -> Language:
        counts = {"ar": 0, "bn": 0, "en": 0}
        for ch in text:
            if _is_arabic(ch):
                counts["ar"] += 1
            elif _is_bengali(ch):
                counts["bn"] += 1
            elif _is_latin(ch):
                counts["en"] += 1
        total = sum(counts.values())
        if total == 0:
            return Language(code="other", confidence=0.0)
        code = max(counts, key=lambda key: (counts[key], key))
        return Language(code=code, confidence=counts[code] / total)


def get_language_detector(settings: Settings | None = None) -> LanguageDetector:
    """Return the configured language detector.

    `heuristic` is the default (Phase 9); the fastText LID-176 head is the
    future production adapter and is deliberately not wired in this phase.
    """
    resolved = settings or get_settings()
    if resolved.language_detector_provider == "heuristic":
        return HeuristicLanguageDetector()
    if resolved.language_detector_provider == "fasttext":
        raise NotImplementedError(
            "the fastText LID-176 language head is an external model; "
            "use language_detector_provider=heuristic until it lands"
        )
    raise ValueError(f"unknown language_detector_provider: {resolved.language_detector_provider}")


def require_language(text: str, detector: LanguageDetector) -> Language:
    """Detect a query language, refusing undetectable input.

    An empty/detect-free query is a pipeline error (the preprocessor should
    already have rejected it), mapped to `TranslationError` because the
    translation stage consumes this detection.
    """
    language = detector.detect(text)
    if language.code == "other":
        raise TranslationError("could not detect the query language")
    return language
