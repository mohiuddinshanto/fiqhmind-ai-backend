"""Arabic translation (Phase 9 — Retrieval Pipeline).

ARCHITECTURE §Phase 9 step 3: translate bn/en queries to Arabic via a free API
(Google Translate free tier OR Gemini Flash) — an *external dependency*. The
pipeline therefore depends on a `Translator` interface, and
`PassthroughTranslator` is the dependency-free default: it returns the input
unchanged with `translated=False`. That is the correct behavior for an
already-Arabic query and the safe default until a provider is configured; the
retrieval runner stores BOTH the original query and the canonical Arabic query.
"""

from dataclasses import dataclass
from typing import Protocol

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source_lang: str
    target_lang: str
    translated: bool
    confidence: float


class Translator(Protocol):
    """Port implemented by every translation adapter (passthrough, Google…)."""

    def translate(
        self, text: str, *, source_lang: str, target_lang: str = "ar"
    ) -> TranslationResult: ...


class PassthroughTranslator:
    """Identity adapter: returns the input unchanged (`translated=False`).

    Safe for already-Arabic queries and for environments with no external
    translation provider configured. `translated=False` tells the pipeline (and
    later confidence estimation, Phase 12) that the canonical text is the user's
    original wording, not a machine translation.
    """

    def translate(
        self, text: str, *, source_lang: str, target_lang: str = "ar"
    ) -> TranslationResult:
        return TranslationResult(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            translated=False,
            confidence=1.0,
        )


def get_translator(settings: Settings | None = None) -> Translator:
    """Return the configured translator.

    `passthrough` is the default (Phase 9); the free-tier APIs (Google free
    tier / Gemini Flash) are external dependencies wired in a later phase.
    """
    resolved = settings or get_settings()
    if resolved.translator_provider == "passthrough":
        return PassthroughTranslator()
    if resolved.translator_provider in ("google_free", "gemini"):
        raise NotImplementedError(
            "the free-tier translation API is an external dependency; "
            f"translator_provider={resolved.translator_provider} is not wired yet "
            "(use passthrough)"
        )
    raise ValueError(f"unknown translator_provider: {resolved.translator_provider}")
