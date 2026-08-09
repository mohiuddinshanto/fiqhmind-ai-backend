"""Answer generation service (Phase 10 grounded loop).

Implements the evidence-first loop from ARCHITECTURE §Phase 10:
  Retrieve → Compress → Generate(structured) → Validate schema + citation
  coverage → (fail? regenerate once) → (still failing? refuse with evidence
  summary) → return answer.

Provider policy:
  - `generator_provider=deterministic` (or no configured provider) → the
    `DeterministicSynthesizer` answers directly from the evidence.
  - provider unreachable (`ProviderUnavailableError`) → log and fall back to
    deterministic synthesis so the chat pipeline never hard-fails.
  - provider returns output that fails validation on every attempt → refuse
    with the evidence summary (never surface an unvalidated answer).

Citations are always rebuilt from the chunk payload by chunk_id — the LLM only
selects which chunks to cite, it never supplies book/volume/page text.
"""

from typing import Any

import structlog

from app.core.config import Settings, get_settings
from app.schemas.chat import ArabicQuote, ChatAnswer, Citation, Explanation, Refusal
from app.services.generation.prompts import get_v1_prompts
from app.services.generation.providers import (
    LLMProvider,
    ProviderUnavailableError,
    get_llm_provider,
)
from app.services.generation.synthesis import (
    DeterministicSynthesizer,
    build_refusal,
    compute_confidence,
)
from app.services.generation.validation import (
    GenerationValidationError,
    validate_llm_answer,
)
from app.services.retrieval import RetrievalResult, RetrievedChunk

logger = structlog.get_logger(__name__)


class GenerationService:
    """Orchestrates evidence → grounded answer with deterministic fallback."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        synthesizer: DeterministicSynthesizer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = get_llm_provider(self._settings) if provider is None else provider
        self._synthesizer = synthesizer or DeterministicSynthesizer(
            max_chunks=self._settings.generation_max_chunks
        )

    def generate(self, retrieval: RetrievalResult, *, answer_language: str = "bn") -> ChatAnswer:
        if not retrieval.evidence_sufficient:
            return build_refusal(
                retrieval, answer_language=answer_language, reason="insufficient_evidence"
            )
        if self._provider is None:
            return self._synthesizer.synthesize(retrieval, answer_language=answer_language)
        try:
            return self._generate_via_llm(retrieval, answer_language)
        except ProviderUnavailableError as exc:
            logger.warning(
                "llm_provider_unavailable; falling back to deterministic", error=str(exc)
            )
            return self._synthesizer.synthesize(retrieval, answer_language=answer_language)

    def _generate_via_llm(self, retrieval: RetrievalResult, answer_language: str) -> ChatAnswer:
        provider = self._provider
        if provider is None:
            raise ProviderUnavailableError("no provider configured")

        chunks = retrieval.chunks[: self._settings.generation_max_chunks]
        evidence = [_evidence_block(chunk) for chunk in chunks]
        prompts = get_v1_prompts(retrieval.query, answer_language, evidence)

        attempts = self._settings.generation_retries + 1
        last_error: GenerationValidationError | None = None
        for attempt in range(1, attempts + 1):
            try:
                raw = provider.complete(
                    system_prompt=prompts["system_prompt"],
                    user_prompt=prompts["user_prompt"],
                )
            except ProviderUnavailableError:
                raise
            except Exception as exc:  # defensive: any vendor error is unavailability
                raise ProviderUnavailableError(f"provider call failed: {exc}") from exc

            try:
                payload = validate_llm_answer(
                    raw,
                    evidence_chunk_ids=[chunk.chunk_id for chunk in chunks],
                    answer_language=answer_language,
                )
            except GenerationValidationError as exc:
                last_error = exc
                logger.warning(
                    "llm_answer_validation_failed",
                    attempt=attempt,
                    attempts=attempts,
                    error=str(exc),
                )
                continue
            return self._assemble(payload, retrieval, chunks, answer_language)

        logger.error(
            "llm_answer_rejected_after_all_attempts; refusing with evidence summary",
            error=str(last_error),
        )
        return build_refusal(
            retrieval, answer_language=answer_language, reason="generation_unavailable"
        )

    @staticmethod
    def _assemble(
        payload: dict[str, Any],
        retrieval: RetrievalResult,
        chunks: list[RetrievedChunk],
        answer_language: str,
    ) -> ChatAnswer:
        by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
        citations = [
            _citation_from_chunk(citation["chunk_id"], by_chunk_id)
            for citation in payload["citations"]
        ]
        quotes = [
            ArabicQuote(
                text=quote["text"],
                translation=quote.get("translation"),
                region=quote.get("region"),
            )
            for quote in payload["arabic_quotes"]
        ]
        refusal = payload.get("refusal")
        return ChatAnswer(
            answer_language=answer_language,
            explanation=Explanation(
                type=payload["explanation"].get("type", "markdown"),
                html=payload["explanation"]["html"],
            ),
            arabic_quotes=quotes,
            citations=citations,
            confidence=compute_confidence(chunks, translated=retrieval.translated),
            refusal=Refusal.model_validate(refusal) if refusal else None,
            caveats=[str(item) for item in payload.get("caveats", [])],
            related=[str(item) for item in payload.get("related", [])],
        )


def _evidence_block(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "book_name": chunk.book_name,
        "volume": chunk.volume,
        "printed_page_start": chunk.printed_page_start,
        "topic": chunk.topic,
    }


def _citation_from_chunk(chunk_id: str, by_chunk_id: dict[str, RetrievedChunk]) -> Citation:
    """Rebuild the citation from the chunk payload — never trust LLM-supplied fields.

    The LLM only chooses *which* chunks to cite; book/volume/page/chapter come
    from the retrieved chunk so a citation can never be hallucinated.
    """
    chunk = by_chunk_id.get(chunk_id)
    if chunk is None:
        # Should be impossible: the validator enforces chunk_id membership.
        raise GenerationValidationError(f"citation references chunk {chunk_id!r} outside context")
    return Citation(
        chunk_id=chunk_id,
        book=chunk.book_name,
        volume=chunk.volume,
        page=str(chunk.printed_page_start) if chunk.printed_page_start is not None else None,
        edition=None,
        chapter=chunk.topic or chunk.kitab or chunk.bab,
    )


def get_generator(settings: Settings | None = None) -> GenerationService:
    """Factory used by the API layer (mirrors the retrieval DI pattern)."""
    return GenerationService(settings=settings)
