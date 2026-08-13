"""Phase 15 M3 tests (A-B): `GenerationService.stream_answer` true streaming.

Covers the service-level contract:
  A. deterministic path streams the synthesized explanation word-group by
     word-group and terminates with a validated `StreamAnswer`;
  B. a stream-capable provider yields raw tokens live; the accumulated text is
     validated and citations are rebuilt from the chunk payload;
plus the parity edge cases (complete()-only providers, provider outage →
deterministic fallback, validation failure → regenerate → refuse).
"""

import json
from collections.abc import Iterator

from app.core.config import Settings
from app.services.generation.providers import ProviderUnavailableError
from app.services.generation.service import (
    GenerationService,
    StreamAnswer,
    StreamToken,
)
from app.services.generation.synthesis import DeterministicSynthesizer
from app.services.retrieval import RetrievalResult, RetrievedChunk


def _chunk(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="الماء طهور لا ينجسه شيء",
        book_name="Al-Hidayah",
        volume="1",
        printed_page_start=5,
        topic="طهارة",
        region="main",
        lang="ar",
        verified=True,
        rerank_score=0.9,
    )


def _retrieval(chunks, *, language: str = "bn") -> RetrievalResult:
    return RetrievalResult(
        query="ওযুর নিয়ম কি",
        canonical_arabic_query="ما حكم الوضوء",
        language=language,
        translated=True,
        evidence_sufficient=bool(chunks),
        chunks=chunks,
    )


def _valid_raw() -> str:
    return json.dumps(
        {
            "answer_language": "bn",
            "explanation": {
                "type": "bengali",
                "html": "<p>উত্তর [EVIDENCE_1]</p>",
            },
            "arabic_quotes": [
                {"text": "الماء طهور", "translation": "পানি পবিত্র", "region": "main"}
            ],
            "citations": [
                {"chunk_id": "c1", "book": "hallucinated", "volume": 99, "page": 999}
            ],
            "confidence": {"rationale": "must be replaced server-side"},
            "refusal": None,
            "caveats": ["caveat"],
            "related": ["related question"],
        }
    )


class FakeStreamProvider:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = list(chunks)
        self.calls: list[tuple[str, str]] = []
        self.streams = 0

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        self.streams += 1
        self.calls.append((system_prompt, user_prompt))
        yield from self.chunks


class _RaisingStreamProvider:
    """Streaming provider that fails before yielding any token."""

    def stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        for _ in ():
            yield ""
        raise ProviderUnavailableError("vendor down")


def _split_words(text: str, width: int = 5) -> list[str]:
    words = text.split(" ")
    return [
        " ".join(words[i : i + width]) + (" " if i + width < len(words) else "")
        for i in range(0, len(words), width)
    ]


# ---------------------------------------------------------------- A. deterministic


def test_stream_answer_deterministic_streams_and_validates() -> None:
    service = GenerationService(settings=Settings(generator_provider="deterministic"))
    events = list(service.stream_answer(_retrieval([_chunk("c1")]), answer_language="bn"))

    tokens = [event.text for event in events if isinstance(event, StreamToken)]
    answers = [event for event in events if isinstance(event, StreamAnswer)]

    assert tokens, "the deterministic path must still stream deltas"
    assert len(answers) == 1
    answer = answers[0].answer
    # Deltas reassemble into the synthesized explanation (modulo trailing space).
    assert "".join(tokens).rstrip() == answer.explanation.html
    assert answer.refusal is None
    assert answer.citations[0].chunk_id == "c1"
    assert answer.citations[0].book == "Al-Hidayah"


# ------------------------------------------------------------------- B. LLM stream


def test_stream_answer_streams_provider_tokens_and_validates() -> None:
    raw = _valid_raw()
    provider = FakeStreamProvider(_split_words(raw))
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k"),
        provider=provider,
    )

    events = list(service.stream_answer(_retrieval([_chunk("c1")]), answer_language="bn"))

    tokens = [event.text for event in events if isinstance(event, StreamToken)]
    answers = [event for event in events if isinstance(event, StreamAnswer)]

    # The raw JSON fragments are forwarded verbatim, in order.
    assert "".join(tokens) == raw
    assert provider.streams == 1
    assert len(provider.calls) == 1
    assert "[EVIDENCE_1]" in provider.calls[0][0] + provider.calls[0][1]

    assert len(answers) == 1
    answer = answers[0].answer
    assert answer.refusal is None
    assert answer.explanation.html == "<p>উত্তর [EVIDENCE_1]</p>"
    # Citations rebuilt from the chunk payload — never the LLM's hallucinated values.
    assert answer.citations[0].book == "Al-Hidayah"
    assert answer.citations[0].volume == "1"
    assert answer.citations[0].page == "5"


def test_stream_answer_falls_back_to_complete_when_provider_has_no_stream() -> None:
    class _CompleteOnlyProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            return _valid_raw()

    provider = _CompleteOnlyProvider()
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k"),
        provider=provider,
    )

    events = list(service.stream_answer(_retrieval([_chunk("c1")]), answer_language="bn"))

    tokens = [event.text for event in events if isinstance(event, StreamToken)]
    answers = [event for event in events if isinstance(event, StreamAnswer)]
    assert "".join(tokens) == _valid_raw()
    assert provider.calls == 1
    assert len(answers) == 1
    assert answers[0].answer.refusal is None


def test_stream_answer_falls_back_to_deterministic_on_provider_outage() -> None:
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k"),
        provider=_RaisingStreamProvider(),
    )

    events = list(service.stream_answer(_retrieval([_chunk("c1")]), answer_language="bn"))

    tokens = [event.text for event in events if isinstance(event, StreamToken)]
    answers = [event for event in events if isinstance(event, StreamAnswer)]
    assert tokens
    assert len(answers) == 1
    assert answers[0].answer.refusal is None
    # The deterministic fallback answers from the same evidence.
    assert "".join(tokens).rstrip() == answers[0].answer.explanation.html
    assert answers[0].answer.citations[0].chunk_id == "c1"


def test_stream_answer_regenerates_once_then_refuses_on_persistent_garbage() -> None:
    provider = FakeStreamProvider(["garbage", "still not json"])
    service = GenerationService(
        settings=Settings(
            generator_provider="gemini", gemini_api_key="k", generation_retries=1
        ),
        provider=provider,
    )

    events = list(service.stream_answer(_retrieval([_chunk("c1")]), answer_language="bn"))

    answers = [event for event in events if isinstance(event, StreamAnswer)]
    assert provider.streams == 2
    assert len(answers) == 1
    assert answers[0].answer.refusal is not None
    assert answers[0].answer.refusal.reason == "generation_unavailable"


def test_stream_answer_refuses_insufficient_evidence_without_tokens() -> None:
    service = GenerationService(settings=Settings(generator_provider="deterministic"))

    events = list(service.stream_answer(_retrieval([]), answer_language="bn"))

    assert all(isinstance(event, StreamAnswer) for event in events)
    assert events[0].answer.refusal is not None
    assert events[0].answer.refusal.reason == "insufficient_evidence"


def test_stream_deltas_reassemble_synthesizer_output() -> None:
    from app.services.generation.service import _stream_deltas

    chunks = [_chunk("c1"), _chunk("c2")]
    answer = DeterministicSynthesizer().synthesize(_retrieval(chunks), answer_language="bn")

    deltas = _stream_deltas(answer.explanation.html)
    assert deltas
    assert "".join(deltas).rstrip() == answer.explanation.html
