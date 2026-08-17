"""Tests for Phase 10 answer generation: prompts, validation, providers,
deterministic synthesis, confidence and the service grounded loop."""

import json

import pytest

from app.core.config import Settings
from app.schemas.chat import ChatAnswer
from app.services.generation.prompts import get_v1_prompts
from app.services.generation.providers import (
    GeminiProvider,
    GroqProvider,
    ProviderUnavailableError,
    get_llm_provider,
)
from app.services.generation.service import GenerationService
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


def _chunk(
    chunk_id: str = "c1",
    *,
    text: str = "الماء طهور لا ينجسه شيء",
    book: str = "Al-Hidayah",
    volume: str = "1",
    page: int = 5,
    topic: str = "طهارة",
    score: float = 0.9,
    verified: bool = True,
    region: str = "main",
    lang: str = "ar",
    kitab: str = "الطهارة",
    bab: str = "باب المياه",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        book_name=book,
        volume=volume,
        printed_page_start=page,
        topic=topic,
        region=region,
        lang=lang,
        verified=verified,
        rerank_score=score,
        kitab=kitab,
        bab=bab,
    )


def _retrieval(chunks, *, language: str = "bn", translated: bool = True) -> RetrievalResult:
    return RetrievalResult(
        query="ওযুর নিয়ম কি",
        canonical_arabic_query="ما حكم الوضوء",
        language=language,
        translated=translated,
        evidence_sufficient=bool(chunks),
        chunks=chunks,
    )


def _valid_raw(
    chunk_ids: tuple[str, ...] = ("c1",),
    *,
    language: str = "bn",
    html: str | None = None,
) -> str:
    return json.dumps(
        {
            "answer_language": language,
            "explanation": {
                "type": "bengali",
                "html": html or "<p>উত্তর [EVIDENCE_1]</p>",
            },
            "arabic_quotes": [{"text": "الماء طهور", "translation": "পানি পবিত্র", "region": "main"}],
            "citations": [
                {"chunk_id": chunk_id, "book": "hallucinated", "volume": 99, "page": 999}
                for chunk_id in chunk_ids
            ],
            "confidence": {"rationale": "must be replaced server-side"},
            "refusal": None,
            "caveats": ["caveat"],
            "related": ["related question"],
        }
    )


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("FakeProvider ran out of responses")
        return self.responses.pop(0)


# ---------------------------------------------------------------- prompts


def test_prompts_embed_query_language_and_evidence() -> None:
    prompts = get_v1_prompts(
        "ওযুর নিয়ম কি",
        "bn",
        [
            {
                "chunk_id": "c1",
                "text": "الماء طهور",
                "book_name": "Al-Hidayah",
                "volume": "1",
                "printed_page_start": 5,
                "topic": "طهارة",
            }
        ],
    )

    assert "[EVIDENCE_1]" in prompts["system_prompt"] + prompts["user_prompt"]
    assert "SOURCE: Al-Hidayah, vol 1, p. 5" in prompts["user_prompt"]
    assert "USER QUESTION: ওযুর নিয়ম কি" in prompts["user_prompt"]
    assert "USER QUESTION LANGUAGE: bn" in prompts["user_prompt"]
    assert "=== OUTPUT SPECIFICATION AND CONTRACT ===" in prompts["user_prompt"]


# ------------------------------------------------------------- validation


def test_validation_accepts_valid_answer() -> None:
    payload = validate_llm_answer(_valid_raw(), evidence_chunk_ids=["c1"], answer_language="bn")

    assert payload["answer_language"] == "bn"
    assert payload["citations"][0]["page"] == "999"  # coerced to string


def test_validation_accepts_fenced_json() -> None:
    raw = f"```json\n{_valid_raw()}\n```"
    payload = validate_llm_answer(raw, evidence_chunk_ids=["c1"], answer_language="bn")
    assert payload["answer_language"] == "bn"


def test_validation_rejects_non_json() -> None:
    with pytest.raises(GenerationValidationError):
        validate_llm_answer("not json at all", evidence_chunk_ids=["c1"], answer_language="bn")


def test_validation_rejects_wrong_language() -> None:
    raw = _valid_raw(language="ar")
    with pytest.raises(GenerationValidationError, match="answer_language mismatch"):
        validate_llm_answer(raw, evidence_chunk_ids=["c1"], answer_language="bn")


def test_validation_rejects_missing_explanation() -> None:
    raw = json.dumps({"answer_language": "bn"})
    with pytest.raises(GenerationValidationError, match="explanation"):
        validate_llm_answer(raw, evidence_chunk_ids=["c1"], answer_language="bn")


def test_validation_rejects_unknown_chunk_citation() -> None:
    with pytest.raises(GenerationValidationError, match="unknown chunk"):
        validate_llm_answer(
            _valid_raw(chunk_ids=("ghost",)), evidence_chunk_ids=["c1"], answer_language="bn"
        )


def test_validation_rejects_uncovered_evidence_reference() -> None:
    # Explanation references EVIDENCE_2 (chunk c2) but no citation covers c2.
    raw = json.dumps(
        {
            "answer_language": "bn",
            "explanation": {"type": "bengali", "html": "<p>[EVIDENCE_2] claims</p>"},
            "arabic_quotes": [],
            "citations": [{"chunk_id": "c1", "book": "b", "page": "1"}],
            "refusal": None,
            "caveats": [],
            "related": [],
        }
    )
    with pytest.raises(GenerationValidationError, match="not cited"):
        validate_llm_answer(raw, evidence_chunk_ids=["c1", "c2"], answer_language="bn")


def test_validation_rejects_out_of_range_evidence_reference() -> None:
    raw = json.dumps(
        {
            "answer_language": "bn",
            "explanation": {"type": "bengali", "html": "<p>[EVIDENCE_9] nope</p>"},
            "arabic_quotes": [],
            "citations": [{"chunk_id": "c1", "book": "b"}],
            "refusal": None,
            "caveats": [],
            "related": [],
        }
    )
    with pytest.raises(GenerationValidationError, match="unknown \\[EVIDENCE_9\\]"):
        validate_llm_answer(raw, evidence_chunk_ids=["c1"], answer_language="bn")


def test_validation_rejects_bad_refusal_reason() -> None:
    raw = json.dumps(
        {
            "answer_language": "bn",
            "explanation": {"type": "bengali", "html": "<p>x</p>"},
            "arabic_quotes": [],
            "citations": [],
            "refusal": {"reason": "made_up"},
            "caveats": [],
            "related": [],
        }
    )
    with pytest.raises(GenerationValidationError, match="refusal reason"):
        validate_llm_answer(raw, evidence_chunk_ids=["c1"], answer_language="bn")


# ------------------------------------------------------------- providers


def test_provider_factory_deterministic_returns_none() -> None:
    assert get_llm_provider(Settings(generator_provider="deterministic")) is None


def test_provider_factory_missing_key_falls_back() -> None:
    assert (
        get_llm_provider(
            Settings(
                generator_provider="gemini",
                gemini_api_key=None,
                groq_api_key=None,
                openrouter_api_key=None,
            )
        )
        is None
    )
    assert (
        get_llm_provider(
            Settings(
                generator_provider="groq",
                gemini_api_key=None,
                groq_api_key=None,
                openrouter_api_key=None,
            )
        )
        is None
    )


def test_provider_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="generator_provider"):
        get_llm_provider(Settings(generator_provider="claude"))


def test_provider_factory_builds_managers() -> None:
    from app.services.generation.providers import ProviderManager

    assert isinstance(
        get_llm_provider(Settings(generator_provider="gemini", gemini_api_key="k")),
        ProviderManager,
    )
    assert isinstance(
        get_llm_provider(Settings(generator_provider="groq", groq_api_key="k")),
        ProviderManager,
    )

    # With a single key, the manager's chain should be one provider
    gem_manager = get_llm_provider(
        Settings(
            generator_provider="gemini",
            gemini_api_key="k",
            groq_api_key=None,
            openrouter_api_key=None,
        )
    )
    assert gem_manager.chain_names() == ["gemini"]


class _FakeResponse:
    status_code = 500
    text = "boom"


class _FakeHttpClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return _FakeResponse()


def test_gemini_provider_raises_on_http_error(monkeypatch) -> None:
    monkeypatch.setattr("app.services.generation.providers.httpx.Client", _FakeHttpClient)
    provider = GeminiProvider(api_key="k", model="m")
    with pytest.raises(ProviderUnavailableError, match="HTTP 500"):
        provider.complete(system_prompt="s", user_prompt="u")


def test_groq_provider_raises_on_http_error(monkeypatch) -> None:
    monkeypatch.setattr("app.services.generation.providers.httpx.Client", _FakeHttpClient)
    provider = GroqProvider(api_key="k", model="m")
    with pytest.raises(ProviderUnavailableError, match="HTTP 500"):
        provider.complete(system_prompt="s", user_prompt="u")


# ------------------------------------------------------------ confidence


def test_confidence_high_on_consensus_verified_strong() -> None:
    chunks = [
        _chunk("c1", score=0.9),
        _chunk("c2", score=0.8, page=6),
        _chunk("c3", score=0.7, page=7),
    ]
    confidence = compute_confidence(chunks)

    assert confidence.level == "high"
    assert confidence.source_agreement == "consensus"
    assert confidence.retrieval_score == pytest.approx(0.9)


def test_confidence_medium_on_weaker_scores() -> None:
    chunks = [_chunk("c1", score=0.4), _chunk("c2", score=0.38, page=6)]
    assert compute_confidence(chunks).level == "medium"


def test_confidence_low_on_weak_scores() -> None:
    chunks = [_chunk("c1", score=0.1)]
    assert compute_confidence(chunks).level == "low"


def test_confidence_conflict_on_different_topics() -> None:
    chunks = [_chunk("c1", score=0.9, topic="طهارة"), _chunk("c2", score=0.8, topic="صلاة")]
    confidence = compute_confidence(chunks)
    assert confidence.source_agreement == "conflict"


def test_confidence_empty_chunks_is_unknown() -> None:
    confidence = compute_confidence([])
    assert confidence.level == "low"
    assert confidence.source_agreement == "unknown"


def test_confidence_mentions_translation() -> None:
    chunks = [_chunk("c1", score=0.9)]
    assert "machine-translated" in compute_confidence(chunks, translated=True).rationale


# ------------------------------------------------- determininistic synthesis


def test_synthesizer_builds_grounded_answer() -> None:
    chunks = [
        _chunk("c1", text="الماء طهور لا ينجسه شيء"),
        _chunk("c2", text="الوضوء شرط الصلاة", topic="طهارة", page=9, bab="باب الوضوء"),
    ]
    answer = DeterministicSynthesizer().synthesize(_retrieval(chunks), answer_language="bn")

    assert isinstance(answer, ChatAnswer)
    assert answer.refusal is None
    assert answer.citations[0].book == "Al-Hidayah"
    assert answer.citations[0].page == "5"
    assert [q.text for q in answer.arabic_quotes] == [
        "الماء طهور لا ينجسه شيء",
        "الوضوء شرط الصلاة",
    ]
    assert answer.answer_language == "bn"
    # The user-facing explanation is a concise, fact-grounded answer — never a
    # raw dump of every evidence block.
    assert "[EVIDENCE_" not in answer.explanation.html
    assert answer.explanation.html.startswith("<p><b>উত্তর:</b>")
    assert "উৎস:" in answer.explanation.html
    assert answer.confidence.level == "high"


def test_synthesizer_skips_non_arabic_chunks_for_quotes() -> None:
    chunks = [_chunk("c1", text="আল-হেদায়াহ থেকে বাংলা অনুবাদ", lang="bn", region="main")]
    answer = DeterministicSynthesizer().synthesize(_retrieval(chunks))
    assert answer.arabic_quotes == []


def test_synthesizer_respects_max_chunks() -> None:
    chunks = [_chunk(f"c{i}", page=i) for i in range(1, 5)]
    answer = DeterministicSynthesizer(max_chunks=2).synthesize(_retrieval(chunks))
    assert len(answer.citations) == 2


def test_build_refusal_shows_closest_evidence() -> None:
    chunks = [_chunk("c1", text="الماء طهور " * 60)]
    answer = build_refusal(_retrieval(chunks), answer_language="bn")

    assert answer.refusal is not None
    assert answer.refusal.reason == "insufficient_evidence"
    assert len(answer.refusal.closest_evidence) == 1
    assert len(answer.refusal.closest_evidence[0]) <= 301


def test_build_refusal_empty_chunks() -> None:
    answer = build_refusal(_retrieval([]), answer_language="bn")
    assert answer.refusal.reason == "insufficient_evidence"
    assert answer.refusal.closest_evidence == []


# ------------------------------------------------------------ service


def test_service_deterministic_path_without_provider() -> None:
    service = GenerationService(settings=Settings(generator_provider="deterministic"))
    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    assert answer.refusal is None
    assert answer.citations[0].chunk_id == "c1"


def test_service_refuses_when_evidence_insufficient() -> None:
    service = GenerationService(settings=Settings(generator_provider="deterministic"))
    answer = service.generate(_retrieval([]), answer_language="bn")

    assert answer.refusal is not None
    assert answer.refusal.reason == "insufficient_evidence"


def test_service_falls_back_to_deterministic_on_provider_error() -> None:
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k"),
        provider=_RaisingProvider(),
    )
    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    assert answer.refusal is None
    assert answer.citations[0].book == "Al-Hidayah"


class _RaisingProvider:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise ProviderUnavailableError("vendor down")


def test_service_uses_llm_when_valid() -> None:
    provider = FakeProvider([_valid_raw()])
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k", generation_retries=1),
        provider=provider,
    )

    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    assert len(provider.calls) == 1
    assert answer.refusal is None
    assert answer.explanation.html == "<p>উত্তর [EVIDENCE_1]</p>"


def test_service_rebuilds_citations_from_chunk_payload() -> None:
    provider = FakeProvider([_valid_raw()])
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k"),
        provider=provider,
    )

    answer = service.generate(
        _retrieval([
            _chunk("c1", book="Al-Hidayah", page=5),
            _chunk("c2", book="Al-Hidayah", page=6),
        ]), answer_language="bn"
    )

    # The LLM claimed book="hallucinated", volume=99, page=999 — never trusted.
    assert answer.citations[0].book == "Al-Hidayah"
    assert answer.citations[0].volume == "1"
    assert answer.citations[0].page == "5"
    assert answer.confidence.level == "high"


def test_service_regenerates_once_then_refuses_on_persistent_failure() -> None:
    provider = FakeProvider(["garbage", "still not json"])
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k", generation_retries=1),
        provider=provider,
    )

    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    assert len(provider.calls) == 2
    assert answer.refusal is not None
    assert answer.refusal.reason == "generation_unavailable"


def test_service_regenerates_once_and_recovers() -> None:
    provider = FakeProvider(["broken json", _valid_raw()])
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k", generation_retries=1),
        provider=provider,
    )

    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    assert len(provider.calls) == 2
    assert answer.refusal is None
    assert answer.citations[0].chunk_id == "c1"


def test_service_sanitizes_llm_html_before_returning() -> None:
    hostile = (
        '<p>উত্তর [EVIDENCE_1]</p><script>alert(document.cookie)</script>'
        '<img src=x onerror="alert(1)">'
    )
    provider = FakeProvider([_valid_raw(html=hostile)])
    service = GenerationService(
        settings=Settings(generator_provider="gemini", gemini_api_key="k"),
        provider=provider,
    )

    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    html = answer.explanation.html
    assert "<script>" not in html and "</script>" not in html
    assert "onerror" not in html and "<img" not in html
    # The legitimate evidence markup and citation marker survive.
    assert "<p>উত্তর [EVIDENCE_1]</p>" in html


def test_service_sanitization_can_be_disabled_via_settings() -> None:
    hostile = '<p>ok [EVIDENCE_1]</p><script>alert(1)</script>'
    provider = FakeProvider([_valid_raw(html=hostile)])
    service = GenerationService(
        settings=Settings(
            generator_provider="gemini",
            gemini_api_key="k",
            generation_sanitize_html=False,
        ),
        provider=provider,
    )

    answer = service.generate(_retrieval([_chunk("c1")]), answer_language="bn")

    assert answer.explanation.html == hostile
