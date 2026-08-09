"""Deterministic answer synthesis + confidence computation (Phase 10).

Three deterministic pieces the LLM path also relies on:

  - `compute_confidence`: a pure function over (a) top rerank scores, (b)
    top-chunk agreement, (c) verified/region flags and (d) translation usage.
    The LLM never sets the confidence level (ARCHITECTURE §Phase 10).
  - `DeterministicSynthesizer`: the dependency-free fallback that answers from
    the retrieved evidence without an external model — verbatim quotes and
    citations built strictly from the chunk payload.
  - `build_refusal`: a graceful refusal for insufficient evidence or a failed
    generation attempt, always with the closest evidence shown.
"""

import re
from typing import Any

from app.core.config import get_settings
from app.schemas.chat import (
    ArabicQuote,
    ChatAnswer,
    Citation,
    Confidence,
    Explanation,
    Refusal,
)
from app.services.retrieval import RetrievalResult, RetrievedChunk

_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")

_EXCERPT_LIMIT = 700
_CLOSEST_LIMIT = 300
_QUOTE_LIMIT = 1500


def compute_confidence(chunks: list[RetrievedChunk], *, translated: bool = False) -> Confidence:
    """Deterministic confidence over the reranked evidence (no LLM involved)."""
    if not chunks:
        return Confidence(
            level="low",
            retrieval_score=0.0,
            source_agreement="unknown",
            rationale="No evidence retrieved for this question.",
        )
    top_score = round(max(chunk.rerank_score for chunk in chunks), 4)
    top = chunks[:3]
    topics = {
        (chunk.topic or "").strip() or (chunk.kitab or "").strip() or "unknown" for chunk in top
    }
    consensus = len(topics) <= 1
    verified = sum(1 for chunk in top if chunk.verified)
    agreement = "consensus" if consensus else "conflict"

    if top_score >= 0.6 and consensus and verified >= 2:
        level = "high"
    elif top_score >= 0.35:
        level = "medium"
    else:
        level = "low"

    parts = [
        f"top rerank score {top_score:.2f}",
        f"{len(top)} chunk(s) considered",
        f"source agreement: {agreement}",
        f"verified evidence: {verified}/{len(top)}",
    ]
    if translated:
        parts.append("query was machine-translated; confidence reduced")
    return Confidence(
        level=level,
        retrieval_score=top_score,
        source_agreement=agreement,
        rationale="; ".join(parts) + ".",
    )


class DeterministicSynthesizer:
    """Builds a grounded answer straight from the evidence (no external model)."""

    def __init__(self, *, max_chunks: int | None = None) -> None:
        self._max_chunks = max_chunks or get_settings().generation_max_chunks

    def synthesize(self, retrieval: RetrievalResult, *, answer_language: str = "bn") -> ChatAnswer:
        chunks = retrieval.chunks[: self._max_chunks]
        t = _templates(answer_language)
        confidence = compute_confidence(chunks, translated=retrieval.translated)

        explanation_html = (
            f"<p><b>{t['lead']}</b></p>\n"
            + "\n".join(
                (
                    f"<p>[EVIDENCE_{i}] <cite>{_source_label(chunk, t)}</cite>: "
                    f"{_excerpt(chunk.text)}</p>"
                )
                for i, chunk in enumerate(chunks, 1)
            )
            + f"\n<p><b>{t['note']}</b></p>"
        )

        return ChatAnswer(
            answer_language=answer_language,
            explanation=Explanation(type="markdown", html=explanation_html),
            arabic_quotes=_arabic_quotes(chunks, answer_language, t),
            citations=_citations(chunks),
            confidence=confidence,
            refusal=None,
            caveats=[t["caveat"]] if chunks else [],
            related=_related(chunks),
        )


def build_refusal(
    retrieval: RetrievalResult,
    *,
    answer_language: str = "bn",
    reason: str = "insufficient_evidence",
) -> ChatAnswer:
    """Refuse gracefully, always showing the closest evidence found."""
    t = _templates(answer_language)
    closest = [
        _truncate(chunk.text, _CLOSEST_LIMIT)
        for chunk in retrieval.chunks[:2]
        if chunk.text.strip()
    ]
    html = (
        f"<p>{t['refusal']}</p>"
        if not closest
        else (
            f"<p>{t['refusal']}</p>\n<p><b>{t['closest_heading']}</b></p>\n"
            + "\n".join(f"<p><i>{text}</i></p>" for text in closest)
        )
    )
    return ChatAnswer(
        answer_language=answer_language,
        explanation=Explanation(type="markdown", html=html),
        confidence=compute_confidence(retrieval.chunks, translated=retrieval.translated),
        refusal=Refusal(reason=reason, closest_evidence=closest),
        caveats=[t["caveat"]],
        related=[],
    )


def _arabic_quotes(
    chunks: list[RetrievedChunk], language: str, t: dict[str, str]
) -> list[ArabicQuote]:
    quotes: list[ArabicQuote] = []
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text or not _ARABIC_SCRIPT.search(text):
            continue
        quotes.append(
            ArabicQuote(
                text=_truncate(text, _QUOTE_LIMIT),
                translation=_excerpt_label(chunk, language, t),
                region=chunk.region,
            )
        )
    return quotes


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            chunk_id=chunk.chunk_id,
            book=chunk.book_name,
            volume=chunk.volume,
            page=str(chunk.printed_page_start) if chunk.printed_page_start is not None else None,
            edition=None,
            chapter=chunk.topic or chunk.kitab or chunk.bab,
        )
        for chunk in chunks
    ]


def _source_label(chunk: RetrievedChunk, t: dict[str, str]) -> str:
    parts = [part for part in (chunk.book_name,) if part]
    if chunk.volume:
        parts.append(f"{t['volume']} {chunk.volume}")
    if chunk.printed_page_start is not None:
        parts.append(f"{t['page']} {chunk.printed_page_start}")
    return ", ".join(parts) or "unknown source"


def _excerpt(text: str) -> str:
    return _truncate(text, _EXCERPT_LIMIT)


def _excerpt_label(chunk: RetrievedChunk, language: str, t: dict[str, str]) -> str:
    location = _source_label(chunk, t)
    if language == "bn":
        return f"উদ্ধৃতাংশ — {location}"
    if language == "ar":
        return f"مقتطف — {location}"
    return f"Excerpt — {location}"


def _related(chunks: list[RetrievedChunk]) -> list[str]:
    topics = list(dict.fromkeys(chunk.topic for chunk in chunks if chunk.topic))
    return topics[:2]


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _templates(language: str) -> dict[str, Any]:
    if language == "ar":
        return {
            "lead": "وجدنا الأدلة التالية من المصادر المتعلقة بسؤالك:",
            "note": (
                "هذا ملخص أدلة آلي (وضع التراجع الحتمي)؛ لم يتم إنتاج تحليل بسبب عدم توفر "
                "نموذج اللغة في هذا الوقت."
            ),
            "volume": "مجلد",
            "page": "صفحة",
            "caveat": (
                "إجابة تلقائية مبنية على الأدلة المسترجعة دون توليد نموذج لغوي؛ "
                "استشر عالماً مؤهلاً للفتوى."
            ),
            "refusal": (
                "لم نجد أدلة كافية في النصوص المتاحة للإجابة على هذا السؤال. "
                "يمكنك إعادة صياغة السؤال أو تضييق نطاقه."
            ),
            "closest_heading": "أقرب الأدلة الموجودة:",
        }
    if language == "en":
        return {
            "lead": "The following evidence was retrieved from the sources for your question:",
            "note": (
                "Automated evidence summary (deterministic fallback) — no reasoned "
                "synthesis was produced because no language model was available at "
                "this time."
            ),
            "volume": "vol.",
            "page": "p.",
            "caveat": (
                "Automated, evidence-grounded answer produced without an LLM; "
                "consult a qualified scholar for a full ruling."
            ),
            "refusal": (
                "Not enough evidence was found in the available corpus to answer this "
                "question. You may rephrase the question or narrow its scope."
            ),
            "closest_heading": "Closest evidence found:",
        }
    return {
        "lead": "আপনার প্রশ্নের সাথে সম্পর্কিত উৎস থেকে নিম্নোক্ত প্রমাণ (evidence) পাওয়া গেছে:",
        "note": (
            "এটি একটি স্বয়ংক্রিয় প্রমাণ-সারসংক্ষেপ (deterministic fallback) — এই মুহূর্তে "
            "কোনো ভাষা মডেল উপলব্ধ না থাকায় যুক্তিসম্পন্ন বিশ্লেষণ তৈরি করা হয়নি।"
        ),
        "volume": "খণ্ড",
        "page": "পৃষ্ঠা",
        "caveat": (
            "এটি LLM ছাড়াই প্রমাণভিত্তিক স্বয়ংক্রিয় উত্তর; পূর্ণাঙ্গ ফতোয়ার জন্য যোগ্য একজন আলেমের পরামর্শ নিন।"
        ),
        "refusal": (
            "প্রদত্ত কোরপাসে এই প্রশ্নের উত্তর দেওয়ার মতো পর্যাপ্ত প্রমাণ পাওয়া যায়নি। "
            "অনুগ্রহ করে প্রশ্নটি পুনরায় লিখুন বা এর পরিধি সংকুচিত করুন।"
        ),
        "closest_heading": "সবচেয়ে কাছাকাছি প্রমাণ:",
    }
