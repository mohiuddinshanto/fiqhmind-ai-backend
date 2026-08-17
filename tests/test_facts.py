"""Tests for question-aware deterministic fact extraction (Phase 10).

The deterministic fallback must answer concisely *from the evidence* — never
dump every chunk as a raw `[EVIDENCE_i]` block — and must never invent
information.
"""

from app.services.generation.facts import (
    _clean_text,
    classify_query,
    extract_fact,
)
from app.services.retrieval import RetrievedChunk


def _chunk(
    chunk_id: str = "c1",
    *,
    text: str = "الماء طهور لا ينجسه شيء",
    book: str = "Al-Hidayah",
    volume: str = "1",
    page: int = 5,
    topic: str = "طهارة",
    score: float = 0.9,
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
        region="main",
        lang="bn",
        verified=True,
        rerank_score=score,
        kitab=kitab,
        bab=bab,
    )


# ------------------------------------------------------------ classification


def test_classify_query_detects_author_in_bengali() -> None:
    assert classify_query("আছারুল হাদীসের লেখক কে?") == "author"
    assert classify_query("এই কিতাবটি কে লিখেছেন?") == "author"


def test_classify_query_detects_author_in_arabic() -> None:
    assert classify_query("من هو مؤلف هذا الكتاب؟") == "author"


def test_classify_query_detects_title() -> None:
    assert classify_query("এই কিতাবের নাম কী?") == "title"
    assert classify_query("আছারুল হাদীস কিতাবটির নাম কি?") == "title"


def test_classify_query_detects_publisher() -> None:
    assert classify_query("কিতাবটি কে প্রকাশ করেছে?") == "publisher"


def test_classify_query_detects_page_count() -> None:
    assert classify_query("আছারুল হাদীসে মোট কত পৃষ্ঠা?") == "page_count"


def test_classify_query_falls_back_to_explanation() -> None:
    assert classify_query("ওযুর নিয়ম কি") == "explanation"
    assert classify_query("") == "explanation"


# -------------------------------------------------------------- extraction


def test_extract_author_from_verbatim_chunk() -> None:
    chunks = [
        _chunk(
            "c1",
            text=(
                "Kitab: Ahsanul Hadith\n"
                "Topic: ১. ভূমিকা\n\n"
                "শায়খ মুহাম্মাদ আওয়ামাহ রচিত "
                "'আছারুল হাদীসিশ শরীফ ফী যামিল মাজালিস' নামক একটি গুরুত্বপূর্ণ গ্রন্থ।"
            ),
        )
    ]
    fact = extract_fact(chunks, query="লেখক কে?", language="bn")

    assert fact.fact_type == "author"
    assert "শায়খ মুহাম্মাদ আওয়ামাহ" in fact.answer
    assert fact.strength == 1.0
    assert fact.chunk_id == "c1"
    assert fact.quote and "শায়খ মুহাম্মাদ আওয়ামাহ রচিত" in fact.quote


def test_extract_title_from_book_metadata() -> None:
    chunks = [_chunk("c1", book="আছারুল হাদীস প্রশ্নোত্তর")]
    fact = extract_fact(chunks, query="কিতাবের নাম কী?", language="bn")

    assert fact.fact_type == "title"
    assert "আছারুল হাদীস প্রশ্নোত্তর" in fact.answer
    assert fact.strength == 0.8
    assert fact.chunk_id == "c1"


def test_extract_publisher_graceful_when_absent() -> None:
    chunks = [_chunk("c1", text="এতে প্রকাশকের কোনো তথ্য নেই।")]
    fact = extract_fact(chunks, query="প্রকাশক কে?", language="bn")

    assert fact.fact_type == "publisher"
    assert "পাওয়া যায়নি" in fact.answer
    assert fact.strength == 0.4
    assert fact.chunk_id == "c1"


def test_extract_page_count_handles_bengali_digits() -> None:
    chunks = [_chunk("c1", text="গ্রন্থটি মোট ৬৪৮ পৃষ্ঠার একটি বড় সংকলন।")]
    fact = extract_fact(chunks, query="মোট কত পৃষ্ঠা?", language="bn")

    assert fact.fact_type == "page_count"
    assert "648" in fact.answer
    assert fact.strength == 1.0


def test_extract_explanation_gist_for_general_question() -> None:
    chunks = [
        _chunk("c1", text="الماء طهور لا ينجسه شيء।"),
        _chunk("c2", text="الوضوء شرط الصلاة।", topic="طهارة", page=9),
    ]
    fact = extract_fact(chunks, query="ওযুর নিয়ম কি", language="bn")

    assert fact.fact_type == "explanation"
    assert fact.answer
    assert fact.quote
    assert fact.chunk_id == "c1"


def test_extract_uses_strongest_chunk_first() -> None:
    weak = _chunk("weak", score=0.2, text="অপ্রাসঙ্গিক লেখা এখানে।")
    strong = _chunk(
        "strong",
        score=0.9,
        text="শায়খ মুহাম্মাদ আওয়ামাহ রচিত একটি বড় গ্রন্থ।",
    )
    fact = extract_fact([weak, strong], query="লেখক?", language="bn")

    assert fact.chunk_id == "strong"


def test_clean_text_strips_metadata_headers() -> None:
    cleaned = _clean_text(
        "Kitab: Ahsanul Hadith\nTopic: ১. ভূমিকা\n\nএখানে মূল লেখা শুরু।"
    )
    assert cleaned.startswith("এখানে মূল লেখা শুরু")
    assert "Kitab:" not in cleaned
