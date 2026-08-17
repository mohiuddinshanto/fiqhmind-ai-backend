"""Question-aware deterministic fact extraction (Phase 10).

The deterministic fallback used to dump every retrieved chunk as a raw
evidence block. This module instead classifies the user's intent (author,
title, publisher, page count or a general explanation) and answers *from the
evidence*: everything it returns is either drawn from the chunk payload
(book metadata) or is a verbatim excerpt of the chunk text. It never invents
information and never renders raw ``[EVIDENCE_i]`` blocks.

Two entry points are used:

  - `classify_query` returns the intent for a question.
  - `extract_fact` produces an `ExtractedFact` for a reranked chunk list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.retrieval import RetrievedChunk

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Chunk text is prefixed with metadata header lines ("Kitab: …", "Topic: …").
# They carry no answerable content, so they are stripped before matching.
_LEADING_METADATA = re.compile(
    r"^\s*(?:Kitab|Topic|Bab|Fasl|Book|Chapter|Section|বই|অধ্যায়|কিতাব|প্রকরণ|ভূমিকা|পৃষ্ঠা)\s*[:ঃ]"
)

# Fixed stopwords that add no signal when scoring sentences against the query.
_STOPWORDS = {
    "কী", "কি", "কে", "কেন", "কেমন", "কত", "এই", "এটা", "এটি", "এ",
    "সে", "তার", "তাদের", "মোট", "সম্বন্ধে", "বিষয়ে", "নিয়ে", "বলে",
    "আছে", "ছিল", "হয়", "হলো", "উত্তর", "প্রশ্ন", "দয়া", "করে", "দিয়ে",
    "the", "a", "an", "is", "are", "was", "of", "in", "on", "for", "to",
    "ما", "هو", "هي", "هل", "عن", "في", "من", "على",
}

# Intent detection: each needle is a substring of the normalized query.
# Ordered so a specific intent wins over a generic one.
_FACT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "author",
        (
            "লেখক", "লেখখক", "লেখিকা", "রচয়িতা", "রচনা করেছেন", "লিখেছেন",
            "author", "written by", "who wrote", "authored by",
            "المؤلف", "مؤلف", "تأليف", "من كتب", "المصنف",
        ),
    ),
    (
        "title",
        (
            "কিতাবের নাম", "কিতাবটির নাম", "কিতাবটার নাম", "গ্রন্থের নাম",
            "বইয়ের নাম", "বইটির নাম", "নাম কী", "নাম কি", "নামটা কী",
            "book name", "book title", "called",
            "اسم الكتاب", "اسم الکتاب", "عنوان الكتاب",
        ),
    ),
    (
        "publisher",
        (
            "প্রকাশক", "প্রকাশনী", "প্রকাশনা", "প্রকাশ", "নাশের", "প্রকাশকের নাম",
            "publisher", "published by", "الناشر", "دار النشر", "المطبع", "الطباعة",
        ),
    ),
    (
        "page_count",
        (
            "কত পৃষ্ঠা", "কত পাতা", "পৃষ্ঠা সংখ্যা", "পাতা সংখ্যা",
            "কয় পৃষ্ঠা", "কত পৃষ্ঠার", "মোট পৃষ্ঠা",
            "how many pages", "number of pages",
            "عدد الصفحات", "كم صفحة", "عدد الأوراق",
        ),
    ),
]

# Author / publisher extraction: first pattern that matches wins (chunks are
# scanned in rerank order, so the strongest evidence is preferred).
_AUTHOR_PATTERNS = [
    re.compile(r"([\u0980-\u09FF]{2,25}(?:\s+[\u0980-\u09FF]{2,25}){0,3})\s+রচিত"),
    re.compile(r"([\u0980-\u09FF]{2,25}(?:\s+[\u0980-\u09FF]{2,25}){0,3})\s+লিখেছেন"),
    re.compile(r"লেখক\s*[:ঃ]\s*([^\n।]{2,70})"),
    re.compile(r"লেখিকা\s*[:ঃ]\s*([^\n।]{2,70})"),
    re.compile(r"রচয়িতা\s*[:ঃ]?\s*([^\n।]{2,70})"),
    re.compile(r"(?:المؤلف|مؤلف الكتاب)\s*[:ः]?\s*([^\n]{2,70})"),
    re.compile(r"تأليف\s*[:ः]?\s*([^\n]{2,70})"),
    re.compile(r"(?:written by|authored by|author)\s*[:ঃ]?\s*([^\n]{2,70})", re.IGNORECASE),
]

_PUBLISHER_PATTERNS = [
    re.compile(r"প্রকাশক\s*[:ঃ]\s*([^\n।]{2,70})"),
    re.compile(r"প্রকাশনী\s*[:ঃ]\s*([^\n।]{2,70})"),
    re.compile(r"(?:الناشر|دار النشر)\s*[:ः]?\s*([^\n]{2,70})"),
    re.compile(r"(?:published by)\s*[:ঃ]?\s*([^\n]{2,70})", re.IGNORECASE),
]

_PAGE_PATTERNS = [
    re.compile(r"(\d{1,4})\s*(?:পৃষ্ঠা|পাতা|صفحة|pages?)", re.IGNORECASE),
    re.compile(r"(?:পৃষ্ঠা|পাতা|صفحة|pages?)\s*[:ঃ]?\s*(\d{1,4})\b", re.IGNORECASE),
]

_TITLE_AFTER_AUTHOR = re.compile(
    r"রচিত\s*['\"""''']?\s*([\u0980-\u09FF\u0600-\u06FF0-9A-Za-z\s:-]{4,90})"
)

# Leading filler words that can sit in front of an extracted name.
_NAME_FILLER = {
    "গ্রন্থটি", "কিতাবটি", "বইটি", "এই", "উক্ত", "ইহা", "এটি", "এটা",
    "তার", "তাহার", "ড.", "প্রফেসর", "ডক্টর", "মুহতারাম", "খলিফা",
}


@dataclass(frozen=True)
class ExtractedFact:
    """A concise, evidence-grounded answer for one user question."""

    fact_type: str
    heading: str
    answer: str
    quote: str | None
    chunk_id: str | None
    strength: float  # 1.0 verbatim fact; 0.6–0.8 metadata/partial; ≤0.5 generic gist


def classify_query(query: str) -> str:
    """Return the intent of `query`: author|title|publisher|page_count|explanation."""
    q = (query or "").strip().lower()
    if not q:
        return "explanation"
    for fact_type, needles in _FACT_PATTERNS:
        if any(needle in q for needle in needles):
            return fact_type
    return "explanation"


def extract_fact(
    chunks: list[RetrievedChunk], *, query: str, language: str = "bn"
) -> ExtractedFact:
    """Extract a concise answer from the reranked `chunks` for `query`."""
    fact_type = classify_query(query)
    t = _templates(language)
    pool = [(chunk, _clean_text(chunk.text)) for chunk in chunks if (chunk.text or "").strip()]

    if fact_type == "author":
        return _author_fact(pool, t, language)
    if fact_type == "title":
        return _title_fact(chunks, pool, t, language)
    if fact_type == "publisher":
        return _publisher_fact(pool, t, language)
    if fact_type == "page_count":
        return _page_count_fact(pool, t, language)

    gist, quote, chunk_id = _explanation(pool, query, language)
    return ExtractedFact(
        fact_type="explanation",
        heading=t["answer"],
        answer=gist,
        quote=quote,
        chunk_id=chunk_id,
        strength=0.6 if quote else 0.5,
    )


def _author_fact(pool, t: dict[str, str], language: str) -> ExtractedFact:
    for chunk, text in pool:
        match = _first_match(_AUTHOR_PATTERNS, text)
        if not match:
            continue
        name = _clean_name(match.group(1))
        if not name:
            continue
        return ExtractedFact(
            fact_type="author",
            heading=t["author"],
            answer=_author_answer(name, language),
            quote=_shorten(_sentence_containing(text, match.start()), 180),
            chunk_id=chunk.chunk_id,
            strength=1.0,
        )
    return ExtractedFact(
        fact_type="author",
        heading=t["author"],
        answer=_not_found(language),
        quote=None,
        chunk_id=_best_chunk_id(pool),
        strength=0.4,
    )


def _title_fact(chunks, pool, t: dict[str, str], language: str) -> ExtractedFact:
    for chunk in chunks:
        name = _metadata_title(chunk)
        if name:
            return ExtractedFact(
                fact_type="title",
                heading=t["title"],
                answer=_title_answer(name, language),
                quote=None,
                chunk_id=chunk.chunk_id,
                strength=0.8,
            )
    for chunk, text in pool:
        match = _TITLE_AFTER_AUTHOR.search(text)
        if match:
            return ExtractedFact(
                fact_type="title",
                heading=t["title"],
                answer=_title_answer(match.group(1).strip(), language),
                quote=None,
                chunk_id=chunk.chunk_id,
                strength=0.7,
            )
    return ExtractedFact(
        fact_type="title",
        heading=t["title"],
        answer=_not_found(language),
        quote=None,
        chunk_id=_best_chunk_id(pool),
        strength=0.4,
    )


def _publisher_fact(pool, t: dict[str, str], language: str) -> ExtractedFact:
    for chunk, text in pool:
        match = _first_match(_PUBLISHER_PATTERNS, text)
        if not match:
            continue
        name = _clean_name(match.group(1))
        if not name:
            continue
        return ExtractedFact(
            fact_type="publisher",
            heading=t["publisher"],
            answer=_publisher_answer(name, language),
            quote=_shorten(_sentence_containing(text, match.start()), 180),
            chunk_id=chunk.chunk_id,
            strength=1.0,
        )
    return ExtractedFact(
        fact_type="publisher",
        heading=t["publisher"],
        answer=_not_found(language),
        quote=None,
        chunk_id=_best_chunk_id(pool),
        strength=0.4,
    )


def _page_count_fact(pool, t: dict[str, str], language: str) -> ExtractedFact:
    for chunk, text in pool:
        match = _first_match(_PAGE_PATTERNS, text.translate(_BN_DIGITS))
        if not match:
            continue
        return ExtractedFact(
            fact_type="page_count",
            heading=t["page_count"],
            answer=_page_answer(match.group(1), language),
            quote=None,
            chunk_id=chunk.chunk_id,
            strength=1.0,
        )
    return ExtractedFact(
        fact_type="page_count",
        heading=t["page_count"],
        answer=_not_found(language),
        quote=None,
        chunk_id=_best_chunk_id(pool),
        strength=0.4,
    )


def _explanation(pool, query: str, language: str):
    """Concise gist for a general question: strongest matching sentence(s)."""
    if not pool:
        return "", None, None
    tokens = _query_tokens(query)
    best_chunk, best_text = pool[0]
    best_sentence, _ = _best_sentence(best_text, tokens)
    if not best_sentence:
        best_sentence = _shorten(best_text, 220)

    parts = [_shorten(best_sentence, 280)]

    if len(pool) > 1:
        second_chunk, second_text = pool[1]
        second_sentence, _ = _best_sentence(second_text, tokens)
        if second_sentence and second_sentence != best_sentence:
            parts.append(_shorten(second_sentence, 200))

    return " ".join(parts), _shorten(best_sentence, 180), best_chunk.chunk_id


# ------------------------------------------------------------- phrase builders


def _author_answer(name: str, language: str) -> str:
    if language == "ar":
        return f"مؤلف هذا الكتاب هو {name}."
    if language == "en":
        return f"The author of this book is {name}."
    return f"এই কিতাবের লেখক {name}।"


def _title_answer(name: str, language: str) -> str:
    if language == "ar":
        return f"اسم هذا الكتاب: {name}."
    if language == "en":
        return f"The title of this book is: {name}."
    return f"এই কিতাবের নাম {name}।"


def _publisher_answer(name: str, language: str) -> str:
    if language == "ar":
        return f"ناشر هذا الكتاب: {name}."
    if language == "en":
        return f"This book is published by {name}."
    return f"এই কিতাবটি প্রকাশ করেছে {name}।"


def _page_answer(count: str, language: str) -> str:
    if language == "ar":
        return f"يحتوي هذا الكتاب على {count} صفحة."
    if language == "en":
        return f"This book has {count} pages."
    return f"এই কিতাবে {count} পৃষ্ঠা রয়েছে।"


def _not_found(language: str) -> str:
    if language == "ar":
        return "لم نعثر على هذه المعلومة في الأدلة المتاحة."
    if language == "en":
        return "This information was not found in the available evidence."
    return "প্রদত্ত প্রমাণে এই তথ্যটি পাওয়া যায়নি।"


# -------------------------------------------------------------- text helpers


def _templates(language: str) -> dict[str, str]:
    if language == "ar":
        return {
            "author": "المؤلف",
            "title": "اسم الكتاب",
            "publisher": "الناشر",
            "page_count": "عدد الصفحات",
            "answer": "الجواب",
        }
    if language == "en":
        return {
            "author": "Author",
            "title": "Book title",
            "publisher": "Publisher",
            "page_count": "Pages",
            "answer": "Answer",
        }
    return {
        "author": "লেখক",
        "title": "কিতাবের নাম",
        "publisher": "প্রকাশক",
        "page_count": "পৃষ্ঠা সংখ্যা",
        "answer": "উত্তর",
    }


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    while lines and _LEADING_METADATA.match(lines[0]):
        lines.pop(0)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[।.?؟])\s+", text) if part.strip()]


def _query_tokens(query: str) -> set[str]:
    words = re.findall(r"[\w\u0980-\u09FF\u0600-\u06FF]+", (query or "").lower())
    return {word for word in words if word not in _STOPWORDS}


def _score_sentence(sentence: str, tokens: set[str]) -> int:
    words = re.findall(r"[\w\u0980-\u09FF\u0600-\u06FF]+", sentence.lower())
    return sum(1 for word in words if word in tokens)


def _best_sentence(text: str, tokens: set[str]) -> tuple[str, int]:
    sentences = _sentences(text)
    if not sentences:
        return _shorten(text, 220), 0
    ranked = sorted(
        (
            (_score_sentence(sentence, tokens), index, sentence)
            for index, sentence in enumerate(sentences)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    score, _, sentence = ranked[0]
    return sentence, score


def _sentence_containing(text: str, pos: int) -> str:
    start = max(text.rfind("।", 0, pos), text.rfind(".", 0, pos)) + 1
    end = text.find("।", pos)
    if end == -1:
        end = text.find(".", pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _clean_name(name: str) -> str:
    words = (name or "").strip().strip(":-ঃ،,।").split()
    while words and words[0] in _NAME_FILLER:
        words.pop(0)
    return " ".join(words).strip()


def _metadata_title(chunk: RetrievedChunk) -> str | None:
    name = (chunk.book_name or "").strip()
    if not name or name.lower() in {"unknown", "unknown book"}:
        return None
    return name


def _shorten(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _first_match(patterns: list[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def _best_chunk_id(pool) -> str | None:
    return pool[0][0].chunk_id if pool else None
