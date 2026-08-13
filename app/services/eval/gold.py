"""Phase 17 gold-set schema and loader (ARCHITECTURE §17.1).

A gold item is one curated QA pair with the question in three languages
(bn/ar/en), the expected citation anchor(s), a short expected-answer summary,
and a difficulty/category tag. The set is stored as versioned JSONL in-repo
(`backend/eval/gold_qa.jsonl`) so changes are code-reviewed like any other
artifact, and grows continuously as books are ingested (ARCHITECTURE §17.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "single_clear_ruling",
    "cross_chapter",
    "known_ikhtilaf",
    "should_refuse",
]
Difficulty = Literal["easy", "medium", "hard"]


class GoldCitation(BaseModel):
    """Expected book/volume/printed-page anchor for one gold item."""

    book: str = Field(description="Book name exactly as stored in the corpus")
    volume: str | None = None
    page: int | None = None


class GoldQuestion(BaseModel):
    """The same underlying question phrased in Bengali, Arabic, and English.

    Three variants test cross-lingual retrieval directly: each is replayed
    through the live pipeline and scored independently (ARCHITECTURE §17.1).
    """

    bn: str
    ar: str
    en: str


class GoldItem(BaseModel):
    """One curated gold QA pair (ARCHITECTURE §17.1)."""

    id: str = Field(description="Stable id like `mabsut-0001`; unique across the set")
    book_id: str | None = Field(
        default=None, description="Corpus book_id when the item is scoped to one book"
    )
    category: Category
    difficulty: Difficulty
    question: GoldQuestion
    expected_citations: list[GoldCitation] = Field(default_factory=list)
    expected_answer: str | None = Field(
        default=None, description="Short expected-answer summary (gold summary)"
    )
    expect_refusal: bool = Field(
        default=False, description="True for `should_refuse` items: system must refuse"
    )

    def validate_consistency(self) -> None:
        if self.category == "should_refuse" and not self.expect_refusal:
            raise ValueError(
                f"gold item {self.id}: category='should_refuse' requires expect_refusal=True"
            )
        if self.category != "should_refuse" and not self.expected_citations:
            raise ValueError(
                f"gold item {self.id}: non-refusal item must list expected citations"
            )


def load_gold_set(path: str | Path) -> list[GoldItem]:
    """Load and validate a versioned gold_qa.jsonl file.

    Raises `ValidationError` for malformed rows and `ValueError` for
    consistency violations or duplicate ids.
    """
    gold_path = Path(path)
    if not gold_path.exists():
        raise FileNotFoundError(f"gold set not found: {gold_path}")

    items: list[GoldItem] = []
    seen: set[str] = set()
    with gold_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"gold_qa.jsonl line {line_no}: invalid JSON: {exc}") from exc
            item = GoldItem.model_validate(raw)
            item.validate_consistency()
            if item.id in seen:
                raise ValueError(f"gold_qa.jsonl: duplicate item id {item.id!r}")
            seen.add(item.id)
            items.append(item)

    if not items:
        raise ValueError(f"gold_qa.jsonl is empty: {gold_path}")
    return items
