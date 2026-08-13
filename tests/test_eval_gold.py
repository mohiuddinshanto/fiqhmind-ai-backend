"""Tests for the Phase 17 gold-set schema and loader (ARCHITECTURE §17.1)."""

import json

import pytest

from app.services.eval.gold import GoldItem, load_gold_set

GOLD_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "gold_qa.jsonl"


def test_gold_set_loads_and_has_reasonable_starter_size() -> None:
    items = load_gold_set(GOLD_PATH)
    assert 30 <= len(items) <= 50, "starter gold set should have 30-50 items (ARCHITECTURE §17.1)"

    categories = {item.category for item in items}
    assert "should_refuse" in categories, "gold set must include should-refuse items"
    assert categories <= {
        "single_clear_ruling",
        "cross_chapter",
        "known_ikhtilaf",
        "should_refuse",
    }


def test_every_item_has_three_language_variants() -> None:
    items = load_gold_set(GOLD_PATH)
    for item in items:
        assert item.question.bn.strip(), item.id
        assert item.question.ar.strip(), item.id
        assert item.question.en.strip(), item.id


def test_refusal_items_have_no_citations_and_are_marked() -> None:
    items = load_gold_set(GOLD_PATH)
    for item in items:
        if item.category == "should_refuse":
            assert item.expect_refusal is True
            assert item.expected_citations == []
        else:
            assert item.expected_citations, item.id


def test_non_refusal_items_require_citations() -> None:
    with pytest.raises(ValueError):
        GoldItem(
            id="bad-1",
            category="single_clear_ruling",
            difficulty="easy",
            question={"bn": "q", "ar": "q", "en": "q"},
            expected_citations=[],
            expect_refusal=False,
        ).validate_consistency()


def test_duplicate_ids_rejected(tmp_path) -> None:
    path = tmp_path / "gold.jsonl"
    row = {
        "id": "dup",
        "category": "single_clear_ruling",
        "difficulty": "easy",
        "question": {"bn": "b", "ar": "a", "en": "e"},
        "expected_citations": [{"book": "Al-Mabsut", "page": 1}],
    }
    path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_gold_set(path)


def test_malformed_json_rejected(tmp_path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_gold_set(path)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_gold_set("/nonexistent/gold_qa.jsonl")
