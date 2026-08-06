"""Tests for the Phase 9 reranker layer."""

import pytest

from app.core.config import Settings
from app.services.reranker import DefaultReranker, get_reranker


def test_identical_text_scores_1() -> None:
    scores = DefaultReranker().score("ما حكم الوضوء", ["ما حكم الوضوء"])

    assert scores[0] == pytest.approx(1.0)


def test_disjoint_text_scores_0() -> None:
    scores = DefaultReranker().score("زكاة", ["قواعد الصرف والنحو"])

    assert scores[0] == pytest.approx(0.0)


def test_partial_overlap_scores_between_0_and_1() -> None:
    scores = DefaultReranker().score("حكم صلاة المسافر", ["المسافر يقصر الصلاة"])

    assert 0.0 < scores[0] < 1.0


def test_scores_are_case_insensitive() -> None:
    reranker = DefaultReranker()

    assert reranker.score("Wudu", ["wudu"]) == reranker.score("wudu", ["wudu"])


def test_empty_query_scores_zero() -> None:
    scores = DefaultReranker().score("", ["anything"])

    assert scores[0] == 0.0


def test_empty_text_scores_zero() -> None:
    scores = DefaultReranker().score("query", [""])

    assert scores[0] == 0.0


def test_scores_are_deterministic() -> None:
    reranker = DefaultReranker()
    texts = ["a b c", "b c d", "e f g"]

    assert reranker.score("b c", texts) == reranker.score("b c", texts)


def test_get_reranker_defaults_to_default() -> None:
    assert isinstance(get_reranker(Settings()), DefaultReranker)


def test_get_reranker_bge_not_wired() -> None:
    with pytest.raises(NotImplementedError):
        get_reranker(Settings(reranker_provider="bge_reranker_v2_m3"))


def test_get_reranker_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_reranker(Settings(reranker_provider="nope"))
