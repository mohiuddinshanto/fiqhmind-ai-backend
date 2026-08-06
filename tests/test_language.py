"""Tests for the Phase 9 language detection layer."""

import pytest

from app.core.config import Settings
from app.core.exceptions import TranslationError
from app.services.language import (
    HeuristicLanguageDetector,
    get_language_detector,
    has_arabic_script,
    require_language,
)


def test_detects_arabic_script() -> None:
    language = HeuristicLanguageDetector().detect("ما حكم الوضوء؟")

    assert language.code == "ar"
    assert language.confidence > 0.9


def test_detects_bengali_script() -> None:
    language = HeuristicLanguageDetector().detect("ওযুর নিয়ম কি?")

    assert language.code == "bn"
    assert language.confidence > 0.9


def test_detects_english_script() -> None:
    language = HeuristicLanguageDetector().detect("What is the rule of wudu?")

    assert language.code == "en"
    assert language.confidence == 1.0


def test_mixed_script_majority_wins() -> None:
    language = HeuristicLanguageDetector().detect("الوضوء wudu ওযু")

    assert language.code == "ar"


def test_no_letters_returns_other() -> None:
    language = HeuristicLanguageDetector().detect("12345 !!!")

    assert language.code == "other"
    assert language.confidence == 0.0


def test_has_arabic_script() -> None:
    assert has_arabic_script("وضوء")
    assert has_arabic_script("wudu وضوء")
    assert not has_arabic_script("wudu")


def test_require_language_rejects_undetectable() -> None:
    with pytest.raises(TranslationError):
        require_language("12345", HeuristicLanguageDetector())


def test_get_language_detector_defaults_to_heuristic() -> None:
    detector = get_language_detector(Settings())

    assert isinstance(detector, HeuristicLanguageDetector)


def test_get_language_detector_fasttext_not_wired() -> None:
    with pytest.raises(NotImplementedError):
        get_language_detector(Settings(language_detector_provider="fasttext"))


def test_get_language_detector_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_language_detector(Settings(language_detector_provider="nope"))
