"""Tests for the Phase 9 translation layer."""

from app.core.config import Settings
from app.services.translation import (
    PassthroughTranslator,
    TranslationResult,
    get_translator,
)


def test_passthrough_returns_input_unchanged() -> None:
    result = PassthroughTranslator().translate(
        "What is the rule of wudu?", source_lang="en"
    )

    assert isinstance(result, TranslationResult)
    assert result.text == "What is the rule of wudu?"
    assert result.source_lang == "en"
    assert result.target_lang == "ar"
    assert result.translated is False
    assert result.confidence == 1.0


def test_passthrough_keeps_arabic_intact() -> None:
    result = PassthroughTranslator().translate("ما حكم الوضوء؟", source_lang="ar")

    assert result.text == "ما حكم الوضوء؟"
    assert result.translated is False


def test_get_translator_defaults_to_passthrough() -> None:
    translator = get_translator(Settings())

    assert isinstance(translator, PassthroughTranslator)


def test_get_translator_external_providers_not_wired() -> None:
    for provider in ("google_free", "gemini"):
        try:
            get_translator(Settings(translator_provider=provider))
        except NotImplementedError:
            continue
        raise AssertionError(f"translator_provider={provider} should raise NotImplementedError")


def test_get_translator_unknown_provider() -> None:
    try:
        get_translator(Settings(translator_provider="nope"))
    except ValueError:
        return
    raise AssertionError("unknown translator_provider should raise ValueError")
