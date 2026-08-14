"""Tests for the allowlist HTML sanitizer that guards LLM `explanation.html`.

The frontend renders this field through `dangerouslySetInnerHTML`, so the
sanitizer must neutralise scripts, event handlers, dangerous URL schemes and
entity-encoded payloads while preserving the deterministic evidence markup
(`<p>`, `<b>`, `<cite>`, ...).
"""

import pytest

from app.services.html_sanitizer import sanitize_html


def test_allowlisted_markup_survives() -> None:
    html = "<p><b>সারাংশ</b> <cite>Al-Hidayah</cite></p>\n<ul><li>প্রথম</li></ul>"
    assert sanitize_html(html) == html


def test_script_tag_is_stripped_content_stays_escaped_text() -> None:
    out = sanitize_html("<p>ok</p><script>alert(1)</script>")
    assert "<script>" not in out and "</script>" not in out
    assert out == "<p>ok</p>alert(1)"


def test_script_escaped_by_charref_cannot_reform() -> None:
    out = sanitize_html("&lt;script&gt;alert(1)&lt;/script&gt;")
    assert "<script>" not in out
    assert out == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_event_handler_attributes_are_dropped() -> None:
    out = sanitize_html('<p onclick="alert(1)">safe</p>')
    assert "onclick" not in out
    assert out == "<p>safe</p>"


def test_javascript_url_scheme_is_dropped() -> None:
    out = sanitize_html('<a href="javascript:alert(1)">x</a>')
    assert "javascript" not in out
    assert out == "<a>x</a>"


def test_entity_encoded_javascript_scheme_is_neutralised() -> None:
    out = sanitize_html('<a href="javascript&#58;alert(1)">x</a>')
    assert "javascript" not in out and "&#58;" not in out
    assert out == "<a>x</a>"


def test_data_and_invalid_schemes_are_dropped() -> None:
    for scheme in ("data:text/html;base64,PHNjcmlwdD4=", "vbscript:msgbox(1)"):
        out = sanitize_html(f'<a href="{scheme}">x</a>')
        assert "href" not in out
        assert out == "<a>x</a>"


def test_safe_https_href_is_preserved() -> None:
    out = sanitize_html('<a href="https://example.com/a?b=1">link</a>')
    assert out == '<a href="https://example.com/a?b=1">link</a>'


def test_dropped_tags_lose_their_token_only() -> None:
    out = sanitize_html("<style>h1{color:red}</style><img src=x><p>ok</p>")
    assert out == "h1{color:red}<p>ok</p>"


def test_comments_and_declarations_are_dropped() -> None:
    out = sanitize_html("<!-- nope --><p>ok</p>")
    assert "<!--" not in out and out == "<p>ok</p>"


def test_text_is_escaped_not_reinterpreted() -> None:
    out = sanitize_html("<p>5 &lt; 6 && true</p>")
    assert out == "<p>5 &lt; 6 &amp;&amp; true</p>"


def test_rtl_dir_attribute_is_preserved_for_arabic() -> None:
    out = sanitize_html('<p dir="rtl">نص</p>')
    assert out == '<p dir="rtl">نص</p>'


def test_unbalanced_markup_is_tolerated() -> None:
    out = sanitize_html("<p>unclosed <b>bold")
    assert "unclosed" in out and "bold" in out


def test_split_across_stream_tokens_is_safe_after_join() -> None:
    joined = "<scr" + "ipt>alert(1)</script><p>ok</p>"
    out = sanitize_html(joined)
    assert "<script>" not in out and out == "alert(1)<p>ok</p>"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_and_none_inputs_return_empty(value) -> None:
    assert sanitize_html(value).strip() == ""


def test_output_length_is_capped() -> None:
    out = sanitize_html("<p>" + "a" * 200_000 + "</p>")
    assert len(out) <= 64 * 1024 + 8
