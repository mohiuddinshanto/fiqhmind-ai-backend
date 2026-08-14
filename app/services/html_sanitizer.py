"""Allowlist HTML sanitization for LLM-generated answer markup.

The LLM writes `explanation.html`, the answer is persisted in chat history, and
the frontend renders it through `dangerouslySetInnerHTML`
(frontend/app/page.tsx). A hostile, confused, or prompt-injected model can emit
`<script>`, event handlers, or `javascript:` URLs. Rather than trust that
output, every fragment is passed through this dependency-free allowlist
sanitizer before it is stored or served.

Policy:
  - text is always HTML-escaped, so nothing in element content can become markup;
  - only the tags in `_ALLOWED_TAGS` survive; everything else (scripts, styles,
    images, iframes, comments, declarations) is dropped with its token;
  - only the attributes in `_ALLOWED_ATTRIBUTES` survive, and only `href` is
    meaningful: it must match an explicit http(s)/mailto scheme. Every allowed
    attribute value is re-escaped on output so entity-encoded payloads (e.g.
    `javascript&#58;`) can never reform into active URLs.
"""

import html
import re
from html.parser import HTMLParser

_ALLOWED_TAGS = frozenset(
    {
        "p",
        "br",
        "b",
        "strong",
        "i",
        "em",
        "u",
        "s",
        "blockquote",
        "cite",
        "code",
        "pre",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "span",
        "a",
    }
)

_ALLOWED_ATTRIBUTES = frozenset({"href", "dir", "lang"})

_SAFE_SCHEMES = re.compile(r"^(https?:|mailto:)", re.IGNORECASE)

_MAX_OUTPUT_LENGTH = 64 * 1024


class _Sanitizer(HTMLParser):
    """Token walker that rebuilds only allowlisted, inert markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._length = 0
        self._truncated = False

    def _emit(self, token: str) -> None:
        if self._truncated:
            return
        remaining = _MAX_OUTPUT_LENGTH - self._length
        if len(token) > remaining:
            self._parts.append(token[:remaining])
            self._truncated = True
            return
        self._parts.append(token)
        self._length += len(token)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _ALLOWED_TAGS:
            self._emit(f"<{tag}{self._render_attributes(attrs)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _ALLOWED_TAGS:
            self._emit(f"<{tag}{self._render_attributes(attrs)}/>")

    def handle_endtag(self, tag: str) -> None:
        if tag in _ALLOWED_TAGS:
            self._emit(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if data:
            self._emit(html.escape(data))

    def handle_comment(self, data: str) -> None:  # noqa: ARG002 - comments never survive
        pass

    def handle_decl(self, decl: str) -> None:  # noqa: ARG002 - declarations never survive
        pass

    def handle_pi(self, data: str) -> None:  # noqa: ARG002 - processing instructions never survive
        pass

    @staticmethod
    def _render_attributes(attrs: list[tuple[str, str | None]]) -> str:
        rendered: list[str] = []
        for name, value in attrs:
            if name not in _ALLOWED_ATTRIBUTES:
                continue
            value = value or ""
            if name == "href" and not _SAFE_SCHEMES.match(value.strip()):
                continue
            rendered.append(f'{name}="{html.escape(value, quote=True)}"')
        return f" {' '.join(rendered)}" if rendered else ""


def sanitize_html(fragment: str) -> str:
    """Return `fragment` reduced to allowlisted, inert HTML.

    Disallowed tags are dropped token-by-token; their text content survives as
    escaped plain text. Unbalanced or malformed markup is tolerated by the
    parser and never raises.
    """
    sanitizer = _Sanitizer()
    try:
        sanitizer.feed(fragment or "")
        sanitizer.close()
    except Exception:  # pragma: no cover - parser is lenient; fail closed anyway
        return ""
    return "".join(sanitizer._parts)
