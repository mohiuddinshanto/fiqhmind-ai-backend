"""Validation of LLM answer output (Phase 10 grounded loop).

`validate_llm_answer` enforces the structured-JSON contract and citation
coverage *before* the answer is returned to the client:
  - the output must parse as a single JSON object (markdown fences tolerated),
  - `answer_language` must match the requested language,
  - `explanation.html` must be non-empty,
  - every `arabic_quote.text` must be present,
  - every `citations[].chunk_id` must exist in the provided evidence,
  - every `[EVIDENCE_i]` reference inside the explanation must be backed by a
    citation to that chunk (citation coverage).

Confidence is deliberately *not* part of the LLM contract: it is computed
server-side by a deterministic function (ARCHITECTURE §Phase 10).
"""

import json
import re

_EVIDENCE_REFERENCE = re.compile(r"\[EVIDENCE_(\d+)\]")


class GenerationValidationError(Exception):
    """Raised when an LLM answer fails the schema or citation-coverage contract."""


def validate_llm_answer(
    raw: str,
    *,
    evidence_chunk_ids: list[str],
    answer_language: str,
) -> dict:
    """Parse and validate the raw LLM output; return the normalized payload.

    `evidence_chunk_ids` is ordered the same way the prompts module formatted
    the `[EVIDENCE_i]` blocks, so index i maps to evidence_chunk_ids[i - 1].
    """
    payload = _parse_json(raw)
    if not isinstance(payload, dict):
        raise GenerationValidationError("answer is not a JSON object")

    if payload.get("answer_language") != answer_language:
        raise GenerationValidationError(
            f"answer_language mismatch: expected {answer_language!r}, "
            f"got {payload.get('answer_language')!r}"
        )

    explanation = payload.get("explanation")
    if not isinstance(explanation, dict):
        raise GenerationValidationError("explanation must be an object")
    html = explanation.get("html")
    if not isinstance(html, str) or not html.strip():
        raise GenerationValidationError("explanation.html is empty")
    explanation.setdefault("type", "markdown")

    quotes = payload.get("arabic_quotes")
    if quotes is None:
        quotes = []
    if not isinstance(quotes, list):
        raise GenerationValidationError("arabic_quotes must be a list")
    for quote in quotes:
        text = quote.get("text") if isinstance(quote, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise GenerationValidationError("arabic_quote text is missing")
    payload["arabic_quotes"] = quotes

    citations = payload.get("citations")
    if citations is None:
        citations = []
    if not isinstance(citations, list):
        raise GenerationValidationError("citations must be a list")

    evidence_ids = set(evidence_chunk_ids)
    normalized_citations = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise GenerationValidationError("citation must be an object")
        chunk_id = citation.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise GenerationValidationError("citation is missing chunk_id")
        if chunk_id not in evidence_ids:
            raise GenerationValidationError(f"citation references unknown chunk {chunk_id!r}")
        normalized_citations.append(_normalize_citation(citation))
    payload["citations"] = normalized_citations

    _check_citation_coverage(html, normalized_citations, evidence_chunk_ids)

    refusal = payload.get("refusal")
    if refusal is not None:
        if not isinstance(refusal, dict):
            raise GenerationValidationError("refusal must be an object or null")
        reason = refusal.get("reason")
        if reason != "insufficient_evidence":
            raise GenerationValidationError(f"unsupported refusal reason {reason!r}")
        closest = refusal.get("closest_evidence")
        if closest is None:
            refusal["closest_evidence"] = []
        elif not isinstance(closest, list):
            raise GenerationValidationError("refusal.closest_evidence must be a list")

    for key in ("caveats", "related"):
        value = payload.get(key)
        if value is None:
            payload[key] = []
        elif not isinstance(value, list):
            raise GenerationValidationError(f"{key} must be a list")

    return payload


def _parse_json(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationValidationError(f"invalid JSON: {exc}") from exc


def _normalize_citation(citation: dict) -> dict:
    """Coerce numeric page/volume values to strings for the stable schema."""
    normalized = dict(citation)
    for key in ("page", "volume"):
        value = normalized.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[key] = str(value)
    return normalized


def _check_citation_coverage(
    html: str, citations: list[dict], evidence_chunk_ids: list[str]
) -> None:
    """Every evidence block referenced in the explanation must be cited."""
    index_to_chunk = {i + 1: chunk_id for i, chunk_id in enumerate(evidence_chunk_ids)}
    cited = {citation["chunk_id"] for citation in citations}
    for match in _EVIDENCE_REFERENCE.finditer(html):
        index = int(match.group(1))
        chunk_id = index_to_chunk.get(index)
        if chunk_id is None:
            raise GenerationValidationError(f"explanation references unknown [EVIDENCE_{index}]")
        if chunk_id not in cited:
            raise GenerationValidationError(
                f"explanation references {match.group(0)} but {chunk_id!r} is not cited"
            )
