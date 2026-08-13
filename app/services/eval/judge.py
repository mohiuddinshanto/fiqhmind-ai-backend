"""LLM-as-judge for Answer Accuracy (ARCHITECTURE §17.2, §17.3).

The judge scores a generated explanation (0 = wrong, 1 = partially right,
2 = correct) against the gold expected-answer summary, using a strict bounded
rubric. It reuses the same free Gemini Flash tier already budgeted for
production via the provider port (`LLMProvider.complete`). When no provider is
configured the judge falls back to a deterministic lexical scorer (token F1
between the explanation and the gold summary), so the harness produces a number
in every environment — including CI without a key.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from app.services.generation.providers import LLMProvider

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, bounded fiqh-answer evaluator. Score the model's answer "
    "against the gold summary using ONLY these three scores:\n"
    "0 = wrong or fabricated (core claim contradicts or is absent from gold),\n"
    "1 = partially right (some correct content, missing key ruling or adding "
    "minor unsupported detail),\n"
    "2 = correct (semantically matches the gold summary, no fabrication).\n"
    "Respond with a single JSON object of the form "
    '{"score": 0|1|2, "rationale": "one short sentence"}. Do not include '
    "anything else."
)


class AnswerJudge(Protocol):
    def score(self, *, question: str, explanation: str, expected_answer: str) -> int:
        """Return an integer 0-2 per the rubric above."""
        ...


class LLMAnswerJudge:
    """Rubric-graded LLM judge using the shared provider port (§17.3)."""

    def __init__(self, provider: LLMProvider, *, timeout_seconds: float = 30.0) -> None:
        self._provider = provider
        self._timeout = timeout_seconds

    def score(self, *, question: str, explanation: str, expected_answer: str) -> int:
        user_prompt = (
            f"QUESTION: {question}\n\n"
            f"GOLD EXPECTED ANSWER SUMMARY: {expected_answer}\n\n"
            f"MODEL ANSWER: {explanation}\n\n"
            "Score the model answer against the gold summary with the rubric above."
        )
        try:
            raw = self._provider.complete(
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception:  # noqa: BLE001 - a judge failure must not sink the eval run
            return 0
        return _parse_score(raw)


class LexicalAnswerJudge:
    """Deterministic fallback: token-F1 between explanation and gold summary.

    Used only when no provider key is configured (tests, keyless CI). It is a
    weak proxy for semantic correctness and is documented as such; the real
    metric is the LLM judge.
    """

    def score(self, *, question: str, explanation: str, expected_answer: str) -> int:
        del question  # lexical judge ignores the question text
        exp_tokens = _tokens(expected_answer)
        if not exp_tokens:
            return 0
        got_tokens = _tokens(explanation)
        if not got_tokens:
            return 0
        overlap = exp_tokens & got_tokens
        if not overlap:
            return 0
        precision = len(overlap) / len(got_tokens)
        recall = len(overlap) / len(exp_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 >= 0.5:
            return 2
        if f1 >= 0.2:
            return 1
        return 0


def get_answer_judge(provider: LLMProvider | None) -> AnswerJudge:
    """Pick the LLM judge when a provider is configured, else the lexical one."""
    if provider is not None:
        return LLMAnswerJudge(provider)
    return LexicalAnswerJudge()


def _parse_score(raw: str) -> int:
    """Extract an integer 0-2 from possibly-malformed judge output."""
    match = re.search(r"\"?score\"?\s*[:=]\s*([012])", raw)
    if match:
        return int(match.group(1))
    stripped = raw.strip().strip('"')
    if stripped in ("0", "1", "2"):
        return int(stripped)
    try:
        payload = json.loads(raw)
        value = payload.get("score")
        if isinstance(value, int) and value in (0, 1, 2):
            return value
    except (json.JSONDecodeError, AttributeError):
        pass
    return 0


def _tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[\u064B-\u0652\u0670\u200C\u200D]", "", text.lower())
    return set(re.findall(r"[\w\u0600-\u06FF\u0980-\u09FF]+", cleaned))
