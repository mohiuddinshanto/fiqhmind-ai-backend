"""Phase 17 metrics (ARCHITECTURE §17.2) — all computed with free/open tooling.

Every metric is a pure function over already-collected artifacts so the same
aggregates can be recomputed offline from a persisted `eval_run_results` row.
Retrieval metrics (Recall@K, MRR, nDCG@10) compare the *ordered* retrieved
chunk ids against the resolved gold chunk ids; citation accuracy compares the
generated answer's citations to the gold citation anchors; groundedness and
hallucination rate are rule-based sentence-level checks; refusal precision /
recall is a confusion matrix over the refusal flag; latency reports p50/p95.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field

_MIN_OVERLAP_TOKENS = 2


def _tokens(text: str) -> set[str]:
    """Normalize and tokenize text for overlap checks (multilingual, robust).

    Keeps alphanumeric runs only; folds case; strips Arabic diacritics and
    Bengali combining marks so a match doesn't depend on exact vowel marks.
    """
    text = re.sub(r"[\u064B-\u0652\u0670\u200C\u200D\uFE70-\uFEFF]", "", text.lower())
    return set(re.findall(r"[\w\u0600-\u06FF\u0980-\u09FF]+", text))


def _sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[।.!؟])\s+|\n+", text) if part.strip()]


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """1.0 when any expected chunk is in the top-k retrieved, else 0.0."""
    if not expected:
        return 0.0
    return 1.0 if expected & set(retrieved[:k]) else 0.0


def mrr(retrieved: list[str], expected: set[str]) -> float:
    """Reciprocal rank of the first expected chunk; 0.0 when none is retrieved."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG: rewards expected chunks ranking higher."""
    if not expected:
        return 0.0
    dcg = sum(
        1.0 / (math.log2(rank + 1))
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in expected
    )
    ideal = sum(1.0 / (math.log2(rank + 1)) for rank in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0


def _normalize_book(book: str) -> str:
    return re.sub(r"\s+", " ", book.strip().lower())


def _citations_match(
    actual_book: str,
    actual_volume: str | None,
    actual_page: str | None,
    expected_book: str,
    expected_volume: str | None,
    expected_page: int | None,
) -> bool:
    """Exact/fuzzy citation match: book must match (case/space-insensitive);
    volume and page match when present, page tolerating ±1 (fuzzy, per §17.2)."""
    if _normalize_book(actual_book) != _normalize_book(expected_book):
        return False
    if expected_volume is not None and actual_volume is not None:
        if str(actual_volume).strip() != str(expected_volume).strip():
            return False
    if expected_page is not None and actual_page is not None:
        try:
            actual = int(str(actual_page).strip())
        except ValueError:
            return False
        if abs(actual - expected_page) > 1:
            return False
    return True


def citation_accuracy(
    answer_citations: list[dict],
    expected_citations: list[dict],
) -> float:
    """Fraction of expected citations matched by at least one answer citation."""
    if not expected_citations:
        return 0.0
    matched = 0
    for expected in expected_citations:
        for actual in answer_citations:
            if _citations_match(
                actual.get("book") or "",
                actual.get("volume"),
                actual.get("page"),
                expected.get("book") or "",
                expected.get("volume"),
                expected.get("page"),
            ):
                matched += 1
                break
    return matched / len(expected_citations)


def groundedness(explanation: str, cited_chunk_texts: list[str]) -> float:
    """Fraction of explanation sentences that map to a *cited* chunk.

    Rule-based (§17.2): a sentence is grounded when it shares at least two
    normalized tokens with one of the chunks the answer actually cites.
    """
    sentences = _sentence_split(explanation)
    if not sentences:
        return 1.0 if not cited_chunk_texts else 0.0
    cited_token_sets = [_tokens(text) for text in cited_chunk_texts]
    grounded_count = sum(
        1
        for sentence in sentences
        if any(
            len(_tokens(sentence) & cited) >= _MIN_OVERLAP_TOKENS
            for cited in cited_token_sets
        )
    )
    return grounded_count / len(sentences)


def hallucination_rate(explanation: str, evidence_texts: list[str]) -> float:
    """Fraction of explanation sentences not traceable to ANY retrieved chunk.

    Complement of grounding against the full evidence window: any claim that
    shares no tokens with the retrieved context is a hallucination event (§17.2).
    """
    sentences = _sentence_split(explanation)
    if not sentences:
        return 0.0
    if not evidence_texts:
        return 1.0
    evidence_token_sets = [_tokens(text) for text in evidence_texts]
    unsupported = sum(
        1
        for sentence in sentences
        if not any(
            len(_tokens(sentence) & evidence) >= _MIN_OVERLAP_TOKENS
            for evidence in evidence_token_sets
        )
    )
    return unsupported / len(sentences)


@dataclass
class RefusalMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def refusal_confusion(pairs: list[tuple[bool, bool]]) -> RefusalMatrix:
    """Confusion matrix over (expect_refusal, refusal_given) pairs."""
    matrix = RefusalMatrix()
    for expect, given in pairs:
        if expect and given:
            matrix.tp += 1
        elif not expect and given:
            matrix.fp += 1
        elif expect and not given:
            matrix.fn += 1
        else:
            matrix.tn += 1
    return matrix


@dataclass
class LatencyPercentiles:
    p50: float = 0.0
    p95: float = 0.0


def latency_percentiles(latencies_ms: list[float]) -> LatencyPercentiles:
    if not latencies_ms:
        return LatencyPercentiles()
    ordered = sorted(latencies_ms)
    return LatencyPercentiles(
        p50=statistics.median(ordered),
        p95=_percentile(ordered, 0.95),
    )


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(len(ordered) * q)))
    return ordered[index]


@dataclass
class AggregateMetrics:
    """Aggregates every item-level score into the run-level numbers (§17.2)."""

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    citation_accuracy: float = 0.0
    groundedness: float = 0.0
    hallucination_rate: float = 0.0
    refusal_precision: float = 1.0
    refusal_recall: float = 1.0
    answer_accuracy: float = 0.0
    latency_ms: LatencyPercentiles = field(default_factory=LatencyPercentiles)

    def to_dict(self) -> dict:
        return {
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "recall@20": round(self.recall_at_20, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "groundedness": round(self.groundedness, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "refusal_precision": round(self.refusal_precision, 4),
            "refusal_recall": round(self.refusal_recall, 4),
            "answer_accuracy": round(self.answer_accuracy, 4),
            "latency_p50_ms": round(self.latency_ms.p50, 2),
            "latency_p95_ms": round(self.latency_ms.p95, 2),
        }
