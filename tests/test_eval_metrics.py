"""Tests for Phase 17 metrics (ARCHITECTURE §17.2)."""

import pytest

from app.services.eval import metrics as m


class TestRetrievalMetrics:
    def test_recall_at_k_finds_expected_in_top_k(self) -> None:
        assert m.recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
        assert m.recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
        assert m.recall_at_k(["a", "b"], {"x"}, 20) == 0.0
        assert m.recall_at_k(["a", "b"], set(), 10) == 0.0

    def test_mrr_ranks_first_correct_hit(self) -> None:
        assert m.mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
        assert m.mrr(["a", "b"], {"b"}) == 0.5
        assert m.mrr(["a", "b"], {"x"}) == 0.0

    def test_ndcg_rewards_higher_ranking(self) -> None:
        good = m.ndcg_at_k(["a", "b", "c"], {"a"})
        bad = m.ndcg_at_k(["x", "y", "a"], {"a"})
        assert good == 1.0
        assert 0.0 < bad < 1.0

    def test_ndcg_empty_expected_is_zero(self) -> None:
        assert m.ndcg_at_k(["a"], set()) == 0.0


class TestCitationAccuracy:
    def test_exact_match(self) -> None:
        actual = [{"book": "Al-Mabsut", "volume": "1", "page": "3"}]
        expected = [{"book": "Al-Mabsut", "volume": "1", "page": 3}]
        assert m.citation_accuracy(actual, expected) == 1.0

    def test_fuzzy_page_within_one(self) -> None:
        actual = [{"book": "Al-Mabsut", "volume": "1", "page": "4"}]
        expected = [{"book": "Al-Mabsut", "volume": "1", "page": 3}]
        assert m.citation_accuracy(actual, expected) == 1.0

    def test_wrong_book_fails(self) -> None:
        actual = [{"book": "Al-Hidayah", "volume": "1", "page": "3"}]
        expected = [{"book": "Al-Mabsut", "volume": "1", "page": 3}]
        assert m.citation_accuracy(actual, expected) == 0.0

    def test_partial_coverage(self) -> None:
        actual = [{"book": "Al-Mabsut", "volume": "1", "page": "1"}]
        expected = [
            {"book": "Al-Mabsut", "volume": "1", "page": 1},
            {"book": "Al-Mabsut", "volume": "1", "page": 10},
        ]
        assert m.citation_accuracy(actual, expected) == 0.5

    def test_volume_mismatch_fails(self) -> None:
        actual = [{"book": "Al-Mabsut", "volume": "2", "page": "3"}]
        expected = [{"book": "Al-Mabsut", "volume": "1", "page": 3}]
        assert m.citation_accuracy(actual, expected) == 0.0

    def test_no_expected_is_zero(self) -> None:
        assert m.citation_accuracy([{"book": "Al-Mabsut"}], []) == 0.0


class TestGroundednessAndHallucination:
    def test_all_sentences_grounded(self) -> None:
        explanation = "Water is pure in itself. It purifies water for prayer."
        cited = ["Water is pure in itself and purifying for others."]
        assert m.groundedness(explanation, cited) == 1.0

    def test_unmapped_sentence_flagged(self) -> None:
        explanation = "Water is pure. The sky is blue today."
        cited = ["Water is pure in itself."]
        assert m.groundedness(explanation, cited) == 0.5

    def test_hallucination_rate_counts_unsupported_claims(self) -> None:
        explanation = "Water is pure. Aliens exist among us."
        evidence = ["Water is pure in itself and purifying for others."]
        assert m.hallucination_rate(explanation, evidence) == 0.5

    def test_hallucination_with_no_evidence_is_max(self) -> None:
        assert m.hallucination_rate("Anything at all.", []) == 1.0

    def test_empty_explanation_not_hallucination(self) -> None:
        assert m.hallucination_rate("", ["evidence"]) == 0.0


class TestRefusalAndLatency:
    def test_refusal_confusion_matrix(self) -> None:
        pairs = [
            (True, True),  # tp
            (False, True),  # fp
            (True, False),  # fn
            (False, False),  # tn
        ]
        matrix = m.refusal_confusion(pairs)
        assert (matrix.tp, matrix.fp, matrix.fn, matrix.tn) == (1, 1, 1, 1)
        assert matrix.precision == 0.5
        assert matrix.recall == 0.5

    def test_refusal_empty_defaults_to_perfect(self) -> None:
        matrix = m.RefusalMatrix()
        assert matrix.precision == 1.0
        assert matrix.recall == 1.0

    def test_latency_percentiles(self) -> None:
        result = m.latency_percentiles([100, 200, 300, 400, 500])
        assert result.p50 == 300
        assert result.p95 == 500

    def test_latency_empty(self) -> None:
        result = m.latency_percentiles([])
        assert result.p50 == 0.0
        assert result.p95 == 0.0


class TestAggregate:
    def test_aggregate_metrics_to_dict(self) -> None:
        aggregate = m.AggregateMetrics(
            recall_at_5=0.5,
            recall_at_10=0.6,
            recall_at_20=0.8,
            mrr=0.5,
            ndcg_at_10=0.5,
            citation_accuracy=0.5,
            groundedness=0.75,
            hallucination_rate=0.25,
            refusal_precision=0.8,
            refusal_recall=0.8,
            answer_accuracy=1.0,
            latency_ms=m.LatencyPercentiles(p50=150.0, p95=400.0),
        )
        d = aggregate.to_dict()
        assert d["recall@5"] == 0.5
        assert d["recall@20"] == 0.8
        assert d["groundedness"] == 0.75
        assert d["hallucination_rate"] == 0.25
        assert d["latency_p95_ms"] == 400.0


class TestThresholds:
    def test_passing_metrics_produce_no_failures(self) -> None:
        from app.services.eval.runner import check_thresholds

        metrics = {
            "recall@10": 0.9,
            "citation_accuracy": 0.95,
            "hallucination_rate": 0.02,
        }
        thresholds = {"recall@10": 0.8, "citation_accuracy": 0.9, "hallucination_rate": 0.05}
        assert check_thresholds(metrics, thresholds) == []

    def test_thresholds_yaml_loads(self) -> None:
        from pathlib import Path

        from app.services.eval.runner import _load_thresholds

        path = Path(__file__).resolve().parents[1] / "eval" / "thresholds.yaml"
        thresholds = _load_thresholds(path)
        assert "recall@10" in thresholds
        assert "citation_accuracy" in thresholds
        assert "hallucination_rate" in thresholds
        assert all(isinstance(value, float) for value in thresholds.values())

    def test_floor_violation_flagged(self) -> None:
        from app.services.eval.runner import check_thresholds

        failures = check_thresholds({"recall@10": 0.5}, {"recall@10": 0.8})
        assert any("recall@10" in failure for failure in failures)

    def test_ceiling_violation_flagged(self) -> None:
        from app.services.eval.runner import check_thresholds

        failures = check_thresholds({"hallucination_rate": 0.3}, {"hallucination_rate": 0.05})
        assert any("hallucination_rate" in failure for failure in failures)

    def test_missing_metric_flagged(self) -> None:
        from app.services.eval.runner import check_thresholds

        failures = check_thresholds({}, {"mrr": 0.6})
        assert failures
