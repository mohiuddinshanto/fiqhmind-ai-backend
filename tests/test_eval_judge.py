"""Tests for the Phase 17 LLM-as-judge (ARCHITECTURE §17.2/§17.3)."""

from app.services.eval.judge import LexicalAnswerJudge, _parse_score, get_answer_judge


class TestLexicalJudge:
    def test_correct_answer_scores_two(self) -> None:
        judge = LexicalAnswerJudge()
        score = judge.score(
            question="Is water pure?",
            explanation="Water is pure in itself and purifying for others.",
            expected_answer="Water is pure in itself and purifying for others.",
        )
        assert score == 2

    def test_partial_overlap_scores_one(self) -> None:
        judge = LexicalAnswerJudge()
        score = judge.score(
            question="q",
            explanation="Water purity rulings and purification for prayer.",
            expected_answer="Water is pure in itself and purifying for others.",
        )
        assert score == 1

    def test_no_overlap_scores_zero(self) -> None:
        judge = LexicalAnswerJudge()
        score = judge.score(
            question="q",
            explanation="The stock market moves unpredictably today.",
            expected_answer="Water is pure in itself and purifying for others.",
        )
        assert score == 0

    def test_empty_explanation_scores_zero(self) -> None:
        assert LexicalAnswerJudge().score(question="q", explanation="", expected_answer="x") == 0


class TestScoreParsing:
    def test_json_object_parsed(self) -> None:
        assert _parse_score('{"score": 2, "rationale": "ok"}') == 2

    def test_naked_integer_parsed(self) -> None:
        assert _parse_score("1") == 1

    def test_garbage_returns_zero(self) -> None:
        assert _parse_score("no score here at all") == 0


class TestJudgeSelection:
    def test_provider_absent_uses_lexical(self) -> None:
        assert isinstance(get_answer_judge(None), LexicalAnswerJudge)

    def test_provider_present_uses_llm(self) -> None:
        class FakeProvider:
            def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                return '{"score": 2, "rationale": "correct"}'

        from app.services.eval.judge import LLMAnswerJudge

        assert isinstance(get_answer_judge(FakeProvider()), LLMAnswerJudge)


class TestLLMJudge:
    def test_scores_provider_output(self) -> None:
        class FakeProvider:
            def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                return '{"score": 2, "rationale": "matches gold"}'

        judge = __import__("app.services.eval.judge", fromlist=["LLMAnswerJudge"]).LLMAnswerJudge(
            FakeProvider()
        )
        assert judge.score(question="q", explanation="e", expected_answer="g") == 2

    def test_provider_error_degrades_to_zero(self) -> None:
        class BrokenProvider:
            def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                raise RuntimeError("provider down")

        from app.services.eval.judge import LLMAnswerJudge

        judge = LLMAnswerJudge(BrokenProvider())
        assert judge.score(question="q", explanation="e", expected_answer="g") == 0
