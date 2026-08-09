from app.db.models import ChatHistory
from app.db.repositories.base import RepositoryBase


class ChatHistoryRepository(RepositoryBase[ChatHistory]):
    """Access to the chat_history table (Phase 10 persistence target).

    The table pre-exists from the Phase 10 schema (ARCHITECTURE Phase 13 lists
    qa/chat history); this repository adds the write path used by the chat API.
    """

    model = ChatHistory

    def add(
        self,
        *,
        question: str,
        normalized_query: str | None,
        answer_language: str,
        answer: dict | None,
        sources: list | None,
        confidence: str | None,
        refusal: str | None,
        user_id: str | None = None,
    ) -> ChatHistory:
        return self.create(
            ChatHistory(
                user_id=user_id,
                question=question,
                normalized_query=normalized_query,
                answer_language=answer_language,
                answer=answer,
                sources=sources,
                confidence=confidence,
                refusal=refusal,
            )
        )
