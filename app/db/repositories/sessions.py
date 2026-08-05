from datetime import datetime

from sqlalchemy import select

from app.db.models import Session
from app.db.repositories.base import RepositoryBase


class SessionRepository(RepositoryBase[Session]):
    model = Session

    def get_by_refresh_token_hash(self, refresh_token_hash: str) -> Session | None:
        return self._session.scalar(
            select(Session).where(Session.refresh_token_hash == refresh_token_hash)
        )

    def revoke(self, instance: Session, *, at: datetime | None = None) -> Session:
        instance.revoked_at = at or datetime.utcnow()
        return self.update(instance)

    def list_active_for_user(self, user_id: str) -> list[Session]:
        return list(
            self._session.scalars(
                select(Session)
                .where(Session.user_id == user_id, Session.revoked_at.is_(None))
                .order_by(Session.created_at.desc())
            )
        )
