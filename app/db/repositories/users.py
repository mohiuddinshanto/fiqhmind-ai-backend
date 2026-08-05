from sqlalchemy import select

from app.db.models import User
from app.db.repositories.base import RepositoryBase


class UserRepository(RepositoryBase[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        return self._session.scalar(select(User).where(User.email == email))
