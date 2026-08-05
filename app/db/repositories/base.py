from collections.abc import Sequence
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class RepositoryBase(Generic[ModelT]):
    """Minimal generic data-access base. Only this layer touches SQLAlchemy models."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, pk: str) -> ModelT | None:
        return self._session.get(self.model, pk)

    def get_multi(self, *, skip: int = 0, limit: int = 100) -> Sequence[ModelT]:
        return self._session.scalars(select(self.model).offset(skip).limit(limit)).all()

    def create(self, instance: ModelT) -> ModelT:
        self._session.add(instance)
        self._session.commit()
        self._session.refresh(instance)
        return instance

    def update(self, instance: ModelT) -> ModelT:
        self._session.commit()
        self._session.refresh(instance)
        return instance

    def delete(self, instance: ModelT) -> None:
        self._session.delete(instance)
        self._session.commit()
