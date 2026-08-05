import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class UUIDPrimaryKeyMixin:
    """Primary key of 32-char lowercase hex UUID (ADR: string UUIDs)."""

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: uuid.uuid4().hex
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Role(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    users: Mapped[list["User"]] = relationship(back_populates="role")

    def __repr__(self) -> str:
        return f"<Role {self.name}>"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    role_id: Mapped[str | None] = mapped_column(ForeignKey("roles.id"), index=True)

    role: Mapped[Role | None] = relationship(back_populates="users")
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list["Bookmark"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    chat_history: Mapped[list["ChatHistory"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Session(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "sessions"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    refresh_token_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(255))

    user: Mapped[User] = relationship(back_populates="sessions")

    def __repr__(self) -> str:
        return f"<Session user={self.user_id}>"


class Book(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "books"
    __table_args__ = (
        UniqueConstraint("title", "author", name="uq_books_title_author"),
        CheckConstraint(
            "status IN ('draft', 'publishing', 'published')",
            name="status",
        ),
    )

    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title_transliteration: Mapped[str | None] = mapped_column(String(255))
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="ar", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    editions: Mapped[list["Edition"]] = relationship(
        back_populates="book", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="book")
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="book")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Book):
            return (self.title, self.author) == (other.title, other.author)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.title, self.author))

    def __repr__(self) -> str:
        return f"<Book {self.title} — {self.author}>"


class Edition(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "editions"
    __table_args__ = (
        UniqueConstraint("book_id", "edition_number", name="uq_editions_book_number"),
    )

    book_id: Mapped[str] = mapped_column(ForeignKey("books.id"), nullable=False, index=True)
    publisher: Mapped[str | None] = mapped_column(String(255))
    publication_year: Mapped[int | None] = mapped_column(Integer)
    muhaqqiq: Mapped[str | None] = mapped_column(String(255))
    edition_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    isbn: Mapped[str | None] = mapped_column(String(32), index=True)
    language: Mapped[str] = mapped_column(String(10), default="ar", nullable=False)

    book: Mapped[Book] = relationship(back_populates="editions")
    volumes: Mapped[list["Volume"]] = relationship(
        back_populates="edition", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="edition")

    def __repr__(self) -> str:
        return f"<Edition {self.publisher} #{self.edition_number}>"


class Volume(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "volumes"
    __table_args__ = (
        UniqueConstraint("edition_id", "volume_number", name="uq_volumes_edition_number"),
    )

    edition_id: Mapped[str] = mapped_column(ForeignKey("editions.id"), nullable=False, index=True)
    volume_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    printed_page_start: Mapped[int | None] = mapped_column(Integer)
    printed_page_end: Mapped[int | None] = mapped_column(Integer)

    edition: Mapped[Edition] = relationship(back_populates="volumes")
    pages: Mapped[list["Page"]] = relationship(
        back_populates="volume", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="volume")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Volume):
            return (self.edition_id, self.volume_number) == (other.edition_id, other.volume_number)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.edition_id, self.volume_number))

    def __repr__(self) -> str:
        return f"<Volume {self.volume_number}>"


class Page(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("volume_id", "printed_page", name="uq_pages_volume_page"),
        CheckConstraint(
            "status IN ('pending', 'extracted', 'review', 'verified')",
            name="status",
        ),
    )

    volume_id: Mapped[str] = mapped_column(ForeignKey("volumes.id"), nullable=False, index=True)
    printed_page: Mapped[int] = mapped_column(Integer, nullable=False)
    pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    arabic_char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    page_number_uncertain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="extracted", nullable=False)

    volume: Mapped[Volume] = relationship(back_populates="pages")
    chunk_starts: Mapped[list["Chunk"]] = relationship(
        back_populates="page_start", foreign_keys="Chunk.page_start_id"
    )
    chunk_ends: Mapped[list["Chunk"]] = relationship(
        back_populates="page_end", foreign_keys="Chunk.page_end_id"
    )

    def __repr__(self) -> str:
        return f"<Page printed={self.printed_page} pdf={self.pdf_page}>"


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(
            "region IN ('main', 'footnote', 'margin', 'header', 'footer')",
            name="region",
        ),
        Index("ix_chunks_book_region_verified", "book_id", "region", "verified"),
    )

    # chunk_id = sha256(raw_text): content-addressed, immutable (Phase 6).
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    book_id: Mapped[str] = mapped_column(ForeignKey("books.id"), nullable=False, index=True)
    edition_id: Mapped[str] = mapped_column(ForeignKey("editions.id"), nullable=False, index=True)
    volume_id: Mapped[str] = mapped_column(ForeignKey("volumes.id"), nullable=False, index=True)
    page_start_id: Mapped[str | None] = mapped_column(ForeignKey("pages.id"), index=True)
    page_end_id: Mapped[str | None] = mapped_column(ForeignKey("pages.id"), index=True)

    printed_page_start: Mapped[int | None] = mapped_column(Integer)
    printed_page_end: Mapped[int | None] = mapped_column(Integer)
    pdf_page_start: Mapped[int | None] = mapped_column(Integer)
    pdf_page_end: Mapped[int | None] = mapped_column(Integer)

    kitab: Mapped[str | None] = mapped_column(String(255))
    bab: Mapped[str | None] = mapped_column(String(255))
    fasl: Mapped[str | None] = mapped_column(String(255))
    topic: Mapped[str | None] = mapped_column(String(255))
    region: Mapped[str] = mapped_column(String(20), default="main", nullable=False)
    lang: Mapped[str] = mapped_column(String(10), default="ar", nullable=False)

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    book: Mapped[Book] = relationship(back_populates="chunks")
    edition: Mapped[Edition] = relationship(back_populates="chunks")
    volume: Mapped[Volume] = relationship(back_populates="chunks")
    page_start: Mapped[Page | None] = relationship(
        back_populates="chunk_starts", foreign_keys=[page_start_id]
    )
    page_end: Mapped[Page | None] = relationship(
        back_populates="chunk_ends", foreign_keys=[page_end_id]
    )
    bookmarks: Mapped[list["Bookmark"]] = relationship(back_populates="chunk")

    def __repr__(self) -> str:
        return f"<Chunk {self.chunk_id[:8]}… region={self.region}>"


class IngestionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('initial', 'reindex', 'ocr')",
            name="kind",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'sanitizing', 'extracting', 'ocr', 'ocr_correcting', "
            "'structuring', 'chunking', 'embedding', 'indexed', 'failed')",
            name="status",
        ),
        CheckConstraint("progress_percent BETWEEN 0 AND 100", name="progress_range"),
    )

    book_id: Mapped[str] = mapped_column(ForeignKey("books.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(20), default="initial", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="uploaded", nullable=False, index=True)
    current_step: Mapped[str | None] = mapped_column(String(30))
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    book: Mapped[Book] = relationship(back_populates="ingestion_jobs")
    errors: Mapped[list["IngestionError"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<IngestionJob {self.kind} status={self.status}>"


class IngestionError(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "ingestion_errors"

    job_id: Mapped[str] = mapped_column(ForeignKey("ingestion_jobs.id"), nullable=False, index=True)
    step: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[IngestionJob] = relationship(back_populates="errors")

    def __repr__(self) -> str:
        return f"<IngestionError step={self.step}>"


class Bookmark(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "bookmarks"
    __table_args__ = (
        UniqueConstraint("user_id", "chunk_id", name="uq_bookmarks_user_chunk"),
    )

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.chunk_id"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    topic: Mapped[str | None] = mapped_column(String(255))
    note: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="bookmarks")
    chunk: Mapped[Chunk] = relationship(back_populates="bookmarks")

    def __repr__(self) -> str:
        return f"<Bookmark chunk={self.chunk_id[:8]}…>"


class ChatHistory(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "chat_history"

    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str | None] = mapped_column(Text)
    answer_language: Mapped[str] = mapped_column(String(10), default="bn", nullable=False)
    answer: Mapped[dict | None] = mapped_column(JSON)
    sources: Mapped[list | None] = mapped_column(JSON)
    confidence: Mapped[str | None] = mapped_column(String(10))
    refusal: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[User | None] = relationship(back_populates="chat_history")

    def __repr__(self) -> str:
        return f"<ChatHistory user={self.user_id}>"


class AuditLog(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "audit_log"

    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    actor_type: Mapped[str] = mapped_column(
        String(30), default="system", nullable=False
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(50), index=True)
    resource_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict | None] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[User | None] = relationship()

    def __repr__(self) -> str:
        return f"<AuditLog {self.action}>"
