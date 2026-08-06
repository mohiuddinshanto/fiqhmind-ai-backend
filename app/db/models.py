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
            "region IN ('main', 'footnote', 'margin', 'header', 'footer', 'unknown')",
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


class MetadataDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Run-level container for one upload's extracted metadata (Phase 6).

    Bibliographic + structural fields live in `MetadataField`, per-page
    numbering in `PageMetadata`, hierarchical sections in `MetadataStructure`.
    Nothing here writes into books/volumes/pages — chunking is a later phase.
    """

    __tablename__ = "metadata_documents"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_metadata_documents_job"),
    )

    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id"), nullable=False, index=True
    )
    upload_id: Mapped[str] = mapped_column(ForeignKey("uploads.id"), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    numbering_system: Mapped[str] = mapped_column(
        String(20), default="none", nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    fields: Mapped[list["MetadataField"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    pages: Mapped[list["PageMetadata"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    structures: Mapped[list["MetadataStructure"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<MetadataDocument {self.original_filename} numbering={self.numbering_system}>"


class MetadataField(UUIDPrimaryKeyMixin, Base):
    """One extracted metadata field with its confidence and provenance.

    Every field carries (value, confidence, extraction source) as required by
    Phase 6 — e.g. title from the filename vs. from the cover page.
    """

    __tablename__ = "metadata_fields"
    __table_args__ = (
        UniqueConstraint("document_id", "field", name="uq_metadata_fields_document_field"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_metadata_fields_confidence"),
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("metadata_documents.id"), nullable=False, index=True
    )
    field: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[MetadataDocument] = relationship(back_populates="fields")

    def __repr__(self) -> str:
        return f"<MetadataField {self.field}={self.value!r}>"


class PageMetadata(UUIDPrimaryKeyMixin, Base):
    """PDF page → printed page mapping. Both values are always stored.

    `printed_page` keeps the raw label (Arabic-Indic digits, Roman numerals,
    Latin digits, or a mixed label); `printed_page_numeric` is the normalized
    value used for continuity checks. Current structural context (kitab/bab/fasl)
    is copied here per page so later chunking can attach chapter context.
    """

    __tablename__ = "page_metadata"
    __table_args__ = (
        UniqueConstraint("document_id", "pdf_page", name="uq_page_metadata_document_page"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_page_metadata_confidence"),
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("metadata_documents.id"), nullable=False, index=True
    )
    pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based
    printed_page: Mapped[str] = mapped_column(String(255), nullable=False)
    printed_page_numeric: Mapped[int | None] = mapped_column(Integer)
    numbering_system: Mapped[str] = mapped_column(String(20), default="none", nullable=False)
    page_number_uncertain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    kitab: Mapped[str | None] = mapped_column(String(255))
    bab: Mapped[str | None] = mapped_column(String(255))
    fasl: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[MetadataDocument] = relationship(back_populates="pages")

    def __repr__(self) -> str:
        return f"<PageMetadata pdf={self.pdf_page} printed={self.printed_page!r}>"


class MetadataStructure(UUIDPrimaryKeyMixin, Base):
    """A detected hierarchical section (kitab/bab/fasl/topic) with its page range."""

    __tablename__ = "metadata_structures"
    __table_args__ = (
        CheckConstraint(
            "level IN ('kitab', 'bab', 'fasl', 'topic')", name="ck_metadata_structures_level"
        ),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_metadata_structures_confidence"),
        Index("ix_metadata_structures_doc_level_start", "document_id", "level", "page_start"),
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("metadata_documents.id"), nullable=False, index=True
    )
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[MetadataDocument] = relationship(back_populates="structures")

    def __repr__(self) -> str:
        return f"<MetadataStructure {self.level}: {self.name} p.{self.page_start}>"


class IngestionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('initial', 'reindex', 'ocr', 'extraction', 'layout', 'metadata')",
            name="kind",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'queued', 'processing', 'sanitizing', 'extracting', "
            "'ocr', 'ocr_correcting', 'structuring', 'chunking', 'embedding', 'indexed', "
            "'completed', 'failed')",
            name="status",
        ),
        CheckConstraint("progress_percent BETWEEN 0 AND 100", name="progress_range"),
    )

    # book_id is nullable: an upload job exists before book metadata is known.
    book_id: Mapped[str | None] = mapped_column(ForeignKey("books.id"), index=True)
    upload_id: Mapped[str | None] = mapped_column(ForeignKey("uploads.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="initial", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="uploaded", nullable=False, index=True)
    current_step: Mapped[str | None] = mapped_column(String(30))
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    book: Mapped[Book | None] = relationship(back_populates="ingestion_jobs")
    upload: Mapped["Upload | None"] = relationship(back_populates="ingestion_job")
    errors: Mapped[list["IngestionError"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    page_extractions: Mapped[list["PageExtraction"]] = relationship(
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


class Upload(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "uploads"
    __table_args__ = (
        CheckConstraint(
            "status IN ('uploading', 'uploaded', 'queued', 'processing', 'completed', 'failed')",
            name="status",
        ),
        Index("ix_uploads_sha256", "sha256", unique=True),
    )

    # original_filename: the client-provided name (sanitized, display-safe).
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # filename: the object name inside the storage backend (generated, unique).
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    size: Mapped[int | None] = mapped_column(Integer)
    mime: Mapped[str | None] = mapped_column(String(100))
    page_count: Mapped[int | None] = mapped_column(Integer)
    storage_path: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="uploading", nullable=False, index=True)
    received_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    ingestion_job: Mapped["IngestionJob | None"] = relationship(
        back_populates="upload", uselist=False
    )
    logs: Mapped[list["UploadLog"]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )

    @property
    def uploaded_at(self) -> datetime:
        """The upload record creation timestamp (API field alias)."""
        return self.created_at

    def __repr__(self) -> str:
        return f"<Upload {self.original_filename} status={self.status}>"


class UploadLog(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "upload_logs"

    upload_id: Mapped[str] = mapped_column(ForeignKey("uploads.id"), nullable=False, index=True)
    event: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    upload: Mapped[Upload] = relationship(back_populates="logs")

    def __repr__(self) -> str:
        return f"<UploadLog {self.event}>"


class PageExtraction(UUIDPrimaryKeyMixin, Base):
    """One extracted PDF page (Phase 4 — coordinates preserved, no layout labels)."""

    __tablename__ = "page_extractions"
    __table_args__ = (
        UniqueConstraint("job_id", "page_number", name="uq_page_extractions_job_page"),
    )

    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    rotation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    has_text: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    block_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    drawing_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped[IngestionJob] = relationship(back_populates="page_extractions")
    blocks: Mapped[list["PageBlock"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )
    images: Mapped[list["PageImage"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )
    drawings: Mapped[list["PageDrawing"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PageExtraction job={self.job_id[:8]}… page={self.page_number}>"


class PageBlock(UUIDPrimaryKeyMixin, Base):
    """A text block with its spans. Region/order are set by the Phase 5 layout engine."""

    __tablename__ = "page_blocks"
    __table_args__ = (
        UniqueConstraint("page_extraction_id", "block_index", name="uq_page_blocks_page_index"),
        CheckConstraint(
            "region IN ('main', 'footnote', 'margin', 'header', 'footer', 'unknown')",
            name="ck_page_blocks_region",
        ),
    )

    page_extraction_id: Mapped[str] = mapped_column(
        ForeignKey("page_extractions.id"), nullable=False, index=True
    )
    block_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)  # [x0, y0, x1, y1]
    text: Mapped[str] = mapped_column(Text, nullable=False)
    font: Mapped[str | None] = mapped_column(String(255))
    font_size: Mapped[float | None] = mapped_column(Float)
    span_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    region: Mapped[str] = mapped_column(String(20), default="unknown", nullable=False)
    reading_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    classification_reason: Mapped[str | None] = mapped_column(String(500))

    page: Mapped[PageExtraction] = relationship(back_populates="blocks")
    spans: Mapped[list["PageSpan"]] = relationship(
        back_populates="block", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PageBlock {self.block_index} region={self.region}>"


class PageSpan(UUIDPrimaryKeyMixin, Base):
    """A single span (same font+size run) inside a block, with its own bbox."""

    __tablename__ = "page_spans"
    __table_args__ = (
        UniqueConstraint("block_id", "span_index", name="uq_page_spans_block_index"),
    )

    block_id: Mapped[str] = mapped_column(ForeignKey("page_blocks.id"), nullable=False, index=True)
    span_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    font: Mapped[str] = mapped_column(String(255), nullable=False)
    font_size: Mapped[float] = mapped_column(Float, nullable=False)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)
    flags: Mapped[int | None] = mapped_column(Integer)

    block: Mapped[PageBlock] = relationship(back_populates="spans")

    def __repr__(self) -> str:
        return f"<PageSpan {self.font} {self.font_size}>"


class PageImage(UUIDPrimaryKeyMixin, Base):
    """An image object on a page (pixel dims from the embedded image)."""

    __tablename__ = "page_images"
    __table_args__ = (
        UniqueConstraint("page_extraction_id", "image_index", name="uq_page_images_page_index"),
    )

    page_extraction_id: Mapped[str] = mapped_column(
        ForeignKey("page_extractions.id"), nullable=False, index=True
    )
    image_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)  # [x0, y0, x1, y1]
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    xref: Mapped[int | None] = mapped_column(Integer)

    page: Mapped[PageExtraction] = relationship(back_populates="images")

    def __repr__(self) -> str:
        return f"<PageImage {self.image_index}>"


class PageDrawing(UUIDPrimaryKeyMixin, Base):
    """A vector drawing object on a page (rect/line/curve/quad)."""

    __tablename__ = "page_drawings"
    __table_args__ = (
        UniqueConstraint("page_extraction_id", "drawing_index", name="uq_page_drawings_page_index"),
    )

    page_extraction_id: Mapped[str] = mapped_column(
        ForeignKey("page_extractions.id"), nullable=False, index=True
    )
    drawing_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)  # [x0, y0, x1, y1]
    kind: Mapped[str | None] = mapped_column(String(10))  # 'f' | 's' | 'fs'
    stroke_width: Mapped[float | None] = mapped_column(Float)

    page: Mapped[PageExtraction] = relationship(back_populates="drawings")

    def __repr__(self) -> str:
        return f"<PageDrawing {self.kind}>"


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
