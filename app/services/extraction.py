"""PDF extraction engine (Phase 4).

PyMuPDF (`fitz`) reads the native PDF text layer — pages, blocks, spans,
fonts, sizes, bounding boxes, images and vector drawings — and preserves all
coordinates. No OCR, no layout classification, no metadata extraction and no
chunking: later phases consume these structured rows.

`extract_pdf` is a pure function (no DB). `ExtractionRunner` streams the
per-page output into PostgreSQL page by page so progress is durable and a
failed book can resume.
"""

from dataclasses import dataclass, field
from datetime import datetime

import fitz
import structlog
from sqlalchemy.orm import Session

from app.core.exceptions import (
    EncryptedPdfError,
    MalformedPdfError,
    PdfPageLimitError,
    TransientExtractionError,
)
from app.core.storage import StorageProvider
from app.db.models import IngestionJob, Upload
from app.db.repositories import ExtractionRepository

logger = structlog.get_logger(__name__)

# Pages below this character count are treated as not having a usable text layer
# (stamps, page numbers, watermarks alone do not make a "text" page).
MIN_TEXT_CHARS_PER_PAGE = 20
# Rough capacity of a dense Arabic typeset page, used to normalize page confidence.
TEXT_CAPACITY_PER_PAGE = 4000
# Minimum text pages required for the document to claim a text layer.
MIN_TEXT_PAGES_FOR_LAYER = 1


@dataclass
class SpanInfo:
    index: int
    text: str
    font: str
    size: float
    bbox: list[float]
    flags: int | None = None


@dataclass
class BlockInfo:
    index: int
    bbox: list[float]
    text: str
    font: str | None
    size: float | None
    spans: list[SpanInfo] = field(default_factory=list)


@dataclass
class ImageInfo:
    index: int
    bbox: list[float]
    width: int | None
    height: int | None
    xref: int | None


@dataclass
class DrawingInfo:
    index: int
    bbox: list[float]
    kind: str | None
    stroke_width: float | None


@dataclass
class PageInfo:
    number: int  # 1-based
    width: float
    height: float
    rotation: int
    blocks: list[BlockInfo] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    drawings: list[DrawingInfo] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    @property
    def has_text(self) -> bool:
        return self.char_count >= MIN_TEXT_CHARS_PER_PAGE

    @property
    def confidence(self) -> float:
        if not self.has_text:
            return 0.0
        return min(1.0, self.char_count / TEXT_CAPACITY_PER_PAGE)


@dataclass
class PdfExtraction:
    page_count: int
    pages: list[PageInfo] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def has_text_layer(self) -> bool:
        return sum(page.has_text for page in self.pages) >= MIN_TEXT_PAGES_FOR_LAYER

    @property
    def confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(page.confidence for page in self.pages) / len(self.pages)


def _guard_page_count(page_count: int, max_pages: int | None) -> None:
    """Reject decompression-bomb PDFs that exceed the configured page cap."""
    if max_pages is not None and page_count > max_pages:
        raise PdfPageLimitError(
            f"pdf has {page_count} pages, exceeding the {max_pages}-page limit"
        )


def _open_pdf(path: str) -> fitz.Document:
    """Open a PDF with PyMuPDF, mapping permanent failures to typed errors."""
    try:
        document = fitz.open(path)
    except fitz.FileDataError as exc:
        raise MalformedPdfError(f"invalid or corrupted pdf: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive for exotic formats
        raise MalformedPdfError(f"unable to open pdf: {exc}") from exc
    if document.needs_pass:
        document.close()
        raise EncryptedPdfError("pdf is password-protected; encrypted PDFs are not supported")
    return document


def extract_pdf(path: str, *, max_pages: int | None = None) -> PdfExtraction:
    """Extract every page of the PDF at `path` into structured (coordinate-preserving) data."""
    document = _open_pdf(path)
    try:
        _guard_page_count(document.page_count, max_pages)
        extraction = PdfExtraction(page_count=document.page_count)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            rect = page.rect
            page_info = PageInfo(
                number=page_index + 1,
                width=rect.width,
                height=rect.height,
                rotation=int(page.rotation),
            )
            page_info.blocks = _extract_blocks(page)
            page_info.images = _extract_images(page)
            page_info.drawings = _extract_drawings(page)
            extraction.pages.append(page_info)
        return extraction
    finally:
        document.close()


def _extract_blocks(page: fitz.Page) -> list[BlockInfo]:
    blocks: list[BlockInfo] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue  # image blocks are captured separately via get_image_info
        lines = block.get("lines", [])
        spans: list[SpanInfo] = []
        line_texts: list[str] = []
        for line in lines:
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                spans.append(
                    SpanInfo(
                        index=len(spans),
                        text=text,
                        font=span.get("font", ""),
                        size=float(span.get("size", 0.0)),
                        bbox=[float(value) for value in span.get("bbox", [0, 0, 0, 0])],
                        flags=int(span.get("flags", 0)),
                    )
                )
                line_texts.append(text)
        if not spans:
            continue
        dominant = max(spans, key=lambda span: span.size)
        blocks.append(
            BlockInfo(
                index=len(blocks),
                bbox=[float(value) for value in block.get("bbox", [0, 0, 0, 0])],
                text="".join(line_texts),
                font=dominant.font or None,
                size=dominant.size,
                spans=spans,
            )
        )
    return blocks


def _extract_images(page: fitz.Page) -> list[ImageInfo]:
    images: list[ImageInfo] = []
    for index, image in enumerate(page.get_image_info(xrefs=True)):
        bbox = image.get("bbox", [0, 0, 0, 0])
        images.append(
            ImageInfo(
                index=index,
                bbox=[float(value) for value in bbox],
                width=int(image["width"]) if image.get("width") else None,
                height=int(image["height"]) if image.get("height") else None,
                xref=int(image["xref"]) if image.get("xref") else None,
            )
        )
    return images


def _extract_drawings(page: fitz.Page) -> list[DrawingInfo]:
    drawings: list[DrawingInfo] = []
    for index, drawing in enumerate(page.get_drawings()):
        rect = drawing.get("rect")
        drawings.append(
            DrawingInfo(
                index=index,
                bbox=[float(value) for value in rect] if rect else [0, 0, 0, 0],
                kind=drawing.get("type"),
                stroke_width=float(drawing["width"]) if drawing.get("width") is not None else None,
            )
        )
    return drawings


class ExtractionRunner:
    """Streams a PDF's extracted pages into PostgreSQL, updating job progress."""

    def __init__(
        self, session: Session, storage: StorageProvider, *, max_pages: int | None = None
    ) -> None:
        self._session = session
        self._storage = storage
        self._max_pages = max_pages
        self._repo = ExtractionRepository(session)

    def run(self, job: IngestionJob, upload: Upload) -> PdfExtraction:
        """Extract `upload`'s PDF into `job`'s page rows. Returns the extraction summary."""
        self._begin(job, upload)
        path = self._storage.resolve(upload.storage_path or "")
        document = _open_pdf(path)
        _guard_page_count(document.page_count, self._max_pages)

        upload.page_count = document.page_count
        self._session.commit()

        try:
            extraction = PdfExtraction(page_count=document.page_count)
            for page_index in range(document.page_count):
                page_info = self._extract_one_page(document, page_index, job, upload)
                extraction.pages.append(page_info)
                progress = round(((page_index + 1) / document.page_count) * 100)
                job.progress_percent = progress
                self._session.commit()
            return extraction
        except TransientExtractionError:
            self._session.rollback()
            raise
        finally:
            document.close()

    def _begin(self, job: IngestionJob, upload: Upload) -> None:
        if not upload.storage_path or not self._storage.exists(upload.storage_path):
            raise MalformedPdfError("stored file is missing for this upload")
        job.status = "extracting"
        job.current_step = "extracting"
        job.progress_percent = 0
        job.started_at = datetime.utcnow()
        upload.status = "processing"
        self._session.commit()

    def _extract_one_page(
        self,
        document: fitz.Document,
        page_index: int,
        job: IngestionJob,
        upload: Upload,
    ) -> PageInfo:
        page_number = page_index + 1
        page = document.load_page(page_index)
        try:
            page_info = _parse_page(page, page_number)
        except Exception as exc:
            logger.warning(
                "page_extraction_failed", job_id=job.id, page_number=page_number, error=str(exc)
            )
            page_info = self._record_failed_page(job, page_number, page, str(exc))
            return page_info

        page_row = self._repo.create_page(
            job_id=job.id,
            page_number=page_number,
            width=page_info.width,
            height=page_info.height,
            rotation=page_info.rotation,
            has_text=page_info.has_text,
            char_count=page_info.char_count,
            block_count=len(page_info.blocks),
            image_count=len(page_info.images),
            drawing_count=len(page_info.drawings),
            confidence=page_info.confidence,
        )
        for block in page_info.blocks:
            block_row = self._repo.add_block(
                page_row,
                block_index=block.index,
                bbox=block.bbox,
                text=block.text,
                font=block.font,
                font_size=block.size,
            )
            block_row.span_count = len(block.spans)
            for span in block.spans:
                self._repo.add_span(
                    block_row,
                    span_index=span.index,
                    text=span.text,
                    font=span.font,
                    font_size=span.size,
                    bbox=span.bbox,
                    flags=span.flags,
                )
        for image in page_info.images:
            self._repo.add_image(
                page_row,
                image_index=image.index,
                bbox=image.bbox,
                width=image.width,
                height=image.height,
                xref=image.xref,
            )
        for drawing in page_info.drawings:
            self._repo.add_drawing(
                page_row,
                drawing_index=drawing.index,
                bbox=drawing.bbox,
                kind=drawing.kind,
                stroke_width=drawing.stroke_width,
            )
        return page_info

    def _record_failed_page(
        self, job: IngestionJob, page_number: int, page: fitz.Page, error: str
    ) -> PageInfo:
        rect = page.rect
        self._repo.create_page(
            job_id=job.id,
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            rotation=int(page.rotation),
            has_text=False,
            char_count=0,
            block_count=0,
            image_count=0,
            drawing_count=0,
            confidence=0.0,
            error_message=error,
        )
        return PageInfo(
            number=page_number,
            width=rect.width,
            height=rect.height,
            rotation=int(page.rotation),
        )


def _parse_page(page: fitz.Page, page_number: int) -> PageInfo:
    rect = page.rect
    return PageInfo(
        number=page_number,
        width=rect.width,
        height=rect.height,
        rotation=int(page.rotation),
        blocks=_extract_blocks(page),
        images=_extract_images(page),
        drawings=_extract_drawings(page),
    )
