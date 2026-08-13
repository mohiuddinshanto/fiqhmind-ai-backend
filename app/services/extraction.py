"""PDF extraction engine (Phase 4).

PyMuPDF (`fitz`) reads the native PDF text layer — pages, blocks, spans,
fonts, sizes, bounding boxes, images and vector drawings — and preserves all
coordinates. No OCR, no layout classification, no metadata extraction and no
chunking: later phases consume these structured rows.

`extract_pdf` is a pure function (no DB). `ExtractionRunner` parses pages in
parallel (Phase 15 §713) and streams the per-page output into PostgreSQL so
progress is durable and a failed book can resume.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import fitz
import structlog
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    EncryptedPdfError,
    MalformedPdfError,
    PdfPageLimitError,
    TransientExtractionError,
)
from app.core.storage import StorageProvider
from app.db.models import IngestionJob, PageExtraction, Upload
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
    # When set, carries the persisted aggregate instead of deriving it from
    # `blocks` (used when resuming a book from its Postgres page rows).
    stored_char_count: int | None = field(default=None, repr=False)

    @property
    def char_count(self) -> int:
        if self.stored_char_count is not None:
            return self.stored_char_count
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


@dataclass
class _PageParseResult:
    """One page's parse outcome: a full `PageInfo` or a failure record."""

    number: int
    info: PageInfo | None = None
    error: str | None = None
    width: float = 0.0
    height: float = 0.0
    rotation: int = 0


def _partition_pages(page_count: int, workers: int) -> list[list[int]]:
    """Contiguous, roughly equal slices of page indices — one per worker."""
    workers = max(1, int(workers))
    size = max(1, (page_count + workers - 1) // workers)
    return [list(range(i, min(i + size, page_count))) for i in range(0, page_count, size)]


def _extract_page_slice(
    path: str,
    page_indices: list[int],
    parse_fn: Callable[[fitz.Page, int], PageInfo],
) -> list[_PageParseResult]:
    """Parse a contiguous slice of pages in a worker thread (its own document).

    PyMuPDF documents are not thread-safe (Phase 15 §713 page-level parallel
    extraction), so each worker opens its own document. Per-page parse failures
    are captured as `_PageParseResult` error records so one broken page never
    aborts the book; the document itself uses the same `_open_pdf` guards.
    """
    document = _open_pdf(path)
    try:
        results: list[_PageParseResult] = []
        for page_index in page_indices:
            page = document.load_page(page_index)
            rect = page.rect
            try:
                info = parse_fn(page, page_index + 1)
                results.append(_PageParseResult(number=page_index + 1, info=info))
            except Exception as exc:  # noqa: BLE001 - per-page failure is recorded
                results.append(
                    _PageParseResult(
                        number=page_index + 1,
                        error=str(exc),
                        width=rect.width,
                        height=rect.height,
                        rotation=int(page.rotation),
                    )
                )
        return results
    finally:
        document.close()


class ExtractionRunner:
    """Streams a PDF's extracted pages into PostgreSQL, updating job progress.

    Pages are parsed in parallel (a thread pool whose workers each open their
    own PyMuPDF document) while rows are persisted by the calling thread in
    page order, so `page_number` ordering and progress stay deterministic.

    Checkpointing (Phase 15 §Batch processing): page rows are committed every
    `checkpoint_pages` pages, so a crash loses at most one checkpoint window.
    `run()` is resumable — already-persisted page numbers for the job are
    skipped, letting a retried task continue mid-book instead of restarting.
    """

    def __init__(
        self,
        session: Session,
        storage: StorageProvider,
        *,
        max_pages: int | None = None,
        max_workers: int | None = None,
        checkpoint_pages: int | None = None,
        parse_fn: Callable[[fitz.Page, int], PageInfo] | None = None,
    ) -> None:
        self._session = session
        self._storage = storage
        self._max_pages = max_pages
        self._max_workers = max(1, int(max_workers or get_settings().extraction_workers))
        self._checkpoint_pages = max(
            1, int(checkpoint_pages or get_settings().extraction_checkpoint_pages)
        )
        self._parse_fn = parse_fn
        self._repo = ExtractionRepository(session)

    def run(self, job: IngestionJob, upload: Upload) -> PdfExtraction:
        """Extract `upload`'s PDF into `job`'s page rows. Returns the extraction summary."""
        self._begin(job, upload)
        path = self._storage.resolve(upload.storage_path or "")
        document = _open_pdf(path)
        _guard_page_count(document.page_count, self._max_pages)
        total = document.page_count
        document.close()

        upload.page_count = total

        try:
            extraction = PdfExtraction(page_count=total)
            parse_fn = self._parse_fn or _parse_page
            persisted_rows = self._repo.list_pages(job.id)
            persisted = {row.page_number for row in persisted_rows}
            missing = [number - 1 for number in range(1, total + 1) if number not in persisted]
            done = len(persisted)
            job.progress_percent = round(done / total * 100) if total else 100
            self._session.commit()

            slices = _partition_pages(len(missing), self._max_workers)
            slices = [[missing[index] for index in slice_] for slice_ in slices]
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = [
                    pool.submit(_extract_page_slice, path, indices, parse_fn)
                    for indices in slices
                ]
                since_checkpoint = 0
                for future in futures:
                    for result in future.result():
                        extraction.pages.append(self._persist_result(job, result))
                        done += 1
                        since_checkpoint += 1
                        job.progress_percent = round(done / total * 100) if total else 100
                        if since_checkpoint >= self._checkpoint_pages:
                            self._session.commit()
                            since_checkpoint = 0
                self._session.commit()

            if persisted_rows:
                # Resume: merge the already-persisted pages (reconstructed from
                # their rows) with the newly parsed ones, in page order.
                by_number = {page.number: page for page in extraction.pages}
                for row in persisted_rows:
                    by_number.setdefault(row.page_number, _row_to_page_info(row))
                extraction.pages = [by_number[number] for number in sorted(by_number)]
            return extraction
        except TransientExtractionError:
            self._session.rollback()
            raise

    def _begin(self, job: IngestionJob, upload: Upload) -> None:
        if not upload.storage_path or not self._storage.exists(upload.storage_path):
            raise MalformedPdfError("stored file is missing for this upload")
        job.status = "extracting"
        job.current_step = "extracting"
        job.progress_percent = 0
        job.started_at = datetime.utcnow()
        upload.status = "processing"
        self._session.commit()

    def _persist_result(self, job: IngestionJob, result: _PageParseResult) -> PageInfo:
        if result.info is not None:
            return self._persist_page(job, result.number, result.info)
        return self._persist_failed_page(job, result)

    def _persist_page(self, job: IngestionJob, page_number: int, page_info: PageInfo) -> PageInfo:
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

    def _persist_failed_page(self, job: IngestionJob, result: _PageParseResult) -> PageInfo:
        self._repo.create_page(
            job_id=job.id,
            page_number=result.number,
            width=result.width,
            height=result.height,
            rotation=result.rotation,
            has_text=False,
            char_count=0,
            block_count=0,
            image_count=0,
            drawing_count=0,
            confidence=0.0,
            error_message=result.error,
        )
        return PageInfo(
            number=result.number,
            width=result.width,
            height=result.height,
            rotation=result.rotation,
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


def _row_to_page_info(row: PageExtraction) -> PageInfo:
    """Reconstruct a `PageInfo` from a persisted `PageExtraction` row (resume).

    Only the summary aggregates are preserved in the row (char/text counts and
    geometry); the block/spans/images/drawings detail lives in child tables and
    is re-readable from the DB when needed, so the returned `PageInfo` carries
    the stored char count instead of a block list.
    """
    return PageInfo(
        number=row.page_number,
        width=row.width,
        height=row.height,
        rotation=row.rotation,
        stored_char_count=row.char_count,
    )
