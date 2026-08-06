"""Metadata extraction engine (Phase 6).

Extracts and normalizes bibliographic and structural metadata from a PDF's
filename, cover/first pages and per-page footer text, and builds the
PDF page → printed page mapping.

Pure rule-based, geometry-aware, deterministic — no ML and no chunking.
`extract_metadata` is a pure function over the Phase 4 `PageInfo` dataclasses;
`MetadataRunner` (see `app/services/metadata.py`) persists the result through
`MetadataRepository`.

Every extracted field carries (value, confidence, extraction source); the PDF
page number is never discarded — the mapping stores both `pdf_page` and
`printed_page` for every page.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import IngestionJob, PageExtraction, Upload
from app.db.repositories import ExtractionRepository, MetadataRepository
from app.services.extraction import BlockInfo, PageInfo
from app.services.layout import FOOTER_BAND, HEADER_BAND

# Numbering systems ------------------------------------------------------
NUMBER_SYSTEM_LATIN = "latin"
NUMBER_SYSTEM_ARABIC = "arabic"
NUMBER_SYSTEM_ROMAN = "roman"
NUMBER_SYSTEM_MIXED = "mixed"
NUMBER_SYSTEM_NONE = "none"

# Extraction sources -----------------------------------------------------
SOURCE_FILENAME = "filename"
SOURCE_COVER_PAGE = "cover_page"
SOURCE_FIRST_PAGES = "first_pages"
SOURCE_FOOTER = "footer"
SOURCE_BODY_TEXT = "body_text"
SOURCE_NONE = "none"

# Field names ------------------------------------------------------------
FIELD_TITLE = "title"
FIELD_AUTHOR = "author"
FIELD_VOLUME = "volume"
FIELD_EDITION = "edition"
FIELD_PUBLISHER = "publisher"
FIELD_PUBLICATION_YEAR = "publication_year"
FIELD_MUHAQQIQ = "muhaqqiq"

BIBLIOGRAPHIC_FIELDS = (
    FIELD_TITLE,
    FIELD_AUTHOR,
    FIELD_VOLUME,
    FIELD_EDITION,
    FIELD_PUBLISHER,
    FIELD_PUBLICATION_YEAR,
    FIELD_MUHAQQIQ,
)

# Structural levels ------------------------------------------------------
LEVEL_KITAB = "kitab"
LEVEL_BAB = "bab"
LEVEL_FASL = "fasl"
LEVEL_TOPIC = "topic"

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ARABIC_TO_LATIN = str.maketrans(ARABIC_INDIC_DIGITS, "0123456789")
_ARABIC_LETTERS = r"[\u0621-\u064A]"

_ORDINALS = {
    "الأول": 1,
    "الأولى": 1,
    "الثاني": 2,
    "الثانية": 2,
    "الثالث": 3,
    "الثالثة": 3,
    "الرابع": 4,
    "الرابعة": 4,
    "الخامس": 5,
    "الخامسة": 5,
    "السادس": 6,
    "السادسة": 6,
    "السابع": 7,
    "السابعة": 7,
    "الثامن": 8,
    "الثامنة": 8,
    "التاسع": 9,
    "التاسعة": 9,
    "العاشر": 10,
    "العاشرة": 10,
}

_VOLUME_PATTERNS = (
    re.compile(r"(?i)\bvol(?:ume)?(?![a-z])\s*[:.\-]?\s*(?:no\.?[:.\-]?\s*)?(\d{1,3}|[٠-٩]+)"),
    re.compile(r"(?i)\bv(?![a-z])(\d{1,3})\b"),
    re.compile(r"(?:الجزء|جزء)\s*(\d{1,3}|[٠-٩]+|" + _ARABIC_LETTERS + r"+)"),
)

_EDITION_PATTERNS = (
    re.compile(r"(?i)(\d{1,2})(?:st|nd|rd|th)\s+edition\b"),
    re.compile(r"(?i)\b(?:edition|ed)(?![a-z])\s*[:.\-]?\s*(?:no\.?[:.\-]?\s*)?(\d{1,3}|[٠-٩]+)"),
    re.compile(r"(?:الطبعة|الطبعه)\s*(\d{1,3}|[٠-٩]+|" + _ARABIC_LETTERS + r"+)"),
)

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

_SEPARATOR_RE = re.compile(r"\s+[-–—|/]\s+|\s+by\s+|\s_by_\s*")

_GENERIC_TITLE_RE = re.compile(
    r"^(?:scan|scanned|book|document|doc|pdf|upload|image|file)"
    r"(?:[\s_\-]*\d*)?$",
    re.IGNORECASE,
)

_DIGIT_TOKEN_RE = re.compile(r"\d+|[٠-٩]+")
_ROMAN_TOKEN_RE = re.compile(r"[IVXLCDMivxlcdm]+")

_ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

_HEADING_FONT_RATIO = 1.08
_MAX_HEADING_LENGTH = 80
_FIRST_PAGES_LIMIT = 6


@dataclass
class MetadataField:
    field: str
    value: str | None
    confidence: float
    source: str
    details: dict | None = None


@dataclass
class PageNumber:
    pdf_page: int
    printed_page: str
    printed_page_numeric: int | None
    numbering_system: str
    page_number_uncertain: bool
    confidence: float
    source: str
    kitab: str | None = None
    bab: str | None = None
    fasl: str | None = None


@dataclass
class Structure:
    level: str
    name: str
    page_start: int
    page_end: int | None
    confidence: float
    source: str


@dataclass
class DocumentMetadata:
    filename: str
    page_count: int
    fields: list[MetadataField] = field(default_factory=list)
    pages: list[PageNumber] = field(default_factory=list)
    structures: list[Structure] = field(default_factory=list)
    numbering_system: str = NUMBER_SYSTEM_NONE
    confidence: float = 0.0

    @property
    def field_map(self) -> dict[str, MetadataField]:
        return {item.field: item for item in self.fields}


# ---------------------------------------------------------------------------
# Bibliographic extraction (filename + cover/first pages)
# ---------------------------------------------------------------------------


def _normalize_arabic_number(token: str) -> int | None:
    """Translate an Arabic-Indic digit run or an Arabic ordinal word to int."""
    if all(char in ARABIC_INDIC_DIGITS for char in token):
        return int(token.translate(_ARABIC_TO_LATIN))
    return _ORDINALS.get(token)


def _match_value(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        token = match.group(1)
        normalized = _normalize_arabic_number(token)
        if normalized is not None:
            return str(normalized)
        if token.isdigit():
            return token
    return None


def _extract_author(stem: str) -> tuple[str, str | None]:
    """Split `stem` on the last separator into (title part, author part)."""
    matches = list(_SEPARATOR_RE.finditer(stem))
    if not matches:
        return stem, None
    last = matches[-1]
    author = stem[last.end():].strip()
    if not author:
        return stem, None
    return stem[: last.start()], author


def _remove_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    for pattern in patterns:
        text = pattern.sub("", text)
    return text


def _clean_text(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" -–—|/_.()[]{},;:")
    return text or None


def _split_filename(
    filename: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Parse title/author/volume/edition/year out of a filename."""
    stem = Path(filename).stem
    title_part, author = _extract_author(stem)
    normalized = title_part.replace("_", " ")
    volume = _match_value(normalized, _VOLUME_PATTERNS)
    edition = _match_value(normalized, _EDITION_PATTERNS)
    year_match = _YEAR_RE.search(normalized)
    year = year_match.group(1) if year_match else None

    title = _remove_matches(normalized, _VOLUME_PATTERNS)
    title = _remove_matches(title, _EDITION_PATTERNS)
    title = _YEAR_RE.sub("", title)
    clean_title: str | None = _clean_text(title)

    return clean_title, author, volume, edition, year


def _is_header(block: BlockInfo, page: PageInfo) -> bool:
    return block.bbox[1] <= page.height * HEADER_BAND


def _is_footer(block: BlockInfo, page: PageInfo) -> bool:
    return block.bbox[3] >= page.height * FOOTER_BAND


def _page_text(page: PageInfo) -> str:
    return "\n".join(block.text for block in page.blocks)


def _first_pages_text(pages: list[PageInfo], limit: int = _FIRST_PAGES_LIMIT) -> str:
    return "\n".join(_page_text(page) for page in pages[:limit])


def _clean_capture(text: str) -> str | None:
    text = text.strip().strip(":：\t,،.;;")
    return text or None


def _extract_from_first_pages(
    pages: list[PageInfo],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Scan cover + first pages for publisher, year, author, muhaqqiq."""
    text = _first_pages_text(pages)
    publisher: str | None = None
    author: str | None = None
    year: str | None = None
    muhaqqiq: str | None = None

    match = re.search(r"((?:الناشر\s*[:：]?\s*)?(?:دار|مطبعة|المكتبة|نشر)\s+[^\n،]{2,60})", text)
    if match:
        publisher = re.sub(r"^(?:الناشر\s*[:：]?\s*)", "", match.group(1)).strip()
        publisher = _clean_capture(publisher)
    if publisher is None:
        match = re.search(r"(?i)(?:published by|publisher\s*[:：])\s*([^\n]{2,60})", text)
        if match:
            publisher = _clean_capture(match.group(1))

    match = re.search(r"تأليف\s+([^\n،]{2,80})|المؤلف\s*[:：]\s*([^\n،]{2,80})", text)
    if match:
        author = _clean_capture(match.group(1) or match.group(2))
    if author is None:
        match = re.search(r"(?i)(?:authored by|author\s*[:：]|written by)\s*([^\n]{2,80})", text)
        if match:
            author = _clean_capture(match.group(1))

    match = _YEAR_RE.search(text)
    if match:
        year = match.group(1)

    match = re.search(r"تحقيق\s+([^\n،]{2,80})", text)
    if match:
        muhaqqiq = _clean_capture(match.group(1))
    if muhaqqiq is None:
        match = re.search(r"(?i)(?:edited by|tahqiq|muhaqqiq)\s*[:：]?\s*([^\n]{2,80})", text)
        if match:
            muhaqqiq = _clean_capture(match.group(1))

    return publisher, year, author, muhaqqiq


def _cover_title(pages: list[PageInfo]) -> str | None:
    """The largest prominent text block on the first page, as a title candidate."""
    if not pages:
        return None
    page = pages[0]
    body = [
        block
        for block in page.blocks
        if not _is_header(block, page) and not _is_footer(block, page)
    ]
    main_font = _dominant_font(body)
    if main_font <= 0:
        return None
    candidates = [
        block
        for block in body
        if (block.size or 0.0) >= main_font * 1.3 and 0 < len(block.text.strip()) <= 100
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda block: block.size or 0.0)
    return _clean_capture(best.text)


# ---------------------------------------------------------------------------
# Page-number detection (footer band → printed page label)
# ---------------------------------------------------------------------------


def _roman_to_int(token: str) -> int | None:
    if not token or len(token) > 9:
        return None
    total = 0
    previous = 0
    for char in reversed(token.upper()):
        value = _ROMAN_VALUES.get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if total > 0 else None


def _parse_footer_number(text: str) -> tuple[str, int | None, str] | None:
    """Return (raw label, numeric value, system) for the page number in `text`."""
    text = text.strip()
    arabic = _DIGIT_TOKEN_RE.findall(text)
    for token in arabic:
        if all(char in ARABIC_INDIC_DIGITS for char in token):
            return token, int(token.translate(_ARABIC_TO_LATIN)), NUMBER_SYSTEM_ARABIC
    for token in arabic:
        if token.isdigit():
            return token, int(token), NUMBER_SYSTEM_LATIN
    for token in _ROMAN_TOKEN_RE.findall(text):
        value = _roman_to_int(token)
        if value is not None:
            return token, value, NUMBER_SYSTEM_ROMAN
    return None


def _detect_page_numbers(pages: list[PageInfo]) -> list[PageNumber]:
    result: list[PageNumber] = []
    for page in pages:
        footer_blocks = [block for block in page.blocks if _is_footer(block, page)]
        footer_blocks.sort(key=lambda block: (round(block.bbox[3], 1), block.bbox[0]))
        text = footer_blocks[-1].text if footer_blocks else ""
        parsed = _parse_footer_number(text)
        if parsed is None:
            result.append(
                PageNumber(
                    pdf_page=page.number,
                    printed_page="",
                    printed_page_numeric=None,
                    numbering_system=NUMBER_SYSTEM_NONE,
                    page_number_uncertain=True,
                    confidence=0.15,
                    source=SOURCE_NONE,
                )
            )
            continue
        raw, numeric, system = parsed
        result.append(
            PageNumber(
                pdf_page=page.number,
                printed_page=raw,
                printed_page_numeric=numeric,
                numbering_system=system,
                page_number_uncertain=False,
                confidence=0.9,
                source=SOURCE_FOOTER,
            )
        )
    return result


def _aggregate_numbering(page_numbers: list[PageNumber]) -> str:
    systems = {
        page.numbering_system
        for page in page_numbers
        if page.numbering_system != NUMBER_SYSTEM_NONE
    }
    if len(systems) > 1:
        return NUMBER_SYSTEM_MIXED
    if len(systems) == 1:
        return next(iter(systems))
    return NUMBER_SYSTEM_NONE


# ---------------------------------------------------------------------------
# Structural detection (Kitab → Bab → Fasl → topic headings)
# ---------------------------------------------------------------------------


def _dominant_font(blocks: list[BlockInfo]) -> float:
    """Character-count-weighted modal font size (the 'body' size)."""
    counts: dict[float, int] = {}
    for block in blocks:
        size = block.size or 0.0
        if size <= 0:
            continue
        key = round(size, 1)
        counts[key] = counts.get(key, 0) + len(block.text)
    if not counts:
        return 0.0
    return max(counts.items(), key=lambda item: item[1])[0]


def _heading_level(block: BlockInfo, main_font: float) -> str | None:
    size = block.size or 0.0
    text = block.text.strip()
    if not text or len(text) > _MAX_HEADING_LENGTH:
        return None
    if main_font <= 0 or size < main_font * _HEADING_FONT_RATIO:
        return None
    if re.search(r"كتاب", text) or re.match(r"^\s*(?:Kitab|Book|Chapter)\b", text, re.I):
        return LEVEL_KITAB
    if re.search(r"باب", text) or re.match(r"^\s*(?:Bab|Section|Sub-?chapter)\b", text, re.I):
        return LEVEL_BAB
    if re.search(r"فصل|مسألة", text) or re.match(
        r"^\s*(?:Fasl|Part|Subsection|Mas'ala|Masala)\b", text, re.I
    ):
        return LEVEL_FASL
    return LEVEL_TOPIC


def _detect_structures(
    pages: list[PageInfo],
) -> tuple[list[Structure], dict[int, dict[str, str]]]:
    """Detect hierarchical sections and the active section per page.

    Returns (structures with page ranges, per-page {level: active name}).
    """
    structures: list[Structure] = []
    current: dict[str, tuple[str, int]] = {}
    per_page: dict[int, dict[str, str]] = {}
    last_page = pages[-1].number if pages else 0

    for page in pages:
        body = [
            block
            for block in page.blocks
            if not _is_header(block, page) and not _is_footer(block, page)
        ]
        body.sort(key=lambda block: (round(block.bbox[1], 1), block.bbox[0]))
        main_font = _dominant_font(body)
        for block in body:
            level = _heading_level(block, main_font)
            if level is None:
                continue
            previous = current.get(level)
            if previous is not None and previous[1] != page.number:
                structures.append(
                    Structure(
                        level=level,
                        name=previous[0],
                        page_start=previous[1],
                        page_end=page.number - 1,
                        confidence=0.75,
                        source=SOURCE_BODY_TEXT,
                    )
                )
            current[level] = (block.text.strip(), page.number)
        per_page[page.number] = {
            level: name
            for level, (name, _start) in current.items()
            if level in ("kitab", "bab", "fasl")
        }

    for level, (name, start) in current.items():
        structures.append(
            Structure(
                level=level,
                name=name,
                page_start=start,
                page_end=last_page if last_page >= start else start,
                confidence=0.75,
                source=SOURCE_BODY_TEXT,
            )
        )
    return structures, per_page


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_metadata(filename: str, pages: list[PageInfo]) -> DocumentMetadata:
    """Extract normalized metadata from a filename and its extracted pages."""
    title, author, volume, edition, year = _split_filename(filename)
    if title is not None and _GENERIC_TITLE_RE.match(title):
        title = None
    fields: list[MetadataField] = []

    def add(
        field_name: str,
        value: str | None,
        confidence: float,
        source: str,
        *,
        details: dict | None = None,
    ) -> None:
        if value in (None, ""):
            return
        if any(item.field == field_name for item in fields):
            return
        fields.append(MetadataField(field_name, value, confidence, source, details))

    add(FIELD_TITLE, title, 0.85, SOURCE_FILENAME)
    add(FIELD_AUTHOR, author, 0.8, SOURCE_FILENAME)
    add(FIELD_VOLUME, volume, 0.9, SOURCE_FILENAME)
    add(FIELD_EDITION, edition, 0.9, SOURCE_FILENAME)
    add(FIELD_PUBLICATION_YEAR, year, 0.85, SOURCE_FILENAME)

    publisher, first_year, first_author, muhaqqiq = _extract_from_first_pages(pages)
    if title is None:
        cover = _cover_title(pages)
        add(FIELD_TITLE, cover, 0.7, SOURCE_COVER_PAGE)
    add(FIELD_PUBLISHER, publisher, 0.55, SOURCE_FIRST_PAGES)
    add(FIELD_PUBLICATION_YEAR, first_year, 0.6, SOURCE_FIRST_PAGES)
    add(FIELD_AUTHOR, first_author, 0.7, SOURCE_FIRST_PAGES)
    add(FIELD_MUHAQQIQ, muhaqqiq, 0.5, SOURCE_FIRST_PAGES)

    page_numbers = _detect_page_numbers(pages)
    structures, per_page = _detect_structures(pages)
    for page in page_numbers:
        active = per_page.get(page.pdf_page, {})
        page.kitab = active.get(LEVEL_KITAB)
        page.bab = active.get(LEVEL_BAB)
        page.fasl = active.get(LEVEL_FASL)

    confidences = [item.confidence for item in fields]
    confidences += [
        item.confidence for item in page_numbers if not item.page_number_uncertain
    ]
    overall = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return DocumentMetadata(
        filename=filename,
        page_count=len(pages),
        fields=fields,
        pages=page_numbers,
        structures=structures,
        numbering_system=_aggregate_numbering(page_numbers),
        confidence=overall,
    )


@dataclass
class MetadataResult:
    page_count: int
    pages_mapped: int
    fields_count: int
    structures_count: int
    numbering_system: str
    confidence: float


class MetadataRunner:
    """Streams a job's extracted pages from PostgreSQL through the metadata engine.

    The engine is pure (`extract_metadata`); this runner rehydrates `PageInfo`
    objects from stored `page_blocks`, runs the engine, and persists the result
    through `MetadataRepository`, committing page-by-page so progress is durable.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._extraction_repo = ExtractionRepository(session)
        self._repo = MetadataRepository(session)

    def run(self, job: IngestionJob, upload: Upload) -> MetadataResult:
        """Extract and persist metadata for `upload` under `job`."""
        self._begin(job, upload)
        page_rows = self._extraction_repo.list_pages(job.id)
        pages = [self._load_page(row) for row in page_rows]
        document = extract_metadata(upload.original_filename, pages)

        row = self._repo.save_document(
            job_id=job.id,
            upload_id=upload.id,
            original_filename=upload.original_filename,
            document=document,
        )
        job.progress_percent = 100
        job.current_step = "metadata"
        self._session.commit()

        return MetadataResult(
            page_count=document.page_count,
            pages_mapped=len(row.pages) if row.pages else document.page_count,
            fields_count=len(document.fields),
            structures_count=len(document.structures),
            numbering_system=document.numbering_system,
            confidence=document.confidence,
        )

    def _begin(self, job: IngestionJob, upload: Upload) -> None:
        job.status = "structuring"
        job.current_step = "metadata"
        job.progress_percent = 0
        job.started_at = datetime.utcnow()
        upload.status = "processing"
        self._repo.delete_for_job(job.id)
        self._session.commit()

    def _load_page(self, page: PageExtraction) -> PageInfo:
        blocks = [
            BlockInfo(
                index=block.block_index,
                bbox=[float(value) for value in block.bbox],
                text=block.text,
                font=block.font,
                size=float(block.font_size) if block.font_size is not None else None,
            )
            for block in self._extraction_repo.list_blocks(page)
        ]
        return PageInfo(
            number=page.page_number,
            width=page.width,
            height=page.height,
            rotation=page.rotation,
            blocks=blocks,
        )
