"""Structure-aware chunking engine (Phase 7).

Consumes the Phase 4 `PageInfo` dataclasses, the Phase 5 layout classification
(`analyze_page`) and the Phase 6 page-number mapping to produce retrieval-ready
chunks that respect the book's hierarchy and its printed page boundaries:

1. **Boundary detection** from heading lines — a font-size jump over the
   dominant body size, a short line, and the Kitab/Bab lexicon (reuses the
   Phase 6 heading heuristics) — plus the printed-page number mapping.
2. **Recursive chunking**: split at *Bab* by default, then *Fasl/Mas'ala*,
   then sentence/paragraph boundaries, targeting 256-384 tokens with 15-20%
   overlap applied *only* at sentence boundaries.
3. **Printed-page preservation**: a chunk never merges two printed pages
   unless a sentence is cut mid-way across the page break; every chunk stores
   both pages as `page_start`/`page_end`.
4. Every chunk carries a **parent context header** (Kitab/Bab names) prepended
   to the stored text and mirrored in its metadata columns.

`extract_chunks` is a pure function over `PageInfo` + `PageContext` (no DB).
`ChunkRunner` rehydrates those from PostgreSQL and persists `ChunkResult` rows
through `ChunkRepository`, replacing any previous chunking run for the job.
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.exceptions import ChunkConflictError
from app.db.models import IngestionJob, PageExtraction, Upload
from app.db.repositories import ChunkRepository, ExtractionRepository, MetadataRepository
from app.services.extraction import BlockInfo, DrawingInfo, PageInfo
from app.services.layout import REGION_MAIN, analyze_page
from app.services.metadata import _dominant_font, _heading_level

# Target chunk size (ARCHITECTURE §Phase 6 "Chunk size & overlap"): 256-384
# tokens. `max_tokens` is a soft limit — a single sentence is never split.
DEFAULT_MAX_TOKENS = 320
MIN_TOKENS = 256
MAX_TOKENS = 384
# Overlap fraction (15-20%) applied at sentence boundaries only.
OVERLAP_FRACTION = 0.15

REGION = REGION_MAIN
LANG = "ar"

_SENTENCE_FINAL_RE = re.compile(r"[.؟!؛]+")

_CONTEXT_KEYS = ("kitab", "bab", "fasl", "topic")
_LEVEL_PRIORITY = ("bab", "fasl", "kitab")


@dataclass
class PageContext:
    """Per-page context from the Phase 6 page-number mapping."""

    pdf_page: int
    printed_page: str
    printed_page_numeric: int | None = None


@dataclass
class ChunkResult:
    """One ready-to-persist chunk (mirrors the `chunks` table)."""

    chunk_id: str
    order_index: int
    printed_page_start: int | None
    printed_page_end: int | None
    pdf_page_start: int | None
    pdf_page_end: int | None
    kitab: str | None
    bab: str | None
    fasl: str | None
    topic: str | None
    context_heading: str | None
    region: str
    lang: str
    raw_text: str
    normalized_text: str | None
    token_count: int


@dataclass
class ChunkingResult:
    page_count: int
    chunk_count: int
    pages_covered: int
    token_count: int


# ---------------------------------------------------------------------------
# Pure chunking engine
# ---------------------------------------------------------------------------


@dataclass
class _Unit:
    text: str
    level: str | None  # None for body text, else a metadata LEVEL_* name
    pdf_page: int
    printed_page: int | None
    ends_sentence: bool


@dataclass
class _Sentence:
    text: str
    pdf_page: int
    printed_page: int | None
    unit_key: int


def extract_chunks(
    pages: list[PageInfo],
    contexts: list[PageContext] | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: float = OVERLAP_FRACTION,
) -> list[ChunkResult]:
    """Chunk `pages` into structure-aware, page-anchored chunks.

    Pure and deterministic: no DB, no randomness. Chunks are returned in
    document order with `order_index` assigned sequentially.
    """
    if not pages:
        return []
    max_tokens = max(int(max_tokens), 1)
    if not 0.0 < overlap < 1.0:
        overlap = OVERLAP_FRACTION

    context_by_page = {context.pdf_page: context for context in contexts or []}
    units: list[_Unit] = []
    for page in pages:
        units.extend(_page_units(page, context_by_page.get(page.number)))
    if not units:
        return []

    primary = _primary_level(units)
    boundary_levels = _boundary_levels(primary)
    sub_levels = _sub_levels(primary)

    starts = _boundary_starts(units, boundary_levels)
    chunks: list[ChunkResult] = []
    order = 0
    for start, end in _ranges(starts, len(units), start=0):
        if not any(units[index].level is None for index in range(start, end)):
            continue
        for sub_start, sub_end in _split_range(units, start, end, max_tokens, sub_levels):
            body_units = [
                units[index] for index in range(sub_start, sub_end) if units[index].level is None
            ]
            if not body_units:
                continue
            context = _context_at(units, sub_start)
            sentences = _flatten_sentences(body_units)
            for pack in _sentence_pack(sentences, max_tokens, overlap):
                chunks.append(_build_chunk(pack, context, order))
                order += 1
    return chunks


def _page_units(page: PageInfo, context: PageContext | None) -> list[_Unit]:
    """Main-region blocks of `page` in reading order, with heading levels."""
    layout = analyze_page(page)
    by_index = {item.block_index: item for item in layout.blocks}
    main_blocks = [
        block
        for block in page.blocks
        if by_index.get(block.index) is not None and by_index[block.index].region == REGION_MAIN
    ]
    main_blocks.sort(key=lambda block: by_index[block.index].reading_order)
    main_font = _dominant_font(main_blocks)

    units: list[_Unit] = []
    printed = context.printed_page_numeric if context else None
    for block in main_blocks:
        text = block.text.strip()
        if not text:
            continue
        level = _heading_level(block, main_font)
        units.append(
            _Unit(
                text=text,
                level=level,
                pdf_page=page.number,
                printed_page=printed,
                ends_sentence=bool(_SENTENCE_FINAL_RE.search(text) and text[-1] in ".؟!؛"),
            )
        )
    return units


def _primary_level(units: list[_Unit]) -> str | None:
    """The coarsest split level present: Bab by default, then Fasl, then Kitab."""
    present = {unit.level for unit in units if unit.level in _LEVEL_PRIORITY}
    for level in _LEVEL_PRIORITY:
        if level in present:
            return level
    return None


def _boundary_levels(primary: str | None) -> set[str]:
    """Levels that always split chunks (the primary level and everything above)."""
    if primary is None:
        return set()
    return set(_CONTEXT_KEYS[: _CONTEXT_KEYS.index(primary) + 1])


def _sub_levels(primary: str | None) -> tuple[str, ...]:
    """Finer levels used to recursively split an oversized segment."""
    if primary is None or primary not in _CONTEXT_KEYS:
        return ()
    return tuple(_CONTEXT_KEYS[_CONTEXT_KEYS.index(primary) + 1 :])


def _boundary_starts(units: list[_Unit], boundary_levels: set[str]) -> list[int]:
    """Indices where a new chunk must begin (sections + clean page breaks)."""
    starts: list[int] = []
    current_page = units[0].pdf_page
    for index, unit in enumerate(units):
        if unit.level is not None and unit.level in boundary_levels:
            starts.append(index)
        if unit.pdf_page != current_page:
            if not _page_continues(units, current_page):
                starts.append(index)
            current_page = unit.pdf_page
    return sorted(set(starts))


def _page_continues(units: list[_Unit], page: int) -> bool:
    """True when a sentence is cut mid-way across `page`'s trailing boundary.

    Only the last main-region unit matters: a heading ends the page's flow, and
    body text that does not end with sentence-final punctuation continues onto
    the next printed page (the "never merge pages unless mid-sentence" rule).
    """
    last: _Unit | None = None
    for unit in units:
        if unit.pdf_page == page:
            last = unit
        elif last is not None and unit.pdf_page > page:
            break
    if last is None or last.level is not None:
        return False
    return not last.ends_sentence


def _ranges(starts: list[int], length: int, *, start: int) -> list[tuple[int, int]]:
    """Consecutive [start, end) index ranges covering [start, length)."""
    bounds = [start, *sorted(set(starts)), length]
    deduped: list[int] = []
    for bound in bounds:
        if not deduped or bound != deduped[-1]:
            deduped.append(bound)
    return [(deduped[i], deduped[i + 1]) for i in range(len(deduped) - 1)]


def _split_range(
    units: list[_Unit],
    start: int,
    end: int,
    max_tokens: int,
    sub_levels: tuple[str, ...],
) -> list[tuple[int, int]]:
    """Recursively split an oversized segment at Fasl/Mas'ala heading boundaries."""
    body_tokens = sum(
        _count_tokens(units[index].text)
        for index in range(start, end)
        if units[index].level is None
    )
    if body_tokens <= max_tokens:
        return [(start, end)]
    if sub_levels:
        next_level = sub_levels[0]
        heading_starts = [index for index in range(start, end) if units[index].level == next_level]
        if heading_starts:
            pieces: list[tuple[int, int]] = []
            for sub_start, sub_end in _ranges(heading_starts, end, start=start):
                pieces.extend(_split_range(units, sub_start, sub_end, max_tokens, sub_levels[1:]))
            return pieces
    return [(start, end)]


def _context_at(units: list[_Unit], start: int) -> dict[str, str | None]:
    """The kitab/bab/fasl/topic names in effect when unit `start` is emitted."""
    context: dict[str, str | None] = {key: None for key in _CONTEXT_KEYS}
    for unit in units[: start + 1]:
        if unit.level is not None and unit.level in context:
            context[unit.level] = unit.text
    return context


def _flatten_sentences(body_units: list[_Unit]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    for key, unit in enumerate(body_units):
        for piece in _split_sentences(unit.text):
            sentences.append(
                _Sentence(
                    text=piece,
                    pdf_page=unit.pdf_page,
                    printed_page=unit.printed_page,
                    unit_key=key,
                )
            )
    return sentences


def _split_sentences(text: str) -> list[str]:
    """Split Arabic/Latin text after sentence-final punctuation runs."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_FINAL_RE.finditer(text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _sentence_pack(
    sentences: list[_Sentence], max_tokens: int, overlap: float
) -> list[list[_Sentence]]:
    """Greedily pack sentences into chunks, overlapping only at sentence ends."""
    if not sentences:
        return []
    overlap_tokens = max(1, int(max_tokens * overlap))
    packs: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    current_tokens = 0
    for sentence in sentences:
        tokens = _count_tokens(sentence.text)
        if current and current_tokens + tokens > max_tokens:
            carry = _tail_sentences(current, overlap_tokens)
            packs.append(current)
            current = carry
            current_tokens = sum(_count_tokens(item.text) for item in current)
        current.append(sentence)
        current_tokens += tokens
    if current:
        packs.append(current)
    return packs


def _tail_sentences(sentences: list[_Sentence], overlap_tokens: int) -> list[_Sentence]:
    """The trailing sentences of a chunk to carry into the next one."""
    tail: list[_Sentence] = []
    total = 0
    for sentence in reversed(sentences):
        tokens = _count_tokens(sentence.text)
        if tail and total + tokens > overlap_tokens:
            break
        total += tokens
        tail.append(sentence)
    return list(reversed(tail))


def _count_tokens(text: str) -> int:
    """Whitespace-token estimate (word count); exact tokenization is Phase 8."""
    return len(re.findall(r"\S+", text))


def _build_chunk(pack: list[_Sentence], context: dict[str, str | None], order: int) -> ChunkResult:
    body = _join_body(pack)
    context_heading = _build_context_heading(context)
    raw_text = f"{context_heading}\n\n{body}" if context_heading else body
    first, last = pack[0], pack[-1]
    return ChunkResult(
        chunk_id=hashlib.sha256(f"{order}:{raw_text}".encode()).hexdigest(),
        order_index=order,
        printed_page_start=first.printed_page,
        printed_page_end=last.printed_page,
        pdf_page_start=first.pdf_page,
        pdf_page_end=last.pdf_page,
        kitab=context["kitab"],
        bab=context["bab"],
        fasl=context["fasl"],
        topic=context["topic"],
        context_heading=context_heading,
        region=REGION,
        lang=LANG,
        raw_text=raw_text,
        normalized_text=_normalize_text(body),
        token_count=_count_tokens(raw_text),
    )


def _join_body(pack: list[_Sentence]) -> str:
    parts: list[str] = []
    last_key: int | None = None
    for sentence in pack:
        if last_key is None:
            parts.append(sentence.text)
        elif sentence.unit_key != last_key:
            parts.append("\n")
            parts.append(sentence.text)
        else:
            parts.append(" ")
            parts.append(sentence.text)
        last_key = sentence.unit_key
    return "".join(parts)


def _build_context_heading(context: dict[str, str | None]) -> str | None:
    lines = []
    for level, label in (("kitab", "Kitab"), ("bab", "Bab"), ("fasl", "Fasl"), ("topic", "Topic")):
        name = context.get(level)
        if name:
            lines.append(f"{label}: {name}")
    return "\n".join(lines) or None


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# ChunkRunner (DB streaming + persistence)
# ---------------------------------------------------------------------------


class ChunkRunner:
    """Streams a job's extracted pages + metadata mapping through the engine.

    The engine is pure (`extract_chunks`); this runner rehydrates `PageInfo`
    objects (with drawings, so `analyze_page` can find footnote separators),
    builds `PageContext` from the Phase 6 page mapping, and persists the
    result through `ChunkRepository`, replacing any previous run for the job.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._extraction_repo = ExtractionRepository(session)
        self._metadata_repo = MetadataRepository(session)
        self._chunk_repo = ChunkRepository(session)

    def run(
        self,
        job: IngestionJob,
        upload: Upload,
        *,
        metadata_job_id: str | None = None,
    ) -> ChunkingResult:
        """Chunk `upload`'s pages under `job` using the job's metadata mapping."""
        document = self._load_metadata_document(job, upload, metadata_job_id)
        self._begin(job, upload)

        page_rows = self._load_page_rows(upload)
        pages = [self._load_page(row) for row in page_rows]
        page_meta = self._metadata_repo.list_pages(document)
        contexts = [
            PageContext(
                pdf_page=item.pdf_page,
                printed_page=item.printed_page,
                printed_page_numeric=item.printed_page_numeric,
            )
            for item in page_meta
        ]
        chunks = extract_chunks(pages, contexts)
        self._chunk_repo.save_for_job(job.id, chunks)

        job.progress_percent = 100
        job.current_step = "chunking"
        self._session.commit()

        return ChunkingResult(
            page_count=len(pages),
            chunk_count=len(chunks),
            pages_covered=len(
                {page for chunk in chunks for page in (chunk.pdf_page_start, chunk.pdf_page_end)}
            ),
            token_count=sum(chunk.token_count for chunk in chunks),
        )

    def _load_metadata_document(
        self,
        job: IngestionJob,
        upload: Upload,
        metadata_job_id: str | None,
    ):
        from app.db.repositories import IngestionJobRepository

        if metadata_job_id is None and upload.id:
            metadata_job = IngestionJobRepository(self._session).find_for_upload(
                upload.id, "metadata"
            )
            metadata_job_id = metadata_job.id if metadata_job else None
        document = self._metadata_repo.get_by_job(metadata_job_id) if metadata_job_id else None
        if document is None:
            raise ChunkConflictError(
                "chunking requires a completed metadata extraction (metadata document is missing)"
            )
        return document

    def _load_page_rows(self, upload: Upload):
        """The page rows live under the upload's extraction job, not this job."""
        from app.db.repositories import IngestionJobRepository

        pipeline_job = IngestionJobRepository(self._session).find_pipeline_job(upload.id)
        if pipeline_job is None:
            raise ChunkConflictError(
                "chunking requires a completed extraction job (no extracted pages found)"
            )
        return self._extraction_repo.list_pages(pipeline_job.id)

    def _begin(self, job: IngestionJob, upload: Upload) -> None:
        job.status = "structuring"
        job.current_step = "chunking"
        job.progress_percent = 0
        job.started_at = datetime.utcnow()
        upload.status = "processing"
        self._chunk_repo.delete_for_job(job.id)
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
        drawings = [
            DrawingInfo(
                index=drawing.drawing_index,
                bbox=[float(value) for value in drawing.bbox],
                kind=drawing.kind,
                stroke_width=drawing.stroke_width,
            )
            for drawing in self._extraction_repo.list_drawings(page)
        ]
        return PageInfo(
            number=page.page_number,
            width=page.width,
            height=page.height,
            rotation=page.rotation,
            blocks=blocks,
            drawings=drawings,
        )
