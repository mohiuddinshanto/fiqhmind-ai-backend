"""Pure rule-based layout analysis engine (Phase 5).

Classifies every extracted text block on a page into exactly one region
(`main`, `footnote`, `margin`, `header`, `footer`, `unknown`) using only
geometry — bounding boxes, relative page position, font size, font weight,
block density and whitespace gaps — and reconstructs a deterministic reading
order across single-column, double-column, RTL and mixed layouts.

No OCR, no metadata extraction, no chunking, no AI models.

`analyze_page` is a pure function over the Phase 4 `PageInfo` dataclasses;
`LayoutRunner` applies it to pages read from PostgreSQL and writes the results
back into `page_blocks`.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import IngestionJob, PageBlock, PageExtraction
from app.db.repositories import ExtractionRepository
from app.services.extraction import BlockInfo, DrawingInfo, PageInfo

REGION_MAIN = "main"
REGION_FOOTNOTE = "footnote"
REGION_MARGIN = "margin"
REGION_HEADER = "header"
REGION_FOOTER = "footer"
REGION_UNKNOWN = "unknown"

REGIONS = (
    REGION_MAIN,
    REGION_FOOTNOTE,
    REGION_MARGIN,
    REGION_HEADER,
    REGION_FOOTER,
    REGION_UNKNOWN,
)

DIRECTION_LTR = "ltr"
DIRECTION_RTL = "rtl"

# Geometry constants (relative to page dimensions).
HEADER_BAND = 0.07  # top strip: y1 <= 7% of height → header
FOOTER_BAND = 0.93  # bottom strip: y0 >= 93% of height → footer
FOOTNOTE_BAND = 0.72  # lower band: y0 >= 72% of height is footnote territory
FOOTNOTE_FONT_RATIO = 0.85  # footnote font must be < 85% of the main font
COLUMN_GAP = 0.18  # whitespace gap > 18% of page width splits columns
MIN_BLOCK_WIDTH_RATIO = 0.02
MIN_BLOCK_HEIGHT_RATIO = 0.004

BOLD_FLAG = 16  # PyMuPDF span flag bit 4 = bold

_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


@dataclass
class BlockLayout:
    block_index: int
    region: str
    reading_order: int
    confidence: float
    classification_reason: str


@dataclass
class PageLayout:
    page_number: int
    direction: str
    column_count: int
    blocks: list[BlockLayout]


def _is_arabic(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _ARABIC_RANGES)


def detect_direction(blocks: list[BlockInfo]) -> str:
    """Detect page direction from the majority script of the text layer."""
    arabic = 0
    letters = 0
    for block in blocks:
        for char in block.text:
            if _is_arabic(char):
                arabic += 1
            elif char.isalpha():
                letters += 1
    return DIRECTION_RTL if arabic > letters else DIRECTION_LTR


def _separator_y(page: PageInfo) -> float | None:
    """Return the y of a horizontal rule separating footnotes, if any."""
    for drawing in page.drawings:
        rect = drawing.bbox
        width = rect[2] - rect[0]
        y_center = (rect[1] + rect[3]) / 2
        if width >= page.width * 0.5 and page.height * 0.55 <= y_center <= page.height * 0.95:
            return y_center
    return None


def _body_blocks(page: PageInfo) -> list[BlockInfo]:
    """Blocks that are not in the header or footer bands."""
    body = []
    for block in page.blocks:
        x0, y0, x1, y1 = block.bbox
        if y1 <= page.height * HEADER_BAND:
            continue
        if y0 >= page.height * FOOTER_BAND:
            continue
        body.append(block)
    return body


def _main_font_size(blocks: list[BlockInfo]) -> float:
    sizes = [block.size or 0.0 for block in blocks]
    return max(sizes) if sizes else 0.0


def _column_span(column: list[BlockInfo]) -> tuple[float, float, float]:
    """(min x0, max x1, width) for a column."""
    min_x = min(block.bbox[0] for block in column)
    max_x = max(block.bbox[2] for block in column)
    return min_x, max_x, max_x - min_x


def _margin_block_indices(page: PageInfo, columns: list[list[BlockInfo]]) -> set[int]:
    """Indices of blocks that sit in narrow columns outside the main column.

    A second *column* (double-column layout) spans a comparable width to the
    dominant column and is not a margin; a *margin* (hamesh gloss) is a narrow
    column that is clearly separated from the dominant one.
    """
    indices: set[int] = set()
    if not columns:
        return indices
    dominant = max(columns, key=lambda column: _column_span(column)[2])
    dominant_x0, dominant_x1, dominant_width = _column_span(dominant)
    if dominant_width <= 0:
        return indices
    for column in columns:
        if column is dominant:
            continue
        column_x0, column_x1, column_width = _column_span(column)
        overlap = min(column_x1, dominant_x1) - max(column_x0, dominant_x0)
        overlap_ratio = max(0.0, overlap) / column_width if column_width > 0 else 1.0
        if column_width < dominant_width * 0.6 and overlap_ratio < 0.5:
            indices.update(block.index for block in column)
    return indices


def _classify_block(
    block: BlockInfo,
    page: PageInfo,
    separator_y: float | None,
    margin_indices: set[int],
) -> tuple[str, float, str]:
    """Classify one block into a region using deterministic geometry rules."""
    x0, y0, x1, y1 = block.bbox
    width = x1 - x0
    height = y1 - y0

    if page.width <= 0 or page.height <= 0:
        return REGION_UNKNOWN, 0.3, "page has invalid dimensions"

    if y1 <= page.height * HEADER_BAND:
        return REGION_HEADER, 0.95, "top band (running header position)"

    if y0 >= page.height * FOOTER_BAND:
        return REGION_FOOTER, 0.95, "bottom band (page number / running footer)"

    if width < page.width * MIN_BLOCK_WIDTH_RATIO or height < page.height * MIN_BLOCK_HEIGHT_RATIO:
        return REGION_UNKNOWN, 0.3, "block too small for reliable region detection"

    if block.index in margin_indices:
        return REGION_MARGIN, 0.85, "narrow column outside the main text column"

    if separator_y is not None and y0 > separator_y:
        return REGION_FOOTNOTE, 0.95, "below the footnote separator rule"

    font_size = block.size or 0.0
    main_font = _main_font_size(_body_blocks(page))
    if y0 >= page.height * FOOTNOTE_BAND and font_size < main_font * FOOTNOTE_FONT_RATIO:
        return REGION_FOOTNOTE, 0.9, "lower band with smaller font than main text"

    return REGION_MAIN, 0.8, "dominant column with main-text font size"


def _cluster_columns(blocks: list[BlockInfo], width: float) -> list[list[BlockInfo]]:
    """Deterministically cluster body blocks into columns by x-center gap."""
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda block: (block.bbox[0] + block.bbox[2]) / 2)
    columns: list[list[BlockInfo]] = []
    current: list[BlockInfo] = []
    previous_center: float | None = None
    for block in ordered:
        center = (block.bbox[0] + block.bbox[2]) / 2
        if previous_center is not None and center - previous_center > width * COLUMN_GAP:
            columns.append(current)
            current = []
        current.append(block)
        previous_center = center
    if current:
        columns.append(current)
    return columns


def _column_position(column: list[BlockInfo]) -> tuple[float, float]:
    return min(block.bbox[0] for block in column), max(block.bbox[2] for block in column)


def _reading_order(page: PageInfo, direction: str) -> dict[int, int]:
    """Map block_index → deterministic reading order (0-based).

    Headers first (top), then body flow in column-major order (right-to-left
    for RTL, left-to-right for LTR), then footers last.
    """
    headers = [block for block in page.blocks if block.bbox[1] <= page.height * HEADER_BAND]
    footers = [block for block in page.blocks if block.bbox[3] >= page.height * FOOTER_BAND]
    body = [block for block in page.blocks if block not in headers and block not in footers]

    columns = _cluster_columns(body, page.width)
    if direction == DIRECTION_RTL:
        columns.sort(key=_column_position, reverse=True)
    else:
        columns.sort(key=_column_position)

    order: dict[int, int] = {}
    next_index = 0
    for block in sorted(headers, key=lambda item: (item.bbox[1], item.bbox[0])):
        order[block.index] = next_index
        next_index += 1
    for column in columns:
        for block in sorted(column, key=lambda item: (round(item.bbox[1], 1), item.bbox[0])):
            order[block.index] = next_index
            next_index += 1
    for block in sorted(footers, key=lambda item: (item.bbox[1], item.bbox[0])):
        order[block.index] = next_index
        next_index += 1
    return order


def analyze_page(page: PageInfo) -> PageLayout:
    """Classify every block on `page` and assign a deterministic reading order."""
    if not page.blocks:
        return PageLayout(
            page_number=page.number,
            direction=DIRECTION_LTR,
            column_count=0,
            blocks=[],
        )

    direction = detect_direction(page.blocks)
    body = _body_blocks(page)
    columns = _cluster_columns(body, page.width)
    margin_indices = _margin_block_indices(page, columns)
    separator_y = _separator_y(page)
    order = _reading_order(page, direction)

    classifications = []
    for block in page.blocks:
        region, confidence, reason = _classify_block(block, page, separator_y, margin_indices)
        classifications.append(
            BlockLayout(
                block_index=block.index,
                region=region,
                reading_order=order.get(block.index, 0),
                confidence=confidence,
                classification_reason=reason,
            )
        )

    return PageLayout(
        page_number=page.number,
        direction=direction,
        column_count=len(columns),
        blocks=classifications,
    )


@dataclass
class LayoutResult:
    page_count: int
    block_count: int
    region_counts: dict[str, int]


class LayoutRunner:
    """Streams a job's extracted pages from PostgreSQL through the layout engine.

    The engine itself is pure (`analyze_page`); this runner reads the stored
    `page_blocks`/`page_drawings` rows, rehydrates `PageInfo` objects, writes
    each classification back, and updates job progress page by page so a
    crashed run is durable and resumable.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._repo = ExtractionRepository(session)

    def run(self, job: IngestionJob) -> LayoutResult:
        """Classify every stored page of `job` and persist the results."""
        self._begin(job)
        pages = self._repo.list_pages(job.id)
        total_blocks = 0
        region_counts = {region: 0 for region in REGIONS}
        for index, page_row in enumerate(pages):
            page_info, block_rows = self._load_page(page_row)
            layout = analyze_page(page_info)
            for item in layout.blocks:
                block_row = block_rows.get(item.block_index)
                if block_row is None:
                    continue
                self._repo.update_block_layout(
                    block_row,
                    region=item.region,
                    reading_order=item.reading_order,
                    confidence=item.confidence,
                    classification_reason=item.classification_reason,
                )
                region_counts[item.region] = region_counts.get(item.region, 0) + 1
                total_blocks += 1
            if pages:
                job.progress_percent = round(((index + 1) / len(pages)) * 100)
            self._session.commit()
        return LayoutResult(
            page_count=len(pages),
            block_count=total_blocks,
            region_counts=region_counts,
        )

    def _begin(self, job: IngestionJob) -> None:
        job.status = "structuring"
        job.current_step = "layout"
        job.progress_percent = 0
        job.started_at = datetime.utcnow()
        self._session.commit()

    def _load_page(self, page: PageExtraction) -> tuple[PageInfo, dict[int, PageBlock]]:
        block_rows = {block.block_index: block for block in self._repo.list_blocks(page)}
        blocks = [
            BlockInfo(
                index=block.block_index,
                bbox=[float(value) for value in block.bbox],
                text=block.text,
                font=block.font,
                size=float(block.font_size) if block.font_size is not None else None,
            )
            for block in block_rows.values()
        ]
        drawings = [
            DrawingInfo(
                index=drawing.drawing_index,
                bbox=[float(value) for value in drawing.bbox],
                kind=drawing.kind,
                stroke_width=drawing.stroke_width,
            )
            for drawing in self._repo.list_drawings(page)
        ]
        page_info = PageInfo(
            number=page.page_number,
            width=page.width,
            height=page.height,
            rotation=page.rotation,
            blocks=blocks,
            drawings=drawings,
        )
        return page_info, block_rows
