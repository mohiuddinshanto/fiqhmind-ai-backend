from sqlalchemy import Integer, cast, func, select

from app.db.models import (
    PageBlock,
    PageDrawing,
    PageExtraction,
    PageImage,
    PageSpan,
)
from app.db.repositories.base import RepositoryBase


class ExtractionRepository(RepositoryBase[PageExtraction]):
    model = PageExtraction

    def get_page(self, job_id: str, page_number: int) -> PageExtraction | None:
        return self._session.scalar(
            select(PageExtraction).where(
                PageExtraction.job_id == job_id,
                PageExtraction.page_number == page_number,
            )
        )

    def list_pages(self, job_id: str) -> list[PageExtraction]:
        return list(
            self._session.scalars(
                select(PageExtraction)
                .where(PageExtraction.job_id == job_id)
                .order_by(PageExtraction.page_number)
            )
        )

    def count_pages(self, job_id: str) -> int:
        return int(
            self._session.scalar(
                select(func.count()).select_from(PageExtraction).where(
                    PageExtraction.job_id == job_id
                )
            )
            or 0
        )

    def create_page(self, **fields: object) -> PageExtraction:
        page = PageExtraction(**fields)
        self._session.add(page)
        self._session.flush()
        return page

    def add_block(
        self,
        page: PageExtraction,
        *,
        block_index: int,
        bbox: list[float],
        text: str,
        font: str | None,
        font_size: float | None,
    ) -> PageBlock:
        block = PageBlock(
            page_extraction_id=page.id,
            block_index=block_index,
            bbox=bbox,
            text=text,
            font=font,
            font_size=font_size,
            span_count=0,
        )
        self._session.add(block)
        self._session.flush()
        return block

    def add_span(
        self,
        block: PageBlock,
        *,
        span_index: int,
        text: str,
        font: str,
        font_size: float,
        bbox: list[float],
        flags: int | None = None,
    ) -> PageSpan:
        span = PageSpan(
            block_id=block.id,
            span_index=span_index,
            text=text,
            font=font,
            font_size=font_size,
            bbox=bbox,
            flags=flags,
        )
        self._session.add(span)
        return span

    def add_image(
        self,
        page: PageExtraction,
        *,
        image_index: int,
        bbox: list[float],
        width: int | None,
        height: int | None,
        xref: int | None,
    ) -> PageImage:
        image = PageImage(
            page_extraction_id=page.id,
            image_index=image_index,
            bbox=bbox,
            width=width,
            height=height,
            xref=xref,
        )
        self._session.add(image)
        return image

    def add_drawing(
        self,
        page: PageExtraction,
        *,
        drawing_index: int,
        bbox: list[float],
        kind: str | None,
        stroke_width: float | None,
    ) -> PageDrawing:
        drawing = PageDrawing(
            page_extraction_id=page.id,
            drawing_index=drawing_index,
            bbox=bbox,
            kind=kind,
            stroke_width=stroke_width,
        )
        self._session.add(drawing)
        return drawing

    def delete_for_job(self, job_id: str) -> None:
        """Remove every extracted page (and its blocks/spans/images/drawings) for a job."""
        pages = self.list_pages(job_id)
        for page in pages:
            self._session.delete(page)
        self._session.commit()

    def list_blocks(self, page: PageExtraction) -> list[PageBlock]:
        """All blocks of a page in extraction order (stable block_index sort)."""
        return list(
            self._session.scalars(
                select(PageBlock)
                .where(PageBlock.page_extraction_id == page.id)
                .order_by(PageBlock.block_index)
            )
        )

    def list_drawings(self, page: PageExtraction) -> list[PageDrawing]:
        """All vector drawings of a page in extraction order."""
        return list(
            self._session.scalars(
                select(PageDrawing)
                .where(PageDrawing.page_extraction_id == page.id)
                .order_by(PageDrawing.drawing_index)
            )
        )

    def update_block_layout(
        self,
        block: PageBlock,
        *,
        region: str,
        reading_order: int,
        confidence: float,
        classification_reason: str,
    ) -> PageBlock:
        """Persist the Phase 5 layout classification for one block."""
        block.region = region
        block.reading_order = reading_order
        block.confidence = confidence
        block.classification_reason = classification_reason
        return block

    def region_summary(self, job_id: str) -> dict[str, int]:
        """Count of blocks per layout region across a job's pages."""
        rows = self._session.execute(
            select(PageBlock.region, func.count(PageBlock.id))
            .join(PageExtraction, PageExtraction.id == PageBlock.page_extraction_id)
            .where(PageExtraction.job_id == job_id)
            .group_by(PageBlock.region)
        ).all()
        return {region: int(count) for region, count in rows}

    def job_summary(self, job_id: str) -> dict[str, int | float]:
        rows = self._session.execute(
            select(
                func.count(PageExtraction.id),
                func.sum(PageExtraction.char_count),
                func.sum(PageExtraction.block_count),
                func.sum(PageExtraction.image_count),
                func.sum(PageExtraction.drawing_count),
                func.sum(cast(PageExtraction.has_text, Integer)),
                func.coalesce(func.avg(PageExtraction.confidence), 0.0),
            ).where(PageExtraction.job_id == job_id)
        ).one()
        return {
            "page_count": int(rows[0] or 0),
            "char_count": int(rows[1] or 0),
            "block_count": int(rows[2] or 0),
            "image_count": int(rows[3] or 0),
            "drawing_count": int(rows[4] or 0),
            "text_pages": int(rows[5] or 0),
            "confidence": float(rows[6] or 0.0),
        }
