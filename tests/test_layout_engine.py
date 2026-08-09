"""Tests for the Phase 5 pure rule-based layout engine (`analyze_page`)."""

from app.services.extraction import BlockInfo, DrawingInfo, PageInfo
from app.services.layout import (
    DIRECTION_LTR,
    DIRECTION_RTL,
    REGION_FOOTER,
    REGION_FOOTNOTE,
    REGION_HEADER,
    REGION_MAIN,
    REGION_MARGIN,
    REGION_UNKNOWN,
    analyze_page,
    detect_direction,
)


def _block(
    index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "sample text block content",
    size: float = 14.0,
    font: str = "Helvetica",
) -> BlockInfo:
    return BlockInfo(index=index, bbox=[x0, y0, x1, y1], text=text, font=font, size=size)


def _page(
    blocks: list[BlockInfo],
    *,
    width: float = 595,
    height: float = 842,
    drawings: list[DrawingInfo] | None = None,
) -> PageInfo:
    return PageInfo(
        number=1,
        width=width,
        height=height,
        rotation=0,
        blocks=blocks,
        drawings=drawings or [],
    )


def _main_block(index: int, y0: float) -> BlockInfo:
    return _block(index, 72, y0, 300, y0 + 16, size=14.0)


def test_empty_page_is_ltr_with_no_columns() -> None:
    layout = analyze_page(_page([]))
    assert layout.direction == DIRECTION_LTR
    assert layout.column_count == 0
    assert layout.blocks == []


def test_single_column_classifies_all_blocks_as_main() -> None:
    layout = analyze_page(_page([_main_block(0, 100), _main_block(1, 140), _main_block(2, 180)]))
    assert layout.column_count == 1
    assert all(item.region == REGION_MAIN for item in layout.blocks)


def test_header_band_classified_as_header() -> None:
    block = _block(0, 200, 10, 400, 26, text="Kitab al-Taharah", size=10)
    layout = analyze_page(_page([block, _main_block(1, 100)]))
    assert layout.blocks[0].region == REGION_HEADER
    assert layout.blocks[0].reading_order == 0
    assert layout.blocks[0].confidence == 0.95


def test_footer_band_classified_as_footer() -> None:
    block = _block(0, 250, 810, 270, 826, text="7", size=10)
    layout = analyze_page(_page([_main_block(1, 100), block]))
    assert layout.blocks[1].region == REGION_FOOTER
    assert layout.blocks[1].reading_order == 1


def test_tiny_block_in_middle_is_unknown() -> None:
    block = _block(0, 200, 300, 205, 305, text="x", size=6)
    layout = analyze_page(_page([_main_block(1, 100), block]))
    assert layout.blocks[1].region == REGION_UNKNOWN
    assert layout.blocks[1].confidence < 0.5


def test_margin_column_detected_from_narrow_side_column() -> None:
    margin = _block(0, 12, 300, 60, 340, text="hashiya", size=10)
    layout = analyze_page(
        _page(
            [
                margin,
                _main_block(1, 100),
                _main_block(2, 140),
                _main_block(3, 180),
                _main_block(4, 220),
            ]
        )
    )
    assert layout.column_count == 2
    assert layout.blocks[0].region == REGION_MARGIN
    assert all(item.region == REGION_MAIN for item in layout.blocks[1:])


def test_footnote_below_separator_rule() -> None:
    drawing = DrawingInfo(index=0, bbox=[72, 650, 420, 651], kind="s", stroke_width=1)
    footnote = _block(0, 72, 680, 300, 695, text="footnote text here", size=10)
    layout = analyze_page(
        _page([_main_block(1, 100), _main_block(2, 140), footnote], drawings=[drawing])
    )
    assert layout.blocks[2].region == REGION_FOOTNOTE
    assert layout.blocks[2].confidence == 0.95


def test_footnote_in_lower_band_with_smaller_font() -> None:
    footnote = _block(0, 72, 650, 300, 665, text="footnote text here", size=10)
    layout = analyze_page(_page([_main_block(1, 100), _main_block(2, 140), footnote]))
    assert layout.blocks[2].region == REGION_FOOTNOTE
    assert layout.blocks[2].confidence == 0.9


def test_reading_order_is_column_major_ltr() -> None:
    left = [_main_block(0, 100), _main_block(1, 140), _main_block(2, 180)]
    right = [_block(3 + i, 320, y, 540, y + 16) for i, y in enumerate((100, 140, 180))]
    layout = analyze_page(_page(left + right))
    order = {item.block_index: item.reading_order for item in layout.blocks}
    assert [order[i] for i in (0, 1, 2)] == [0, 1, 2]
    assert [order[i] for i in (3, 4, 5)] == [3, 4, 5]


def test_reading_order_is_column_major_rtl() -> None:
    right = [
        _block(i, 320, y, 540, y + 16, text="كتاب الطهارة والأحكام")
        for i, y in enumerate((100, 140, 180))
    ]
    left = [_block(3 + i, 72, y, 300, y + 16, text="abc") for i, y in enumerate((100, 140, 180))]
    layout = analyze_page(_page(right + left))
    assert layout.direction == DIRECTION_RTL
    order = {item.block_index: item.reading_order for item in layout.blocks}
    assert [order[i] for i in (0, 1, 2)] == [0, 1, 2]
    assert [order[i] for i in (3, 4, 5)] == [3, 4, 5]


def test_detect_direction_rtl_from_arabic_script() -> None:
    arabic = _block(0, 72, 100, 300, 116, text="كتاب الطهارة والأحكام")
    latin = _block(1, 72, 140, 200, 156, text="some latin text")
    assert detect_direction([arabic, latin]) == DIRECTION_RTL
    assert detect_direction([latin, latin]) == DIRECTION_LTR


def test_page_without_text_blocks_still_reports_page_number() -> None:
    layout = analyze_page(_page([], drawings=[]))
    assert layout.page_number == 1
