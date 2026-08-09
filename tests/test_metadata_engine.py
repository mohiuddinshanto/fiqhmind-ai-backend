"""Tests for the Phase 6 pure metadata extraction engine."""

from app.services.extraction import BlockInfo, PageInfo
from app.services.metadata import (
    NUMBER_SYSTEM_ARABIC,
    NUMBER_SYSTEM_LATIN,
    NUMBER_SYSTEM_MIXED,
    NUMBER_SYSTEM_NONE,
    NUMBER_SYSTEM_ROMAN,
    SOURCE_BODY_TEXT,
    SOURCE_COVER_PAGE,
    SOURCE_FILENAME,
    SOURCE_FIRST_PAGES,
    SOURCE_FOOTER,
    extract_metadata,
)

WIDTH = 595.0
HEIGHT = 842.0


def _block(
    index: int,
    text: str,
    *,
    x0: float = 72.0,
    y0: float = 120.0,
    y1: float = 135.0,
    size: float = 12.0,
) -> BlockInfo:
    return BlockInfo(index=index, bbox=[x0, y0, x0 + 400, y1], text=text, font="helv", size=size)


def _footer(text: str, index: int = 9) -> BlockInfo:
    return _block(index, text, y0=805.0, y1=820.0, size=10.0)


def _page(number: int, blocks: list[BlockInfo]) -> PageInfo:
    return PageInfo(number=number, width=WIDTH, height=HEIGHT, rotation=0, blocks=blocks)


def _body_blocks(*texts: str) -> list[BlockInfo]:
    def _at(index: int, text: str) -> BlockInfo:
        return _block(index, text, y0=160.0 + index * 20, y1=175.0 + index * 20)

    if texts:
        return [_at(index, text) for index, text in enumerate(texts)]
    default = [
        "This is a reasonably long line of body text on this page.",
        "Here is another fairly long line of body text to match it.",
        "A third line, also long enough to dominate font statistics.",
    ]
    return [_at(index, text) for index, text in enumerate(default)]


def test_single_volume_filename() -> None:
    metadata = extract_metadata("Al-Mabsut.pdf", [_page(1, _body_blocks("Some text"))])
    fields = metadata.field_map
    assert fields["title"].value == "Al-Mabsut"
    assert fields["title"].source == SOURCE_FILENAME
    assert "volume" not in fields


def test_multi_volume_filename_variants() -> None:
    metadata = extract_metadata("Al-Mabsut v2.pdf", [_page(1, _body_blocks("text"))])
    assert metadata.field_map["title"].value == "Al-Mabsut"
    assert metadata.field_map["volume"].value == "2"

    metadata = extract_metadata("Al-Mabsut_Volume_2.pdf", [_page(1, _body_blocks("text"))])
    assert metadata.field_map["volume"].value == "2"

    metadata = extract_metadata("Mukhtasar_الجزء_الثاني.pdf", [_page(1, _body_blocks("نص"))])
    assert metadata.field_map["title"].value == "Mukhtasar"
    assert metadata.field_map["volume"].value == "2"


def test_author_from_filename_separator() -> None:
    metadata = extract_metadata("Sahih Muslim - Imam Muslim.pdf", [_page(1, _body_blocks("text"))])
    assert metadata.field_map["title"].value == "Sahih Muslim"
    assert metadata.field_map["author"].value == "Imam Muslim"


def test_edition_and_year_from_filename() -> None:
    metadata = extract_metadata("Al-Mabsut ed3 1998.pdf", [_page(1, _body_blocks("text"))])
    assert metadata.field_map["title"].value == "Al-Mabsut"
    assert metadata.field_map["edition"].value == "3"
    assert metadata.field_map["publication_year"].value == "1998"


def test_latin_page_numbering() -> None:
    pages = [
        _page(1, _body_blocks("body") + [_footer("7")]),
        _page(2, _body_blocks("body") + [_footer("8")]),
    ]
    metadata = extract_metadata("book.pdf", pages)
    assert metadata.numbering_system == NUMBER_SYSTEM_LATIN
    page1 = metadata.pages[0]
    assert page1.pdf_page == 1
    assert page1.printed_page == "7"
    assert page1.printed_page_numeric == 7
    assert page1.page_number_uncertain is False
    assert page1.source == SOURCE_FOOTER


def test_arabic_page_numbering_normalized() -> None:
    pages = [
        _page(1, _body_blocks("body") + [_footer("٥")]),
        _page(2, _body_blocks("body") + [_footer("٦")]),
    ]
    metadata = extract_metadata("book.pdf", pages)
    assert metadata.numbering_system == NUMBER_SYSTEM_ARABIC
    assert metadata.pages[0].printed_page == "٥"
    assert metadata.pages[0].printed_page_numeric == 5
    assert metadata.pages[1].printed_page_numeric == 6


def test_missing_page_numbers() -> None:
    metadata = extract_metadata("book.pdf", [_page(1, _body_blocks("body"))])
    assert metadata.numbering_system == NUMBER_SYSTEM_NONE
    page = metadata.pages[0]
    assert page.page_number_uncertain is True
    assert page.printed_page == ""
    assert page.printed_page_numeric is None
    assert page.pdf_page == 1  # never lost


def test_mixed_numbering() -> None:
    pages = [
        _page(1, _body_blocks("body") + [_footer("i")]),
        _page(2, _body_blocks("body") + [_footer("ii")]),
        _page(3, _body_blocks("body") + [_footer("1")]),
    ]
    metadata = extract_metadata("book.pdf", pages)
    assert metadata.numbering_system == NUMBER_SYSTEM_MIXED
    assert metadata.pages[0].numbering_system == NUMBER_SYSTEM_ROMAN
    assert metadata.pages[0].printed_page_numeric == 1
    assert metadata.pages[2].numbering_system == NUMBER_SYSTEM_LATIN


def test_roman_numbering_only() -> None:
    pages = [
        _page(1, _body_blocks("body") + [_footer("iii")]),
        _page(2, _body_blocks("body") + [_footer("iv")]),
    ]
    metadata = extract_metadata("book.pdf", pages)
    assert metadata.numbering_system == NUMBER_SYSTEM_ROMAN
    assert metadata.pages[0].printed_page_numeric == 3


def test_bibliographic_from_cover_and_first_pages() -> None:
    cover = [
        _block(0, "Al-Mabsut", y0=100, y1=125, size=24.0),
        _block(1, "Authored by Imam Sarakhsi", y0=150, y1=165),
        _block(2, "Published by Dar al-Kutub", y0=180, y1=195),
        _block(3, "1998", y0=210, y1=225),
    ]
    metadata = extract_metadata("scan.pdf", [_page(1, cover)])
    fields = metadata.field_map
    assert fields["title"].value == "Al-Mabsut"
    assert fields["title"].source == SOURCE_COVER_PAGE
    assert fields["author"].value == "Imam Sarakhsi"
    assert fields["author"].source == SOURCE_FIRST_PAGES
    assert fields["publisher"].value == "Dar al-Kutub"
    assert fields["publication_year"].value == "1998"


def test_structural_kitab_bab_detection() -> None:
    pages = [
        _page(
            1,
            [
                _block(0, "Kitab al-Taharah", y0=100, y1=120, size=18.0),
                *_body_blocks(),
            ]
            + [_footer("1")],
        ),
        _page(
            2,
            [
                _block(0, "Bab al-Wudu", y0=100, y1=118, size=14.0),
                *_body_blocks(),
            ]
            + [_footer("2")],
        ),
        _page(3, _body_blocks() + [_footer("3")]),
    ]
    metadata = extract_metadata("book.pdf", pages)

    kitabs = [structure for structure in metadata.structures if structure.level == "kitab"]
    babs = [structure for structure in metadata.structures if structure.level == "bab"]
    assert len(kitabs) == 1
    assert kitabs[0].name == "Kitab al-Taharah"
    assert kitabs[0].page_start == 1
    assert kitabs[0].page_end == 3
    assert len(babs) == 1
    assert babs[0].name == "Bab al-Wudu"
    assert babs[0].page_start == 2
    assert babs[0].page_end == 3

    assert metadata.pages[0].kitab == "Kitab al-Taharah"
    assert metadata.pages[0].bab is None
    assert metadata.pages[1].kitab == "Kitab al-Taharah"
    assert metadata.pages[1].bab == "Bab al-Wudu"
    assert metadata.pages[2].kitab == "Kitab al-Taharah"
    assert metadata.pages[2].bab == "Bab al-Wudu"


def test_structural_source_and_confidence() -> None:
    pages = [_page(1, [_block(0, "Kitab al-Taharah", size=18.0), *_body_blocks()])]
    metadata = extract_metadata("book.pdf", pages)
    kitabs = [structure for structure in metadata.structures if structure.level == "kitab"]
    assert kitabs
    assert kitabs[0].source == SOURCE_BODY_TEXT
    assert 0.0 < kitabs[0].confidence <= 1.0


def test_missing_metadata_is_safe() -> None:
    metadata = extract_metadata("scan.pdf", [_page(1, []), _page(2, [])])
    assert metadata.field_map.get("title") is None
    assert metadata.numbering_system == NUMBER_SYSTEM_NONE
    assert metadata.pages[0].page_number_uncertain is True
    assert metadata.confidence == 0.0
    assert metadata.page_count == 2


def test_every_field_has_value_confidence_source() -> None:
    cover = [
        _block(0, "Al-Mabsut", y0=100, y1=125, size=24.0),
        _block(1, "1998", y0=210, y1=225),
    ]
    metadata = extract_metadata("Al-Mabsut ed2.pdf", [_page(1, cover + [_footer("7")])])
    assert metadata.fields
    for item in metadata.fields:
        assert item.value
        assert 0.0 < item.confidence <= 1.0
        assert item.source in (SOURCE_FILENAME, SOURCE_COVER_PAGE, SOURCE_FIRST_PAGES)


def test_garbage_footer_text_is_uncertain() -> None:
    pages = [
        _page(1, _body_blocks("text") + [_footer("***")]),
        _page(2, _body_blocks("text") + [_footer("@@@###")]),
    ]
    metadata = extract_metadata("book.pdf", pages)
    assert metadata.numbering_system == NUMBER_SYSTEM_NONE
    assert all(page.page_number_uncertain for page in metadata.pages)
    assert all(page.printed_page == "" for page in metadata.pages)
    assert metadata.pages[0].pdf_page == 1
    assert metadata.pages[1].pdf_page == 2


def test_unparseable_filename_is_safe() -> None:
    metadata = extract_metadata("!!??.pdf", [_page(1, _body_blocks("text"))])
    assert metadata.page_count == 1
    assert metadata.pages[0].pdf_page == 1
    assert metadata.confidence >= 0.0


def test_blocks_without_font_size_are_safe() -> None:
    malformed = BlockInfo(
        index=0, bbox=[72.0, 100.0, 472.0, 115.0], text="Kitab one", font="helv", size=None
    )
    metadata = extract_metadata("book.pdf", [_page(1, [malformed, *_body_blocks()])])
    assert metadata.page_count == 1
    assert metadata.numbering_system == NUMBER_SYSTEM_NONE
    assert metadata.pages[0].page_number_uncertain is True
