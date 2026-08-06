"""Tests for the Phase 7 pure chunking engine (`extract_chunks`)."""

import hashlib

from app.services.chunking import PageContext, extract_chunks
from app.services.extraction import BlockInfo, PageInfo


def _block(index, text, *, y0=120.0, y1=140.0, size=12.0, x0=72.0) -> BlockInfo:
    return BlockInfo(
        index=index,
        bbox=[x0, y0, x0 + 400, y1],
        text=text,
        font="helv",
        size=size,
    )


def _page(number, blocks) -> PageInfo:
    return PageInfo(number=number, width=595.0, height=842.0, rotation=0, blocks=blocks)


def _contexts(pages) -> list[PageContext]:
    return [
        PageContext(pdf_page=page.number, printed_page=str(i), printed_page_numeric=i)
        for i, page in enumerate(pages, start=1)
    ]


def test_no_pages_returns_empty() -> None:
    assert extract_chunks([]) == []


def test_pages_without_main_blocks_returns_empty() -> None:
    page = _page(1, [_block(0, "7", y0=810, y1=820, size=10.0)])  # footer band only
    assert extract_chunks([page]) == []


def test_single_paragraph_produces_one_chunk() -> None:
    text = "Water is pure in itself and purifying for others."
    page = _page(1, [_block(0, text)])
    chunks = extract_chunks([page], _contexts([page]))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.raw_text == text
    assert chunk.normalized_text == text
    assert chunk.token_count == 9
    assert chunk.pdf_page_start == 1
    assert chunk.pdf_page_end == 1
    assert chunk.printed_page_start == 1
    assert chunk.printed_page_end == 1
    assert chunk.kitab is None
    assert chunk.bab is None
    assert chunk.context_heading is None
    assert chunk.region == "main"
    assert chunk.lang == "ar"
    assert chunk.order_index == 0


def test_chunk_id_is_sha256_of_position_and_raw_text() -> None:
    page = _page(1, [_block(0, "A single line of body text here.")])
    chunks = extract_chunks([page], _contexts([page]))
    expected = hashlib.sha256(
        f"{chunks[0].order_index}:{chunks[0].raw_text}".encode()
    ).hexdigest()
    assert chunks[0].chunk_id == expected


def test_chunks_are_deterministic() -> None:
    page = _page(1, [_block(0, "First sentence here. Second sentence here.")])
    first = extract_chunks([page], _contexts([page]))
    second = extract_chunks([page], _contexts([page]))
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.raw_text for chunk in first] == [chunk.raw_text for chunk in second]


def test_clean_page_boundaries_are_not_merged() -> None:
    p1 = _page(1, [_block(0, "Line of text ending the first page.")])
    p2 = _page(2, [_block(0, "Line of text opening the second page.")])
    chunks = extract_chunks([p1, p2], _contexts([p1, p2]))

    assert len(chunks) == 2
    assert chunks[0].pdf_page_start == 1
    assert chunks[0].pdf_page_end == 1
    assert chunks[1].pdf_page_start == 2
    assert chunks[1].pdf_page_end == 2
    assert chunks[0].printed_page_start == 1
    assert chunks[1].printed_page_start == 2


def test_mid_sentence_page_break_merges_pages() -> None:
    p1 = _page(1, [_block(0, "The sentence begins on this page and")])
    p2 = _page(2, [_block(0, "continues onto the next page.")])
    chunks = extract_chunks([p1, p2], _contexts([p1, p2]))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.pdf_page_start == 1
    assert chunk.pdf_page_end == 2
    assert chunk.printed_page_start == 1
    assert chunk.printed_page_end == 2
    assert "begins on this page and" in chunk.raw_text
    assert "continues onto the next page." in chunk.raw_text


def test_section_boundaries_split_chunks_and_prepend_context() -> None:
    p1 = _page(
        1,
        [
            _block(0, "Kitab al-Taharah", y0=100, y1=120, size=18.0),
            _block(1, "Body text one with a fair amount of words to dominate."),
        ],
    )
    p2 = _page(
        2,
        [
            _block(0, "Bab al-Wudu", y0=100, y1=118, size=14.0),
            _block(1, "Body text two with a fair amount of words to dominate."),
        ],
    )
    chunks = extract_chunks([p1, p2], _contexts([p1, p2]))

    assert len(chunks) == 2
    first, second = chunks
    assert first.kitab == "Kitab al-Taharah"
    assert first.bab is None
    assert first.context_heading == "Kitab: Kitab al-Taharah"
    assert first.raw_text == (
        "Kitab: Kitab al-Taharah\n\n"
        "Body text one with a fair amount of words to dominate."
    )
    assert first.pdf_page_start == 1

    assert second.kitab == "Kitab al-Taharah"
    assert second.bab == "Bab al-Wudu"
    assert second.context_heading == "Kitab: Kitab al-Taharah\nBab: Bab al-Wudu"
    assert second.raw_text == (
        "Kitab: Kitab al-Taharah\nBab: Bab al-Wudu\n\n"
        "Body text two with a fair amount of words to dominate."
    )
    assert second.pdf_page_start == 2


def test_oversized_segment_splits_on_sentences_with_overlap() -> None:
    sentences = [
        "Alpha beta gamma delta.",
        "Epsilon zeta eta theta.",
        "Iota kappa lambda mu.",
        "Nu xi omicron pi.",
        "Rho sigma tau upsilon.",
    ]
    page = _page(1, [_block(0, " ".join(sentences))])
    chunks = extract_chunks([page], _contexts([page]), max_tokens=10, overlap=0.2)

    assert len(chunks) == 4
    assert chunks[0].raw_text == " ".join(sentences[0:2])
    assert chunks[1].raw_text == " ".join(sentences[1:3])
    assert chunks[2].raw_text == " ".join(sentences[2:4])
    assert chunks[3].raw_text == " ".join(sentences[3:5])
    assert [chunk.order_index for chunk in chunks] == [0, 1, 2, 3]


def test_fasl_recursion_splits_oversized_bab() -> None:
    page = _page(
        1,
        [
            _block(0, "Bab al-Wudu", y0=100, y1=118, size=14.0),
            _block(1, "Fasl first", y0=140, y1=158, size=13.0),
            _block(2, "First fasl body text with several words here.", y0=160, y1=180),
            _block(3, "Fasl second", y0=200, y1=218, size=13.0),
            _block(4, "Second fasl body text with several words here.", y0=220, y1=240),
        ],
    )
    chunks = extract_chunks([page], _contexts([page]), max_tokens=12)

    assert len(chunks) == 2
    assert chunks[0].bab == "Bab al-Wudu"
    assert chunks[0].fasl == "Fasl first"
    assert chunks[0].context_heading == "Bab: Bab al-Wudu\nFasl: Fasl first"
    assert chunks[1].fasl == "Fasl second"
    assert chunks[1].context_heading == "Bab: Bab al-Wudu\nFasl: Fasl second"


def test_single_sentence_never_split() -> None:
    text = ("One long sentence with quite a lot of words " * 30) + "."
    page = _page(1, [_block(0, text)])
    chunks = extract_chunks([page], _contexts([page]), max_tokens=20)

    assert len(chunks) == 1
    assert chunks[0].token_count > 20


def test_header_and_footer_blocks_are_excluded() -> None:
    page = _page(
        1,
        [
            _block(0, "Kitab al-Taharah", y0=30, y1=40, size=10.0),
            _block(1, "Main body text of the page."),
            _block(2, "7", y0=810, y1=820, size=10.0),
        ],
    )
    chunks = extract_chunks([page], _contexts([page]))

    assert len(chunks) == 1
    assert chunks[0].raw_text == "Main body text of the page."
    assert "Kitab al-Taharah" not in chunks[0].raw_text


def test_order_is_sequential_across_pages() -> None:
    pages = [_page(i, [_block(0, f"Body line on page {i} ends here.")]) for i in (1, 2, 3)]
    chunks = extract_chunks(pages, _contexts(pages))

    assert [chunk.order_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.pdf_page_start for chunk in chunks] == [1, 2, 3]


def test_normalized_text_collapses_whitespace() -> None:
    page = _page(1, [_block(0, "First  line.   Second line.")])
    chunks = extract_chunks([page], _contexts([page]))

    assert len(chunks) == 1
    assert chunks[0].raw_text == "First  line. Second line."
    assert chunks[0].normalized_text == "First line. Second line."
