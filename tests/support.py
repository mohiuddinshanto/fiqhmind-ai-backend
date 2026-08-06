"""Shared helpers for the backend test-suite (not collected as tests)."""

import base64
import hashlib
from pathlib import Path

import fitz  # type: ignore[import-not-found]

_1X1_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

_TEXT_BLOCKS = [
    (72, 100, 16, "Bismillah ar-Rahman ar-Rahim"),
    (72, 140, 12, "Chapter one: purity and its rulings"),
    (72, 170, 12, "Water is pure in itself and purifying for others."),
    (72, 200, 12, "A small amount of impurity requires washing before prayer."),
]


def build_text_pdf(path: Path, *, pages: int = 2) -> Path:
    """Build a born-digital PDF with a real text layer on every page."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for x, y, size, text in _TEXT_BLOCKS:
            page.insert_text((x, y), text, fontsize=size)
    doc.save(str(path))
    doc.close()
    return path


def build_scanned_pdf(path: Path, *, pages: int = 1) -> Path:
    """Build a scan-like PDF with image-only pages (no text layer)."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(0, 0, 595, 842), stream=_1X1_PNG)
    doc.save(str(path))
    doc.close()
    return path


def build_image_pdf(path: Path, *, pages: int = 1) -> Path:
    """Build a PDF mixing a text layer with an embedded image and a drawing."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for x, y, size, text in _TEXT_BLOCKS:
            page.insert_text((x, y), text, fontsize=size)
        page.insert_image(fitz.Rect(72, 300, 172, 400), stream=_1X1_PNG)
        page.draw_rect(fitz.Rect(300, 300, 500, 350), color=(0, 0, 0), width=2)
    doc.save(str(path))
    doc.close()
    return path


def build_rotated_pdf(path: Path, *, rotation: int = 90) -> Path:
    """Build a PDF whose page is rotated within the document."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for x, y, size, text in _TEXT_BLOCKS:
        page.insert_text((x, y), text, fontsize=size)
    page.set_rotation(rotation)
    doc.save(str(path))
    doc.close()
    return path


def build_encrypted_pdf(path: Path, *, password: str = "secret") -> Path:
    """Build an AES-256 encrypted PDF with a user password."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Confidential", fontsize=12)
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    doc.close()
    return path


def build_double_column_pdf(path: Path, *, pages: int = 1) -> Path:
    """Build a two-column page.

    PyMuPDF merges spans sharing a baseline into one block, so the two columns
    use staggered baselines to keep every line its own block.
    """
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for index in range(4):
            page.insert_text((60, 120 + index * 40), f"Left column line {index}", fontsize=11)
            page.insert_text((320, 140 + index * 40), f"Right column line {index}", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def build_header_footer_pdf(path: Path, *, pages: int = 1) -> Path:
    """Build a page with a running header, body blocks, and a page-number footer."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((250, 30), "Kitab al-Taharah", fontsize=10)
        for index in range(4):
            page.insert_text((72, 120 + index * 40), f"Body line {index}", fontsize=12)
        page.insert_text((290, 815), "7", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def build_footnote_pdf(path: Path, *, pages: int = 1) -> Path:
    """Build a page with main text, a footnote rule, and small footnote text."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for index in range(4):
            page.insert_text((72, 120 + index * 40), f"Main text line {index}", fontsize=12)
        page.draw_line(fitz.Point(72, 650), fitz.Point(420, 650), color=(0, 0, 0), width=1)
        page.insert_text((72, 680), "Footnote one: a gloss on the main text.", fontsize=9)
        page.insert_text((72, 700), "Footnote two: another textual gloss.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def build_margin_pdf(path: Path, *, pages: int = 1) -> Path:
    """Build a page with a main column and a narrow marginal gloss column."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for index in range(4):
            page.insert_text(
                (72, 150 + index * 40),
                f"Main column text line {index} with a reasonable amount of words here",
                fontsize=12,
            )
        page.insert_text((12, 300), "hashiya gloss one", fontsize=9)
        page.insert_text((12, 320), "hashiya gloss two", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def build_malformed_pdf(path: Path) -> Path:
    """Write bytes that cannot be opened as a PDF."""
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf body at all")
    return path


def make_pdf_bytes(extra: bytes = b"", *, include_eof: bool = True) -> bytes:
    """Return minimal valid PDF bytes passing magic-byte + EOF validation."""
    body = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n"
    if include_eof:
        body += b"%%EOF\n"
    return body + extra


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
