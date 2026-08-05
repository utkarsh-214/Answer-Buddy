"""
pdf_parser.py
-------------
Parses PDF documents page-by-page and splits them into overlapping text chunks.
Each chunk retains metadata: document name and originating page number.
"""

import os
from dataclasses import dataclass
from typing import Generator

import fitz  # PyMuPDF

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """A single text chunk extracted from a PDF."""
    text: str
    doc_name: str   # e.g. "employee_handbook.pdf"
    page_number: int  # 1-indexed


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a string into overlapping chunks of at most `chunk_size` characters.
    Adjacent chunks share `overlap` characters to preserve context across splits.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def parse_pdf(pdf_path: str) -> list[Chunk]:
    """
    Opens a PDF and returns a flat list of Chunk objects.
    Each page's text is split into overlapping chunks; every chunk carries
    the source document name and 1-indexed page number.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of Chunk objects.
    """
    doc_name = os.path.basename(pdf_path)
    chunks: list[Chunk] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1  # convert to 1-indexed

        # Extract plain text; preserve whitespace layout
        raw_text = page.get_text("text")

        if not raw_text.strip():
            continue  # skip blank pages

        page_chunks = _split_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_text in page_chunks:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_name=doc_name,
                    page_number=page_number,
                )
            )

    doc.close()
    return chunks


def parse_all_pdfs(pdf_dir: str) -> list[Chunk]:
    """
    Scans `pdf_dir` for all .pdf files and parses each one.

    Args:
        pdf_dir: Directory containing PDF files.

    Returns:
        Flat list of Chunk objects from all PDFs.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If no PDF files are found in the directory.
    """
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: '{pdf_dir}'")

    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in '{pdf_dir}'. "
            "Please add your PDFs to the pdfs/ folder and try again."
        )

    all_chunks: list[Chunk] = []

    for pdf_path in sorted(pdf_files):
        print(f"  📄 Parsing: {os.path.basename(pdf_path)}")
        file_chunks = parse_pdf(pdf_path)
        print(f"     → {len(file_chunks)} chunks extracted")
        all_chunks.extend(file_chunks)

    return all_chunks
