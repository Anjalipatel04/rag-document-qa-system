"""
document_processor.py
Handles PDF and text file ingestion, cleaning, and chunking.
Uses PyMuPDF (fitz) — pure Python wheel, zero C++ build on Windows.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Optional
import fitz  # PyMuPDF


@dataclass
class DocumentChunk:
    text: str
    source: str          # filename
    page: int            # 1-based page number
    chunk_id: int        # global chunk index
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """
    Ingests PDF / plain-text files and splits them into overlapping chunks.

    Strategy:
      - Extract text page-by-page via PyMuPDF
      - Clean noise (headers, footers, excessive whitespace)
      - Sliding-window chunking with overlap for context continuity
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_length: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_files(self, file_paths: List[str]) -> List[DocumentChunk]:
        """Process a list of file paths and return all chunks."""
        all_chunks: List[DocumentChunk] = []
        chunk_id = 0
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = self._extract_pdf(path)
            elif ext in (".txt", ".md"):
                pages = self._extract_text(path)
            else:
                continue  # skip unsupported

            filename = os.path.basename(path)
            for page_num, page_text in pages:
                chunks = self._chunk_text(page_text)
                for chunk_text in chunks:
                    if len(chunk_text.strip()) < self.min_chunk_length:
                        continue
                    all_chunks.append(
                        DocumentChunk(
                            text=chunk_text.strip(),
                            source=filename,
                            page=page_num,
                            chunk_id=chunk_id,
                            metadata={"file_path": path},
                        )
                    )
                    chunk_id += 1

        return all_chunks

    # ------------------------------------------------------------------ #
    #  Extraction                                                          #
    # ------------------------------------------------------------------ #

    def _extract_pdf(self, path: str) -> List[tuple]:
        """Return list of (page_number, text) tuples."""
        pages = []
        try:
            doc = fitz.open(path)
            for i, page in enumerate(doc, start=1):
                raw = page.get_text("text")
                cleaned = self._clean_text(raw)
                if cleaned:
                    pages.append((i, cleaned))
            doc.close()
        except Exception as e:
            print(f"[DocumentProcessor] Error reading {path}: {e}")
        return pages

    def _extract_text(self, path: str) -> List[tuple]:
        """Return single 'page' for plain text files."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            cleaned = self._clean_text(content)
            return [(1, cleaned)]
        except Exception as e:
            print(f"[DocumentProcessor] Error reading {path}: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Cleaning                                                            #
    # ------------------------------------------------------------------ #

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
        text = re.sub(r"[ \t]+", " ", text)          # collapse spaces/tabs
        text = re.sub(r"\x00", "", text)             # null bytes
        text = re.sub(r"[^\x20-\x7E\n]", " ", text) # non-ASCII noise
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Chunking — sliding window                                          #
    # ------------------------------------------------------------------ #

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks
