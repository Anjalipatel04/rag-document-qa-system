"""
bm25_retriever.py
BM25 sparse retriever using the rank-bm25 library (pure Python, no compilation).
Used in hybrid search alongside FAISS dense retrieval.
"""

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from document_processor import DocumentChunk


class BM25Retriever:
    """
    Sparse keyword-based retriever.

    Why BM25 alongside dense retrieval?
      - Dense models miss exact keyword matches (product codes, names, IDs)
      - BM25 handles these perfectly
      - Combining both = Hybrid RAG = higher recall
    """

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Index built with {len(chunks)} documents.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25Retriever not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices sorted by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()
