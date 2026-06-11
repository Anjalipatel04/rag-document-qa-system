"""
vector_store.py
Builds and queries a FAISS flat index using sentence-transformers embeddings.
faiss-cpu wheel installs without any CUDA/GPU requirement on Windows.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from document_processor import DocumentChunk


class VectorStore:
    """
    Dense retriever backed by FAISS IndexFlatIP (inner-product / cosine similarity).

    Design decisions for Windows i5 compatibility:
      - faiss-cpu: pure CPU FAISS, no CUDA, single pip install
      - all-MiniLM-L6-v2: 80 MB model, fast on CPU, great quality
      - Embeddings computed in batches to avoid OOM on low RAM
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self.dimension: int = 384  # MiniLM output dim

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def build(self, chunks: List[DocumentChunk], batch_size: int = 64) -> None:
        """Embed all chunks and build the FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        print(f"[VectorStore] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.chunks = chunks

        texts = [c.text for c in chunks]
        print(f"[VectorStore] Embedding {len(texts)} chunks in batches of {batch_size}...")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings).astype("float32")
        self.dimension = embeddings.shape[1]

        # IndexFlatIP with normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"[VectorStore] Index built with {self.index.ntotal} vectors (dim={self.dimension}).")

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Return top-k (chunk, score) pairs for a query."""
        if self.model is None or self.index is None:
            raise RuntimeError("VectorStore not built. Call build() first.")

        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump({"model_name": self.model_name, "dimension": self.dimension}, f)
        print(f"[VectorStore] Saved index to {directory}")

    def load(self, directory: str) -> None:
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(directory, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.model_name = meta["model_name"]
        self.dimension = meta["dimension"]
        self.model = SentenceTransformer(self.model_name)
        print(f"[VectorStore] Loaded index from {directory} ({len(self.chunks)} chunks).")
