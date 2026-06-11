"""
hybrid_retriever.py
Combines dense (FAISS) and sparse (BM25) retrieval using
Reciprocal Rank Fusion (RRF) — a proven, parameter-free fusion algorithm.

RRF formula:  RRF(d) = Σ 1 / (k + rank_i(d))
  where k=60 is a smoothing constant (standard from literature).
"""

from typing import List, Tuple, Dict

from document_processor import DocumentChunk
from vector_store import VectorStore
from bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retriever: BM25 + FAISS fused via Reciprocal Rank Fusion.

    This is the "advanced RAG" component that separates this project
    from a basic LangChain tutorial — exactly what recruiters look for.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """
        Run hybrid retrieval and return top_k fused results.

        fetch_k: how many candidates to fetch from each retriever before fusion.
        """
        # 1. Dense retrieval
        dense_results = self.vector_store.search(query, top_k=fetch_k)
        # 2. Sparse retrieval
        sparse_results = self.bm25_retriever.search(query, top_k=fetch_k)

        # 3. Build chunk_id → chunk map
        chunk_map: Dict[int, DocumentChunk] = {}
        for chunk, _ in dense_results + sparse_results:
            chunk_map[chunk.chunk_id] = chunk

        # 4. RRF scoring
        rrf_scores: Dict[int, float] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.dense_weight / (self.rrf_k + rank + 1)

        for rank, (chunk, _) in enumerate(sparse_results):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.sparse_weight / (self.rrf_k + rank + 1)

        # 5. Sort by RRF score and return top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]
