"""
rag_pipeline.py
Orchestrates the full RAG pipeline:
  1. Retrieve relevant chunks via HybridRetriever
  2. Build a cited prompt with source references
  3. Call LLM and parse the answer with inline citations
  4. Return structured response with citation metadata
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

from document_processor import DocumentChunk
from hybrid_retriever import HybridRetriever
from llm_client import LLMClient


@dataclass
class RAGResponse:
    answer: str
    citations: List[dict] = field(default_factory=list)   # [{source, page, chunk_id, excerpt}]
    retrieved_chunks: List[Tuple[DocumentChunk, float]] = field(default_factory=list)
    query: str = ""


SYSTEM_PROMPT = """You are an expert document assistant. You answer questions ONLY based on the provided context chunks.

Rules:
1. Base your answer STRICTLY on the context provided below.
2. After each factual claim, add a citation like [1], [2], etc. referencing the context chunk number.
3. If the answer is not found in the context, say: "I couldn't find relevant information in the uploaded documents."
4. Be concise, accurate, and structured. Use bullet points when listing multiple items.
5. Never make up information not present in the context.
"""


def build_context_block(chunks: List[Tuple[DocumentChunk, float]]) -> str:
    """Format retrieved chunks as a numbered context block for the prompt."""
    lines = []
    for i, (chunk, score) in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] Source: {chunk.source} | Page: {chunk.page} | Relevance: {score:.3f}\n"
            f"{chunk.text}\n"
        )
    return "\n---\n".join(lines)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str) -> RAGResponse:
        """Full RAG pipeline: retrieve → prompt → generate → parse citations."""

        # Step 1: Hybrid retrieval
        results = self.retriever.retrieve(question, top_k=self.top_k, fetch_k=self.top_k * 4)

        if not results:
            return RAGResponse(
                answer="No relevant documents found. Please upload documents first.",
                query=question,
            )

        # Step 2: Build context
        context_block = build_context_block(results)

        # Step 3: Build prompt messages
        user_message = f"""Context:
{context_block}

Question: {question}

Provide a clear, well-cited answer based on the context above:"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Step 4: LLM generation
        answer = self.llm.generate(messages)

        # Step 5: Build citation metadata
        citations = []
        for i, (chunk, score) in enumerate(results, start=1):
            citations.append({
                "ref": f"[{i}]",
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
                "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            })

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=results,
            query=question,
        )
