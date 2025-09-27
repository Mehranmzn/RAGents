"""In-memory document store with lightweight vector-style retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from ..config.rag_config import RAGConfig
from .types import ContentChunk, Document, RetrievalResult


class DocumentStore:
    """Simple document store that supports semantic, hybrid, and graph-style search."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_dim = config.embedding_dimension or 384
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, ContentChunk] = {}
        self.chunk_embeddings: Dict[str, List[float]] = {}
        self.chunk_terms: Dict[str, List[str]] = {}
        self.chunk_ids_by_document: Dict[str, set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def add_document(self, document: Document) -> Document:
        """Store a document without chunking."""
        async with self._lock:
            self.documents[document.id] = document
        return document

    async def add_chunk(self, chunk: ContentChunk) -> ContentChunk:
        """Store a chunk and maintain derived indices."""
        async with self._lock:
            chunk_id = chunk.id or str(uuid4())
            chunk.id = chunk_id
            self.chunks[chunk_id] = chunk
            self.chunk_ids_by_document[chunk.document_id].add(chunk_id)

            embedding = chunk.embedding or self._embed_text(chunk.content)
            chunk.embedding = embedding
            self.chunk_embeddings[chunk_id] = embedding
            self.chunk_terms[chunk_id] = self._tokenize(chunk.content)
        return chunk

    async def remove_document(self, document_id: str) -> bool:
        """Remove a document and all of its chunks."""
        async with self._lock:
            removed = self.documents.pop(document_id, None) is not None
            chunk_ids = self.chunk_ids_by_document.pop(document_id, set())
            for chunk_id in chunk_ids:
                self.chunks.pop(chunk_id, None)
                self.chunk_embeddings.pop(chunk_id, None)
                self.chunk_terms.pop(chunk_id, None)
        return removed

    async def clear(self) -> None:
        """Clear all stored documents and chunks."""
        async with self._lock:
            self.documents.clear()
            self.chunks.clear()
            self.chunk_embeddings.clear()
            self.chunk_terms.clear()
            self.chunk_ids_by_document.clear()

    async def get_stats(self) -> Dict[str, int]:
        """Return simple statistics about the store."""
        async with self._lock:
            return {
                "document_count": len(self.documents),
                "chunk_count": len(self.chunks),
                "embedding_dimension": self.embedding_dim,
            }

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, object]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Search using cosine similarity."""
        candidates = await self._collect_candidates(query_embedding, filters or {})
        if score_threshold is not None:
            candidates = [c for c in candidates if c[1] >= score_threshold]
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]
        return self._build_results(ranked, strategy="semantic")

    async def hybrid_search(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        top_k: int,
        alpha: float,
        filters: Optional[Dict[str, object]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Combine semantic and lexical evidence."""
        filters = filters or {}
        candidates = await self._collect_candidates(query_embedding, filters, multiplier=2)
        query_terms = self._tokenize(query_text)

        scored: List[Tuple[str, float]] = []
        for chunk_id, semantic_score in candidates:
            lexical = self._lexical_overlap(query_terms, self.chunk_terms.get(chunk_id, []))
            combined = (alpha * semantic_score) + ((1 - alpha) * lexical)
            if score_threshold is not None and combined < score_threshold:
                continue
            scored.append((chunk_id, combined))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return self._build_results(ranked, strategy="hybrid")

    async def keyword_search(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, object]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Pure lexical search using token overlap."""
        filters = filters or {}
        query_terms = self._tokenize(query_text)
        scored: List[Tuple[str, float]] = []

        async with self._lock:
            for chunk_id, chunk in self.chunks.items():
                if not self._matches_filters(chunk, filters):
                    continue
                overlap = self._lexical_overlap(query_terms, self.chunk_terms.get(chunk_id, []))
                if score_threshold is not None and overlap < score_threshold:
                    continue
                if overlap > 0:
                    scored.append((chunk_id, overlap))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return self._build_results(ranked, strategy="keyword")

    async def graph_search(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        top_k: int,
        graph_query: Optional[str],
        entry_points: Iterable[str],
        filters: Optional[Dict[str, object]] = None,
        alpha: float = 0.6,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Search with graph-style boosts based on entry points."""
        filters = filters or {}
        base_candidates = await self._collect_candidates(query_embedding, filters, multiplier=2)
        query_terms = self._tokenize(query_text)
        entry_terms = [t.lower() for t in entry_points or [] if isinstance(t, str)]
        graph_term = graph_query.lower() if graph_query else None

        scored: List[Tuple[str, float]] = []
        for chunk_id, semantic_score in base_candidates:
            terms = self.chunk_terms.get(chunk_id, [])
            lexical = self._lexical_overlap(query_terms, terms)
            combined = (alpha * semantic_score) + ((1 - alpha) * lexical)

            chunk = self.chunks.get(chunk_id)
            hay_lower = ""
            if chunk:
                haystack = f"{chunk.content}\n" + " ".join(str(v) for v in chunk.metadata.values())
                hay_lower = haystack.lower()

            boost = 0.0
            if graph_term and graph_term in hay_lower:
                boost += 0.1
            for term in entry_terms:
                if term in hay_lower:
                    boost += 0.08

            final_score = min(combined + boost, 1.0)
            if score_threshold is not None and final_score < score_threshold:
                continue
            scored.append((chunk_id, final_score))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return self._build_results(ranked, strategy="graph")

    async def embed_text(self, text: str) -> List[float]:
        """Public helper to embed text consistently with stored chunks."""
        return self._embed_text(text)

    async def _collect_candidates(
        self,
        query_embedding: Sequence[float],
        filters: Dict[str, object],
        multiplier: int = 1,
    ) -> List[Tuple[str, float]]:
        """Collect candidate chunks scored by cosine similarity."""
        top_limit = self.config.top_k * multiplier if multiplier > 0 else self.config.top_k
        async with self._lock:
            items = list(self.chunks.items())

        scored: List[Tuple[str, float]] = []
        for chunk_id, chunk in items:
            if not self._matches_filters(chunk, filters):
                continue
            embedding = self.chunk_embeddings.get(chunk_id)
            if not embedding:
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            if score > 0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_limit]

    def _build_results(
        self, scored: Sequence[Tuple[str, float]], strategy: str
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for rank, (chunk_id, score) in enumerate(scored, 1):
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(min(max(score, 0.0), 1.0)),
                    rank=rank,
                    retrieval_method=strategy,
                )
            )
        return results

    def _matches_filters(self, chunk: ContentChunk, filters: Dict[str, object]) -> bool:
        if not filters:
            return True
        metadata = chunk.metadata or {}
        for key, expected in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != expected:
                return False
        return True

    def _embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.embedding_dim

        vector = [0.0] * self.embedding_dim
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            for i in range(0, len(digest), 4):
                idx = int.from_bytes(digest[i:i + 4], "little") % self.embedding_dim
                vector[idx] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _lexical_overlap(query_terms: Sequence[str], chunk_terms: Sequence[str]) -> float:
        if not query_terms or not chunk_terms:
            return 0.0
        query_set = set(query_terms)
        chunk_set = set(chunk_terms)
        if not chunk_set:
            return 0.0
        return len(query_set & chunk_set) / len(query_set)

    @staticmethod
    def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return max(min(dot / (norm1 * norm2), 1.0), 0.0)
