"""Retriever that routes queries according to contextual directives."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..config.rag_config import RAGConfig
from .document_store import DocumentStore
from .types import QueryContext, RetrievalResult


class Retriever:
    """Context-aware retriever supporting semantic, hybrid, and graph lookups."""

    def __init__(
        self,
        config: RAGConfig,
        document_store: DocumentStore,
    ):
        self.config = config
        self.document_store = document_store

    async def retrieve(self, context: QueryContext) -> List[RetrievalResult]:
        """Retrieve documents according to the supplied context."""
        strategy = (context.retrieval_strategy or self.config.retrieval_strategy or "semantic").lower()
        top_k = context.top_k or self.config.top_k
        filters = dict(context.filters or {})

        graph_query = filters.pop("_graph_query", None)
        entry_points = self._normalize_entry_points(filters.pop("_graph_entry_points", []))
        hybrid_alpha = filters.pop("_hybrid_alpha", None)

        query_text = context.original_query
        query_embedding = await self.document_store.embed_text(query_text)

        results = await self._execute_strategy(
            strategy=strategy,
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            score_threshold=context.similarity_threshold,
            graph_query=graph_query,
            entry_points=entry_points,
            hybrid_alpha=hybrid_alpha,
        )

        if len(results) < top_k and context.expanded_queries:
            results = await self._augment_with_expansions(
                results,
                context.expanded_queries,
                strategy,
                top_k,
                filters,
                context.similarity_threshold,
                graph_query,
                entry_points,
                hybrid_alpha,
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    async def _execute_strategy(
        self,
        strategy: str,
        query_text: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: Dict[str, object],
        score_threshold: Optional[float],
        graph_query: Optional[str],
        entry_points: Iterable[str],
        hybrid_alpha: Optional[float],
    ) -> List[RetrievalResult]:
        alpha = float(hybrid_alpha) if hybrid_alpha is not None else 0.6
        normalized = strategy.lower()

        if normalized in {"hybrid", "graph_hybrid"}:
            return await self.document_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                alpha=alpha,
                filters=filters,
                score_threshold=score_threshold,
            )

        if normalized == "graph" or (normalized == "graph_hybrid"):
            return await self.document_store.graph_search(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                graph_query=graph_query,
                entry_points=entry_points,
                filters=filters,
                alpha=alpha,
                score_threshold=score_threshold,
            )

        if normalized == "keyword":
            return await self.document_store.keyword_search(
                query_text=query_text,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold,
            )

        if normalized == "mmr":
            # Approximate MMR by using hybrid search with a heavier lexical weight.
            return await self.document_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                alpha=0.5,
                filters=filters,
                score_threshold=score_threshold,
            )

        # Default to semantic search
        return await self.document_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )

    async def _augment_with_expansions(
        self,
        seed_results: List[RetrievalResult],
        expansions: Iterable[str],
        strategy: str,
        top_k: int,
        filters: Dict[str, object],
        score_threshold: Optional[float],
        graph_query: Optional[str],
        entry_points: Iterable[str],
        hybrid_alpha: Optional[float],
    ) -> List[RetrievalResult]:
        collected: List[RetrievalResult] = list(seed_results)
        seen: Set[str] = {res.chunk.id for res in seed_results if res.chunk and res.chunk.id}

        for variant in expansions:
            if not variant or not variant.strip():
                continue
            variant_embedding = await self.document_store.embed_text(variant)
            expanded_results = await self._execute_strategy(
                strategy=strategy,
                query_text=variant,
                query_embedding=variant_embedding,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold,
                graph_query=graph_query,
                entry_points=entry_points,
                hybrid_alpha=hybrid_alpha,
            )
            for result in expanded_results:
                chunk_id = result.chunk.id if result.chunk else None
                if chunk_id and chunk_id not in seen:
                    collected.append(result)
                    seen.add(chunk_id)
                if len(collected) >= top_k * 2:
                    break
            if len(collected) >= top_k * 2:
                break

        return collected

    @staticmethod
    def _normalize_entry_points(raw: object) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            value = raw.strip()
            return [value] if value else []
        if isinstance(raw, (list, tuple, set)):
            normalized: List[str] = []
            for item in raw:
                if isinstance(item, str) and item.strip():
                    normalized.append(item.strip())
            return normalized
        return []
