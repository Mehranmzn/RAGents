"""Concrete implementations of reranking strategies."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..llm.client import LLMClient
from .base import Reranker, RetrievedDocument, RerankingResult, RerankingStrategy


class SemanticReranker(Reranker):
    """Semantic similarity-based reranker using sentence transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(RerankingStrategy.SEMANTIC)
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            print("Warning: sentence-transformers not installed. Semantic reranking will use fallback.")
            self._model = None

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """Rerank documents using semantic similarity."""

        start_time = time.time()

        if not documents:
            return self._create_empty_result(query, documents, start_time)

        if self._model is None:
            # Fallback to original scores with slight perturbation
            return self._fallback_rerank(query, documents, top_k, start_time)

        try:
            # Extract document texts
            doc_texts = [doc.content for doc in documents]

            # Compute embeddings
            query_embedding = self._model.encode([query])
            doc_embeddings = self._model.encode(doc_texts)

            # Calculate similarities
            similarities = np.dot(query_embedding, doc_embeddings.T).flatten()

            # Create scored documents
            scored_docs = []
            for i, doc in enumerate(documents):
                new_doc = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    similarity_score=float(similarities[i]),
                    document_id=doc.document_id,
                    source=doc.source,
                    chunk_index=doc.chunk_index
                )
                scored_docs.append(new_doc)

            # Sort by new similarity scores
            reranked_docs = sorted(scored_docs, key=lambda x: x.similarity_score, reverse=True)

            # Apply top_k limit
            if top_k:
                reranked_docs = reranked_docs[:top_k]

            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(similarities)

            result = RerankingResult(
                query=query,
                original_documents=documents,
                reranked_documents=reranked_docs,
                reranking_scores=[doc.similarity_score for doc in reranked_docs],
                strategy=self.strategy,
                confidence_score=confidence,
                metadata={
                    "model": self.model_name,
                    "embeddings_computed": len(doc_texts),
                    "avg_similarity": float(np.mean(similarities))
                },
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            self.add_to_history(result)
            return result

        except Exception as e:
            print(f"Error in semantic reranking: {e}")
            return self._fallback_rerank(query, documents, top_k, start_time)

    def _fallback_rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int],
        start_time: float
    ) -> RerankingResult:
        """Fallback reranking when model is not available."""

        # Use original similarity scores with small adjustments based on content length
        reranked_docs = []
        for doc in documents:
            # Slight boost for documents with query keywords
            query_words = query.lower().split()
            content_words = doc.content.lower().split()
            overlap = sum(1 for word in query_words if word in content_words)
            boost = overlap * 0.01

            new_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                similarity_score=min(doc.similarity_score + boost, 1.0),
                document_id=doc.document_id,
                source=doc.source,
                chunk_index=doc.chunk_index
            )
            reranked_docs.append(new_doc)

        reranked_docs.sort(key=lambda x: x.similarity_score, reverse=True)

        if top_k:
            reranked_docs = reranked_docs[:top_k]

        processing_time = time.time() - start_time

        result = RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=reranked_docs,
            reranking_scores=[doc.similarity_score for doc in reranked_docs],
            strategy=self.strategy,
            confidence_score=0.5,  # Lower confidence for fallback
            metadata={"method": "fallback", "reason": "model_unavailable"},
            processing_time=processing_time,
            timestamp=datetime.now()
        )

        self.add_to_history(result)
        return result

    def _calculate_confidence(self, similarities: np.ndarray) -> float:
        """Calculate confidence based on similarity distribution."""
        if len(similarities) < 2:
            return 0.5

        # Confidence based on spread and maximum similarity
        max_sim = np.max(similarities)
        std_sim = np.std(similarities)

        # Higher confidence for higher max similarity and greater spread
        confidence = (max_sim * 0.7) + (min(std_sim * 2, 0.3))
        return min(max(confidence, 0.0), 1.0)

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "model_name": self.model_name,
            "model_loaded": self._model is not None
        }

    def _create_empty_result(self, query: str, documents: List[RetrievedDocument], start_time: float) -> RerankingResult:
        """Create empty result for edge cases."""
        return RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=[],
            reranking_scores=[],
            strategy=self.strategy,
            confidence_score=0.0,
            metadata={"reason": "no_documents"},
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )


class CrossEncoderReranker(Reranker):
    """Cross-encoder based reranker for better relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(RerankingStrategy.CROSS_ENCODER)
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            print("Warning: sentence-transformers not installed. Cross-encoder reranking will use fallback.")
            self._model = None

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """Rerank documents using cross-encoder scoring."""

        start_time = time.time()

        if not documents:
            return self._create_empty_result(query, documents, start_time)

        if self._model is None:
            return await self._llm_fallback_rerank(query, documents, top_k, start_time)

        try:
            # Prepare query-document pairs
            pairs = [(query, doc.content) for doc in documents]

            # Get cross-encoder scores
            scores = self._model.predict(pairs)

            # Create reranked documents
            scored_docs = []
            for i, doc in enumerate(documents):
                new_doc = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    similarity_score=float(scores[i]),
                    document_id=doc.document_id,
                    source=doc.source,
                    chunk_index=doc.chunk_index
                )
                scored_docs.append(new_doc)

            # Sort by cross-encoder scores
            reranked_docs = sorted(scored_docs, key=lambda x: x.similarity_score, reverse=True)

            if top_k:
                reranked_docs = reranked_docs[:top_k]

            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(scores)

            result = RerankingResult(
                query=query,
                original_documents=documents,
                reranked_documents=reranked_docs,
                reranking_scores=[doc.similarity_score for doc in reranked_docs],
                strategy=self.strategy,
                confidence_score=confidence,
                metadata={
                    "model": self.model_name,
                    "pairs_scored": len(pairs),
                    "avg_score": float(np.mean(scores))
                },
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            self.add_to_history(result)
            return result

        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            return await self._llm_fallback_rerank(query, documents, top_k, start_time)

    async def _llm_fallback_rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int],
        start_time: float
    ) -> RerankingResult:
        """Fallback to simple lexical overlap scoring."""

        scored_docs = []
        for doc in documents:
            # Simple lexical overlap scoring
            query_words = set(query.lower().split())
            content_words = set(doc.content.lower().split())
            overlap_ratio = len(query_words.intersection(content_words)) / max(len(query_words), 1)

            new_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                similarity_score=min(overlap_ratio + doc.similarity_score * 0.5, 1.0),
                document_id=doc.document_id,
                source=doc.source,
                chunk_index=doc.chunk_index
            )
            scored_docs.append(new_doc)

        reranked_docs = sorted(scored_docs, key=lambda x: x.similarity_score, reverse=True)

        if top_k:
            reranked_docs = reranked_docs[:top_k]

        processing_time = time.time() - start_time

        result = RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=reranked_docs,
            reranking_scores=[doc.similarity_score for doc in reranked_docs],
            strategy=self.strategy,
            confidence_score=0.4,  # Lower confidence for fallback
            metadata={"method": "lexical_fallback", "reason": "model_unavailable"},
            processing_time=processing_time,
            timestamp=datetime.now()
        )

        self.add_to_history(result)
        return result

    def _calculate_confidence(self, scores: np.ndarray) -> float:
        """Calculate confidence based on score distribution."""
        if len(scores) < 2:
            return 0.5

        # Confidence based on score range and distribution
        score_range = np.max(scores) - np.min(scores)
        score_std = np.std(scores)

        # Higher confidence for greater score separation
        confidence = min(score_range * 0.5 + score_std * 0.3, 1.0)
        return max(confidence, 0.2)

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "model_name": self.model_name,
            "model_loaded": self._model is not None
        }

    def _create_empty_result(self, query: str, documents: List[RetrievedDocument], start_time: float) -> RerankingResult:
        """Create empty result for edge cases."""
        return RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=[],
            reranking_scores=[],
            strategy=self.strategy,
            confidence_score=0.0,
            metadata={"reason": "no_documents"},
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )


class HybridReranker(Reranker):
    """Hybrid reranker combining multiple strategies."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__(RerankingStrategy.HYBRID)
        self.weights = weights or {"semantic": 0.6, "cross_encoder": 0.4}
        self.semantic_reranker = SemanticReranker()
        self.cross_encoder_reranker = CrossEncoderReranker()

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """Rerank documents using hybrid approach."""

        start_time = time.time()

        if not documents:
            return self._create_empty_result(query, documents, start_time)

        try:
            # Get results from both rerankers
            semantic_result = await self.semantic_reranker.rerank(query, documents, None)
            cross_encoder_result = await self.cross_encoder_reranker.rerank(query, documents, None)

            # Combine scores using weighted fusion
            combined_scores = self._fuse_scores(semantic_result, cross_encoder_result)

            # Create final ranked documents
            final_docs = []
            for i, doc in enumerate(documents):
                new_doc = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    similarity_score=combined_scores[i],
                    document_id=doc.document_id,
                    source=doc.source,
                    chunk_index=doc.chunk_index
                )
                final_docs.append(new_doc)

            # Sort by combined scores
            reranked_docs = sorted(final_docs, key=lambda x: x.similarity_score, reverse=True)

            if top_k:
                reranked_docs = reranked_docs[:top_k]

            processing_time = time.time() - start_time
            confidence = (semantic_result.confidence_score + cross_encoder_result.confidence_score) / 2

            result = RerankingResult(
                query=query,
                original_documents=documents,
                reranked_documents=reranked_docs,
                reranking_scores=[doc.similarity_score for doc in reranked_docs],
                strategy=self.strategy,
                confidence_score=confidence,
                metadata={
                    "weights": self.weights,
                    "semantic_confidence": semantic_result.confidence_score,
                    "cross_encoder_confidence": cross_encoder_result.confidence_score,
                    "fusion_method": "weighted_average"
                },
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            self.add_to_history(result)
            return result

        except Exception as e:
            print(f"Error in hybrid reranking: {e}")
            # Fallback to semantic reranking only
            return await self.semantic_reranker.rerank(query, documents, top_k)

    def _fuse_scores(
        self,
        semantic_result: RerankingResult,
        cross_encoder_result: RerankingResult
    ) -> List[float]:
        """Fuse scores from multiple rerankers."""

        # Create mapping from document content to scores
        semantic_scores = {doc.content: doc.similarity_score for doc in semantic_result.reranked_documents}
        cross_scores = {doc.content: doc.similarity_score for doc in cross_encoder_result.reranked_documents}

        # Combine scores for original documents
        combined_scores = []
        for doc in semantic_result.original_documents:
            semantic_score = semantic_scores.get(doc.content, doc.similarity_score)
            cross_score = cross_scores.get(doc.content, doc.similarity_score)

            # Weighted combination
            combined_score = (
                semantic_score * self.weights.get("semantic", 0.5) +
                cross_score * self.weights.get("cross_encoder", 0.5)
            )
            combined_scores.append(combined_score)

        return combined_scores

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "weights": self.weights,
            "semantic_config": self.semantic_reranker.get_config(),
            "cross_encoder_config": self.cross_encoder_reranker.get_config()
        }

    def _create_empty_result(self, query: str, documents: List[RetrievedDocument], start_time: float) -> RerankingResult:
        """Create empty result for edge cases."""
        return RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=[],
            reranking_scores=[],
            strategy=self.strategy,
            confidence_score=0.0,
            metadata={"reason": "no_documents"},
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )


class LLMReranker(Reranker):
    """LLM-based reranker for sophisticated relevance assessment."""

    def __init__(self, llm_client: LLMClient, prompt_template: Optional[str] = None):
        super().__init__(RerankingStrategy.LLM_BASED)
        self.llm_client = llm_client
        self.prompt_template = prompt_template or self._default_prompt_template()

    def _default_prompt_template(self) -> str:
        """Default prompt template for LLM reranking."""
        return """Evaluate the relevance of each document to the given query.

Query: {query}

Documents:
{documents}

For each document, provide a relevance score from 0.0 to 1.0 where:
- 1.0 = Highly relevant, directly answers the query
- 0.7-0.9 = Relevant, contains useful information
- 0.4-0.6 = Somewhat relevant, tangentially related
- 0.0-0.3 = Not relevant or irrelevant

Respond with just the scores in order, separated by commas (e.g., 0.9, 0.7, 0.3, 0.8):"""

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """Rerank documents using LLM assessment."""

        start_time = time.time()

        if not documents:
            return self._create_empty_result(query, documents, start_time)

        try:
            # Prepare documents for LLM
            doc_texts = []
            for i, doc in enumerate(documents):
                # Truncate long documents
                content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                doc_texts.append(f"Document {i+1}: {content}")

            documents_text = "\n\n".join(doc_texts)

            # Format prompt
            prompt = self.prompt_template.format(
                query=query,
                documents=documents_text
            )

            # Get LLM response
            response = await self.llm_client.complete(prompt)

            # Parse scores
            scores = self._parse_scores(response, len(documents))

            # Create reranked documents
            scored_docs = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                new_doc = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    similarity_score=score,
                    document_id=doc.document_id,
                    source=doc.source,
                    chunk_index=doc.chunk_index
                )
                scored_docs.append(new_doc)

            # Sort by LLM scores
            reranked_docs = sorted(scored_docs, key=lambda x: x.similarity_score, reverse=True)

            if top_k:
                reranked_docs = reranked_docs[:top_k]

            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(scores)

            result = RerankingResult(
                query=query,
                original_documents=documents,
                reranked_documents=reranked_docs,
                reranking_scores=[doc.similarity_score for doc in reranked_docs],
                strategy=self.strategy,
                confidence_score=confidence,
                metadata={
                    "llm_model": str(self.llm_client.config.provider),
                    "documents_evaluated": len(documents),
                    "llm_response_length": len(response),
                    "avg_score": sum(scores) / len(scores) if scores else 0
                },
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            self.add_to_history(result)
            return result

        except Exception as e:
            print(f"Error in LLM reranking: {e}")
            return self._fallback_rerank(query, documents, top_k, start_time)

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse scores from LLM response."""
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
            scores = [float(num) for num in numbers[:expected_count]]

            # Pad with default scores if needed
            while len(scores) < expected_count:
                scores.append(0.5)

            # Ensure scores are in valid range
            scores = [max(0.0, min(1.0, score)) for score in scores]

            return scores

        except Exception:
            # Fallback to default scores
            return [0.5] * expected_count

    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score distribution."""
        if not scores:
            return 0.0

        # Confidence based on score variance and range
        score_variance = np.var(scores) if len(scores) > 1 else 0
        score_range = max(scores) - min(scores) if len(scores) > 1 else 0

        # Higher confidence for greater score separation
        confidence = min(score_range * 0.8 + score_variance * 0.2, 1.0)
        return max(confidence, 0.3)

    def _fallback_rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int],
        start_time: float
    ) -> RerankingResult:
        """Fallback reranking when LLM fails."""

        # Use original scores
        reranked_docs = sorted(documents, key=lambda x: x.similarity_score, reverse=True)

        if top_k:
            reranked_docs = reranked_docs[:top_k]

        processing_time = time.time() - start_time

        result = RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=reranked_docs,
            reranking_scores=[doc.similarity_score for doc in reranked_docs],
            strategy=self.strategy,
            confidence_score=0.3,  # Lower confidence for fallback
            metadata={"method": "fallback", "reason": "llm_error"},
            processing_time=processing_time,
            timestamp=datetime.now()
        )

        self.add_to_history(result)
        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "llm_provider": str(self.llm_client.config.provider),
            "prompt_template_length": len(self.prompt_template)
        }

    def _create_empty_result(self, query: str, documents: List[RetrievedDocument], start_time: float) -> RerankingResult:
        """Create empty result for edge cases."""
        return RerankingResult(
            query=query,
            original_documents=documents,
            reranked_documents=[],
            reranking_scores=[],
            strategy=self.strategy,
            confidence_score=0.0,
            metadata={"reason": "no_documents"},
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )