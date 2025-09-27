"""Base classes for reranking and relevance filtering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from pydantic import BaseModel, Field


class RerankingStrategy(Enum):
    """Available reranking strategies."""
    SEMANTIC = "semantic"
    CROSS_ENCODER = "cross_encoder"
    HYBRID = "hybrid"
    LLM_BASED = "llm_based"
    SCORE_FUSION = "score_fusion"


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    document_id: Optional[str] = None
    source: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class RerankingResult:
    """Result of a reranking operation."""
    query: str
    original_documents: List[RetrievedDocument]
    reranked_documents: List[RetrievedDocument]
    reranking_scores: List[float]
    strategy: RerankingStrategy
    confidence_score: float
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: datetime


class Reranker(ABC):
    """Abstract base class for document rerankers."""

    def __init__(self, strategy: RerankingStrategy):
        self.strategy = strategy
        self.reranking_history: List[RerankingResult] = []

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """Rerank documents based on relevance to the query."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration for this reranker."""
        pass

    def add_to_history(self, result: RerankingResult):
        """Add a reranking result to the history."""
        self.reranking_history.append(result)

    def get_performance_metrics(self, limit: int = 10) -> Dict[str, float]:
        """Get recent performance metrics."""
        if not self.reranking_history:
            return {}

        recent_results = self.reranking_history[-limit:]

        return {
            "avg_confidence": sum(r.confidence_score for r in recent_results) / len(recent_results),
            "avg_processing_time": sum(r.processing_time for r in recent_results) / len(recent_results),
            "total_reranked": sum(len(r.reranked_documents) for r in recent_results),
            "avg_rerank_ratio": sum(len(r.reranked_documents) / max(len(r.original_documents), 1)
                                  for r in recent_results) / len(recent_results)
        }


class RerankingConfig(BaseModel):
    """Configuration for reranking operations."""

    strategy: RerankingStrategy
    top_k: int = 10
    min_similarity_threshold: float = 0.3
    enable_autocut: bool = True
    autocut_percentile: float = 0.6
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_rerank_prompt: Optional[str] = None
    fusion_weights: Dict[str, float] = Field(default_factory=lambda: {"semantic": 0.6, "cross_encoder": 0.4})
    max_documents: int = 100
    enable_caching: bool = True

    class Config:
        use_enum_values = True


class DocumentScore(BaseModel):
    """Score assigned to a document by a reranker."""

    document_id: str
    score: float
    confidence: float
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RerankingBatch(BaseModel):
    """A batch of documents for reranking."""

    query: str
    documents: List[RetrievedDocument]
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)