"""Configuration for reranking and autocut functionality."""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .base import RerankingStrategy
from .autocut import CutoffStrategy


class RerankingConfig(BaseModel):
    """Comprehensive configuration for reranking operations."""

    # Basic reranking settings
    strategy: RerankingStrategy = RerankingStrategy.HYBRID
    top_k: int = 10
    min_similarity_threshold: float = 0.3
    max_documents: int = 100

    # Autocut settings
    enable_autocut: bool = True
    cutoff_strategy: CutoffStrategy = CutoffStrategy.ADAPTIVE_THRESHOLD
    cutoff_percentile: float = 0.6
    cutoff_std_multiplier: float = 1.0
    cutoff_gradient_threshold: float = 2.0
    cutoff_zscore_threshold: float = -1.0

    # Model configurations
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hybrid strategy weights
    fusion_weights: Dict[str, float] = Field(
        default_factory=lambda: {"semantic": 0.6, "cross_encoder": 0.4}
    )

    # LLM reranking
    llm_rerank_prompt: Optional[str] = None
    llm_max_document_length: int = 500

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    parallel_processing: bool = True
    batch_size: int = 32

    # Quality thresholds
    min_confidence_threshold: float = 0.5
    reranking_timeout_seconds: float = 30.0

    class Config:
        use_enum_values = True


class AutocutConfig(BaseModel):
    """Configuration specifically for Autocut filtering."""

    strategy: CutoffStrategy = CutoffStrategy.ADAPTIVE_THRESHOLD
    percentile: float = 0.6
    std_multiplier: float = 1.0
    gradient_threshold_factor: float = 2.0
    zscore_threshold: float = -1.0
    min_documents_to_keep: int = 1
    max_documents_to_remove: int = 50
    confidence_threshold: float = 0.5

    # Advanced settings
    enable_score_normalization: bool = True
    use_historical_data: bool = True
    adaptive_learning_rate: float = 0.1

    class Config:
        use_enum_values = True


class RerankingProfile(Enum):
    """Pre-configured reranking profiles for different use cases."""
    PRECISION_FOCUSED = "precision_focused"
    RECALL_FOCUSED = "recall_focused"
    BALANCED = "balanced"
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"


def get_profile_config(profile: RerankingProfile) -> RerankingConfig:
    """Get a pre-configured reranking configuration based on profile."""

    base_configs = {
        RerankingProfile.PRECISION_FOCUSED: {
            "strategy": RerankingStrategy.CROSS_ENCODER,
            "top_k": 5,
            "min_similarity_threshold": 0.7,
            "enable_autocut": True,
            "cutoff_strategy": CutoffStrategy.GRADIENT_CHANGE,
            "fusion_weights": {"semantic": 0.3, "cross_encoder": 0.7}
        },

        RerankingProfile.RECALL_FOCUSED: {
            "strategy": RerankingStrategy.SEMANTIC,
            "top_k": 20,
            "min_similarity_threshold": 0.2,
            "enable_autocut": False,
            "cutoff_strategy": CutoffStrategy.PERCENTILE,
            "cutoff_percentile": 0.3
        },

        RerankingProfile.BALANCED: {
            "strategy": RerankingStrategy.HYBRID,
            "top_k": 10,
            "min_similarity_threshold": 0.4,
            "enable_autocut": True,
            "cutoff_strategy": CutoffStrategy.ADAPTIVE_THRESHOLD,
            "fusion_weights": {"semantic": 0.6, "cross_encoder": 0.4}
        },

        RerankingProfile.SPEED_OPTIMIZED: {
            "strategy": RerankingStrategy.SEMANTIC,
            "top_k": 8,
            "min_similarity_threshold": 0.5,
            "enable_autocut": True,
            "cutoff_strategy": CutoffStrategy.PERCENTILE,
            "parallel_processing": True,
            "batch_size": 64
        },

        RerankingProfile.QUALITY_OPTIMIZED: {
            "strategy": RerankingStrategy.LLM_BASED,
            "top_k": 5,
            "min_similarity_threshold": 0.6,
            "enable_autocut": True,
            "cutoff_strategy": CutoffStrategy.ADAPTIVE_THRESHOLD,
            "reranking_timeout_seconds": 60.0
        }
    }

    config_dict = base_configs.get(profile, base_configs[RerankingProfile.BALANCED])
    return RerankingConfig(**config_dict)


class RerankingOptimizationConfig(BaseModel):
    """Configuration for automatic reranking optimization."""

    enable_auto_optimization: bool = False
    optimization_interval_hours: int = 24
    min_samples_for_optimization: int = 100

    # Metrics to optimize for
    target_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "precision_at_k": 0.8,
            "recall_at_k": 0.7,
            "avg_processing_time": 2.0  # seconds
        }
    )

    # Optimization bounds
    parameter_bounds: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "min_similarity_threshold": {"min": 0.1, "max": 0.8},
            "cutoff_percentile": {"min": 0.3, "max": 0.9},
            "top_k": {"min": 3, "max": 20}
        }
    )

    class Config:
        use_enum_values = True