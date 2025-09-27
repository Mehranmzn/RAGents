"""RAG system configuration."""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

from .base import BaseConfig, WorkingDirectories
from ..reranking.config import RerankingConfig


class RAGConfig(BaseConfig):
    """Configuration for RAG system."""

    # Working directories
    working_dirs: WorkingDirectories = WorkingDirectories()

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 100
    supported_formats: List[str] = [
        ".txt",
        ".md",
        ".pdf",
        ".docx",
        ".xlsx",
        ".csv",
        ".html",
        ".json",
    ]

    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32

    # Vector store configuration
    vector_store_type: Literal["chromadb", "weaviate", "memory"] = "chromadb"
    vector_store_path: Optional[str] = None
    collection_name: str = "ragents_documents"

    # Retrieval configuration
    retrieval_strategy: Literal["similarity", "mmr", "hybrid"] = "similarity"
    top_k: int = 5
    similarity_threshold: float = 0.7
    mmr_diversity_threshold: float = 0.7

    # Query processing
    query_expansion: bool = True
    query_rewriting: bool = True
    context_window_size: int = 8000

    # Multimodal processing
    enable_vision: bool = False
    vision_model: str = "openai/clip-vit-base-patch32"
    image_description_prompt: str = (
        "Describe this image in detail, focusing on key elements, text, "
        "and relationships that would be useful for search and retrieval."
    )

    # Content processors
    enable_table_extraction: bool = True
    enable_equation_processing: bool = False
    enable_chart_analysis: bool = False

    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    cache_max_size: int = 1000

    # Reranking and relevance filtering
    reranking: RerankingConfig = RerankingConfig()

    # Logging
    log_level: str = "INFO"
    enable_debug_logs: bool = False

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        config_dict = {}

        # Map environment variables to config fields
        env_mappings = {
            "RAGENTS_CHUNK_SIZE": "chunk_size",
            "RAGENTS_CHUNK_OVERLAP": "chunk_overlap",
            "RAGENTS_MAX_FILE_SIZE_MB": "max_file_size_mb",
            "RAGENTS_EMBEDDING_MODEL": "embedding_model",
            "RAGENTS_EMBEDDING_DIMENSION": "embedding_dimension",
            "RAGENTS_VECTOR_STORE_TYPE": "vector_store_type",
            "RAGENTS_VECTOR_STORE_PATH": "vector_store_path",
            "RAGENTS_COLLECTION_NAME": "collection_name",
            "RAGENTS_RETRIEVAL_STRATEGY": "retrieval_strategy",
            "RAGENTS_TOP_K": "top_k",
            "RAGENTS_SIMILARITY_THRESHOLD": "similarity_threshold",
            "RAGENTS_ENABLE_VISION": "enable_vision",
            "RAGENTS_VISION_MODEL": "vision_model",
            "RAGENTS_ENABLE_CACHING": "enable_caching",
            "RAGENTS_CACHE_TTL_HOURS": "cache_ttl_hours",
            "RAGENTS_LOG_LEVEL": "log_level",
        }

        for env_var, config_field in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert string values to appropriate types
                if config_field in ["chunk_size", "chunk_overlap", "max_file_size_mb",
                                   "embedding_dimension", "top_k", "cache_ttl_hours"]:
                    value = int(value)
                elif config_field in ["similarity_threshold", "mmr_diversity_threshold"]:
                    value = float(value)
                elif config_field in ["enable_vision", "enable_caching", "query_expansion",
                                     "query_rewriting"]:
                    value = value.lower() in ("true", "1", "yes", "on")
                config_dict[config_field] = value

        # Handle working directory
        if "RAGENTS_WORKING_DIR" in os.environ:
            config_dict["working_dirs"] = WorkingDirectories(
                base_dir=Path(os.environ["RAGENTS_WORKING_DIR"])
            )

        return cls(**config_dict)

    def get_vector_store_config(self) -> Dict:
        """Get vector store specific configuration."""
        base_config = {
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
        }

        if self.vector_store_type == "chromadb":
            return {
                **base_config,
                "persist_directory": str(self.working_dirs.cache_dir / "chromadb"),
            }
        elif self.vector_store_type == "weaviate":
            return {
                **base_config,
                "url": os.environ.get("WEAVIATE_URL", "http://localhost:8080"),
                "api_key": os.environ.get("WEAVIATE_API_KEY"),
            }
        else:  # memory
            return base_config