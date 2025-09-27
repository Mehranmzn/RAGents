"""Unit tests for configuration classes."""

import os
import tempfile
from pathlib import Path

import pytest

from ragents.config.rag_config import RAGConfig
from ragents.config.base import WorkingDirectories


class TestRAGConfig:
    """Test cases for RAGConfig."""

    def test_rag_config_defaults(self):
        """Test RAG configuration with default values."""
        config = RAGConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_dimension == 384
        assert config.vector_store_type == "chromadb"
        assert config.collection_name == "ragents_documents"
        assert config.retrieval_strategy == "similarity"
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.enable_vision is False
        assert config.query_expansion is True
        assert config.enable_caching is True

    def test_rag_config_custom_values(self):
        """Test RAG configuration with custom values."""
        config = RAGConfig(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="custom-model",
            vector_store_type="memory",
            top_k=10,
            enable_vision=True
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.embedding_model == "custom-model"
        assert config.vector_store_type == "memory"
        assert config.top_k == 10
        assert config.enable_vision is True

    def test_rag_config_from_env_empty(self):
        """Test creating RAG config from environment with no variables set."""
        # Ensure environment is clean for this test
        original_env = os.environ.copy()

        # Clear relevant environment variables
        rag_env_vars = [key for key in os.environ.keys() if key.startswith("RAGENTS_")]
        for var in rag_env_vars:
            del os.environ[var]

        try:
            config = RAGConfig.from_env()
            # Should use default values when no env vars are set
            assert config.chunk_size == 1000
            assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_rag_config_from_env_with_values(self):
        """Test creating RAG config from environment with values set."""
        env_vars = {
            "RAGENTS_CHUNK_SIZE": "750",
            "RAGENTS_CHUNK_OVERLAP": "150",
            "RAGENTS_EMBEDDING_MODEL": "custom-embedding-model",
            "RAGENTS_VECTOR_STORE_TYPE": "memory",
            "RAGENTS_TOP_K": "8",
            "RAGENTS_SIMILARITY_THRESHOLD": "0.8",
            "RAGENTS_ENABLE_VISION": "true",
            "RAGENTS_ENABLE_CACHING": "false",
            "RAGENTS_LOG_LEVEL": "DEBUG"
        }

        original_env = os.environ.copy()

        try:
            # Set environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            config = RAGConfig.from_env()

            assert config.chunk_size == 750
            assert config.chunk_overlap == 150
            assert config.embedding_model == "custom-embedding-model"
            assert config.vector_store_type == "memory"
            assert config.top_k == 8
            assert config.similarity_threshold == 0.8
            assert config.enable_vision is True
            assert config.enable_caching is False
            assert config.log_level == "DEBUG"
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_rag_config_from_env_boolean_values(self):
        """Test parsing boolean values from environment."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("invalid", False)  # Invalid values should default to False
        ]

        original_env = os.environ.copy()

        for env_value, expected in test_cases:
            try:
                os.environ["RAGENTS_ENABLE_VISION"] = env_value
                config = RAGConfig.from_env()
                assert config.enable_vision == expected, f"Failed for value: {env_value}"
            finally:
                # Clear environment
                if "RAGENTS_ENABLE_VISION" in os.environ:
                    del os.environ["RAGENTS_ENABLE_VISION"]

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_rag_config_from_env_working_dir(self):
        """Test setting working directory from environment."""
        original_env = os.environ.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.environ["RAGENTS_WORKING_DIR"] = temp_dir
                config = RAGConfig.from_env()

                assert config.working_dirs.base_dir == Path(temp_dir)
            finally:
                # Restore original environment
                os.environ.clear()
                os.environ.update(original_env)

    def test_get_vector_store_config_chromadb(self):
        """Test getting ChromaDB vector store configuration."""
        config = RAGConfig(vector_store_type="chromadb")
        store_config = config.get_vector_store_config()

        assert store_config["collection_name"] == "ragents_documents"
        assert store_config["embedding_dimension"] == 384
        assert "persist_directory" in store_config
        assert "chromadb" in store_config["persist_directory"]

    def test_get_vector_store_config_weaviate(self):
        """Test getting Weaviate vector store configuration."""
        config = RAGConfig(vector_store_type="weaviate")

        original_env = os.environ.copy()
        try:
            # Set Weaviate environment variables
            os.environ["WEAVIATE_URL"] = "http://custom-weaviate:8080"
            os.environ["WEAVIATE_API_KEY"] = "test-api-key"

            store_config = config.get_vector_store_config()

            assert store_config["collection_name"] == "ragents_documents"
            assert store_config["embedding_dimension"] == 384
            assert store_config["url"] == "http://custom-weaviate:8080"
            assert store_config["api_key"] == "test-api-key"
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_get_vector_store_config_weaviate_defaults(self):
        """Test getting Weaviate config with default values."""
        config = RAGConfig(vector_store_type="weaviate")

        original_env = os.environ.copy()
        try:
            # Clear Weaviate environment variables
            for key in ["WEAVIATE_URL", "WEAVIATE_API_KEY"]:
                if key in os.environ:
                    del os.environ[key]

            store_config = config.get_vector_store_config()

            assert store_config["url"] == "http://localhost:8080"
            assert store_config["api_key"] is None
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_get_vector_store_config_memory(self):
        """Test getting memory vector store configuration."""
        config = RAGConfig(vector_store_type="memory")
        store_config = config.get_vector_store_config()

        assert store_config["collection_name"] == "ragents_documents"
        assert store_config["embedding_dimension"] == 384
        # Memory store should only have base config
        assert "persist_directory" not in store_config
        assert "url" not in store_config

    def test_supported_formats(self):
        """Test supported file formats."""
        config = RAGConfig()

        expected_formats = [
            ".txt", ".md", ".pdf", ".docx", ".xlsx",
            ".csv", ".html", ".json"
        ]

        for fmt in expected_formats:
            assert fmt in config.supported_formats

    def test_multimodal_settings(self):
        """Test multimodal processing settings."""
        config = RAGConfig(
            enable_vision=True,
            vision_model="custom-vision-model",
            enable_table_extraction=True,
            enable_equation_processing=True,
            enable_chart_analysis=True
        )

        assert config.enable_vision is True
        assert config.vision_model == "custom-vision-model"
        assert config.enable_table_extraction is True
        assert config.enable_equation_processing is True
        assert config.enable_chart_analysis is True

    def test_cache_settings(self):
        """Test caching configuration."""
        config = RAGConfig(
            enable_caching=True,
            cache_ttl_hours=48,
            cache_max_size=2000
        )

        assert config.enable_caching is True
        assert config.cache_ttl_hours == 48
        assert config.cache_max_size == 2000

    def test_reranking_config_included(self):
        """Test that reranking configuration is included."""
        config = RAGConfig()

        assert hasattr(config, 'reranking')
        assert config.reranking is not None


class TestWorkingDirectories:
    """Test cases for WorkingDirectories."""

    def test_working_directories_defaults(self):
        """Test working directories with default values."""
        working_dirs = WorkingDirectories()

        assert working_dirs.base_dir == Path("./output")
        assert working_dirs.documents_dir == Path("./output/documents")
        assert working_dirs.cache_dir == Path("./output/cache")
        assert working_dirs.logs_dir == Path("./output/logs")
        assert working_dirs.temp_dir == Path("./output/temp")

    def test_working_directories_custom_base(self):
        """Test working directories with custom base directory."""
        custom_base = Path("/custom/base")
        working_dirs = WorkingDirectories(base_dir=custom_base)

        assert working_dirs.base_dir == custom_base
        assert working_dirs.documents_dir == custom_base / "documents"
        assert working_dirs.cache_dir == custom_base / "cache"
        assert working_dirs.logs_dir == custom_base / "logs"
        assert working_dirs.temp_dir == custom_base / "temp"

    def test_working_directories_ensure_created(self):
        """Test that working directories can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_base"
            working_dirs = WorkingDirectories(base_dir=base_path)

            # Should not exist initially
            assert not working_dirs.base_dir.exists()

            # Create directories
            working_dirs.ensure_dirs_exist()

            # Should exist after creation
            assert working_dirs.base_dir.exists()
            assert working_dirs.documents_dir.exists()
            assert working_dirs.cache_dir.exists()
            assert working_dirs.logs_dir.exists()
            assert working_dirs.temp_dir.exists()