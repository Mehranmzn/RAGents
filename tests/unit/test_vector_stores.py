"""Unit tests for vector store components."""

import pytest
from unittest.mock import MagicMock, patch

from ragents.vector_stores.factory import (
    create_vector_store,
    get_available_backends,
    recommend_backend,
    create_vector_store_from_url
)
from ragents.vector_stores.base import VectorStoreConfig, VectorStoreType


class TestVectorStoreFactory:
    """Test cases for vector store factory."""

    def test_create_vector_store_memory(self):
        """Test creating memory vector store."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="test_collection"
        )

        store = create_vector_store(config)

        assert store is not None
        assert hasattr(store, 'add_documents')
        assert hasattr(store, 'search')

    def test_create_vector_store_chromadb(self):
        """Test creating ChromaDB vector store."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            collection_name="test_collection"
        )

        store = create_vector_store(config)

        assert store is not None

    def test_create_vector_store_invalid_type(self):
        """Test creating vector store with invalid type."""
        config = VectorStoreConfig(
            store_type="invalid_type",
            collection_name="test_collection"
        )

        with pytest.raises(ValueError, match="Unsupported vector store type"):
            create_vector_store(config)

    @patch('ragents.vector_stores.factory.chromadb')
    def test_get_available_backends_with_chromadb(self, mock_chromadb):
        """Test getting available backends when ChromaDB is installed."""
        backends = get_available_backends()

        assert "memory" in backends
        assert "chromadb" in backends

    @patch('ragents.vector_stores.factory.chromadb', side_effect=ImportError)
    def test_get_available_backends_without_chromadb(self, mock_chromadb):
        """Test getting available backends when ChromaDB is not installed."""
        backends = get_available_backends()

        assert "memory" in backends
        assert "chromadb" not in backends

    @patch('ragents.vector_stores.factory.weaviate')
    def test_get_available_backends_with_weaviate(self, mock_weaviate):
        """Test getting available backends when Weaviate is installed."""
        backends = get_available_backends()

        assert "weaviate" in backends

    @patch('ragents.vector_stores.factory.asyncpg')
    def test_get_available_backends_with_pgvector(self, mock_asyncpg):
        """Test getting available backends when pgvector is installed."""
        backends = get_available_backends()

        assert "pgvector" in backends

    @patch('ragents.vector_stores.factory.elasticsearch')
    def test_get_available_backends_with_elasticsearch(self, mock_elasticsearch):
        """Test getting available backends when Elasticsearch is installed."""
        backends = get_available_backends()

        assert "elasticsearch" in backends

    def test_recommend_backend_small_general(self):
        """Test backend recommendation for small general use case."""
        recommendation = recommend_backend(
            data_size="small",
            use_case="general",
            performance_priority="balanced"
        )

        assert recommendation == "chromadb"

    def test_recommend_backend_development(self):
        """Test backend recommendation for development use case."""
        recommendation = recommend_backend(
            data_size="small",
            use_case="development",
            performance_priority="balanced"
        )

        assert recommendation == "memory"

    def test_recommend_backend_large_production(self):
        """Test backend recommendation for large production use case."""
        recommendation = recommend_backend(
            data_size="large",
            use_case="production",
            performance_priority="performance"
        )

        assert recommendation == "elasticsearch"

    def test_recommend_backend_enterprise(self):
        """Test backend recommendation for enterprise use case."""
        recommendation = recommend_backend(
            data_size="large",
            use_case="enterprise",
            performance_priority="reliability"
        )

        assert recommendation == "pgvector"

    @patch('ragents.vector_stores.factory.get_available_backends')
    def test_recommend_backend_fallback(self, mock_get_backends):
        """Test backend recommendation fallback."""
        mock_get_backends.return_value = ["memory"]

        recommendation = recommend_backend(
            data_size="unknown",
            use_case="unknown",
            performance_priority="unknown"
        )

        assert recommendation == "memory"

    def test_create_vector_store_from_postgresql_url(self):
        """Test creating vector store from PostgreSQL URL."""
        url = "postgresql://user:pass@localhost:5432/db"

        store = create_vector_store_from_url(url)

        assert store is not None

    def test_create_vector_store_from_elasticsearch_url(self):
        """Test creating vector store from Elasticsearch URL."""
        url = "http://localhost:9200"

        store = create_vector_store_from_url(url)

        assert store is not None

    def test_create_vector_store_from_weaviate_url(self):
        """Test creating vector store from Weaviate URL."""
        url = "http://localhost:8080"

        store = create_vector_store_from_url(url)

        assert store is not None

    def test_create_vector_store_from_chroma_url(self):
        """Test creating vector store from ChromaDB URL."""
        url = "chroma:///path/to/chroma"

        store = create_vector_store_from_url(url)

        assert store is not None

    def test_create_vector_store_from_file_url(self):
        """Test creating vector store from file URL."""
        url = "file:///path/to/chroma"

        store = create_vector_store_from_url(url)

        assert store is not None

    def test_create_vector_store_from_invalid_url(self):
        """Test creating vector store from invalid URL."""
        url = "invalid://example.com"

        with pytest.raises(ValueError, match="Cannot determine vector store type"):
            create_vector_store_from_url(url)


class TestVectorStoreConfig:
    """Test cases for VectorStoreConfig."""

    def test_vector_store_config_defaults(self):
        """Test vector store configuration with defaults."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="test"
        )

        assert config.store_type == VectorStoreType.MEMORY
        assert config.collection_name == "test"
        assert config.embedding_dimension == 384  # Default for sentence-transformers
        assert config.distance_metric == "cosine"

    def test_vector_store_config_custom_values(self):
        """Test vector store configuration with custom values."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            collection_name="custom_collection",
            embedding_dimension=768,
            distance_metric="euclidean",
            persist_directory="/custom/path"
        )

        assert config.store_type == VectorStoreType.CHROMADB
        assert config.collection_name == "custom_collection"
        assert config.embedding_dimension == 768
        assert config.distance_metric == "euclidean"
        assert config.persist_directory == "/custom/path"

    def test_vector_store_config_database_settings(self):
        """Test vector store configuration with database settings."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            collection_name="test",
            database_url="postgresql://localhost/test",
            table_name="embeddings",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="testdb"
        )

        assert config.database_url == "postgresql://localhost/test"
        assert config.table_name == "embeddings"
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.username == "user"
        assert config.password == "pass"
        assert config.database == "testdb"