"""End-to-end integration tests for RAGents."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from ragents import (
    RAGEngine,
    RAGConfig,
    LLMClient,
    SimpleAgent,
    AgentConfig,
    DecisionTreeAgent,
    create_vector_store
)
from ragents.llm.types import ModelConfig, ModelProvider
from ragents.vector_stores.base import VectorStoreConfig, VectorStoreType


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.fixture
    def temp_working_dir(self):
        """Create temporary working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def integration_rag_config(self, temp_working_dir):
        """RAG configuration for integration tests."""
        return RAGConfig(
            vector_store_type="memory",  # Use memory store for testing
            collection_name="test_integration",
            chunk_size=200,  # Smaller chunks for testing
            chunk_overlap=50,
            top_k=3,
            enable_caching=False  # Disable caching for predictable tests
        )

    @pytest.fixture
    def integration_llm_client(self, openai_model_config):
        """LLM client with mocked responses for integration testing."""
        client = LLMClient(openai_model_config)

        # Mock all client components
        client._sync_client = MagicMock()
        client._async_client = MagicMock()
        client._instructor_sync = MagicMock()
        client._instructor_async = MagicMock()

        # Configure mock responses for different types of queries
        def mock_acomplete(*args, **kwargs):
            messages = args[0] if args else kwargs.get('messages', [])
            last_message = messages[-1].content if messages else ""

            if "expand" in last_message.lower() or "related" in last_message.lower():
                # Query expansion response
                mock_response = MagicMock()
                mock_response.content = '["what is machine learning", "ML algorithms", "AI fundamentals"]'
                return mock_response
            elif "analyze" in last_message.lower() and "plan" in last_message.lower():
                # Structured thinking response
                from ragents.llm.types import StructuredThought
                return StructuredThought(
                    query_analysis="This is a question about machine learning fundamentals",
                    sources_needed=["ML basics", "algorithms", "definitions"],
                    reasoning_steps=["Define ML", "Explain key concepts", "Provide examples"],
                    final_answer="Machine learning is a subset of AI that enables computers to learn from data."
                )
            else:
                # Standard completion response
                mock_response = MagicMock()
                mock_response.content = "Machine learning is a method of data analysis that automates analytical model building."
                mock_response.model = "gpt-3.5-turbo"
                mock_response.usage = {"total_tokens": 100}
                mock_response.finish_reason = "stop"
                return mock_response

        client.acomplete = AsyncMock(side_effect=mock_acomplete)

        return client

    @pytest.fixture
    def sample_documents(self, temp_working_dir):
        """Create sample documents for testing."""
        docs_dir = temp_working_dir / "documents"
        docs_dir.mkdir(exist_ok=True)

        # Create sample text files
        docs = [
            {
                "filename": "ml_basics.txt",
                "content": """
                Machine Learning Fundamentals

                Machine learning is a subset of artificial intelligence that enables computers
                to learn and improve from data without being explicitly programmed. It involves
                algorithms that can identify patterns in data and make predictions or decisions.

                Key types of machine learning include:
                1. Supervised Learning - learning with labeled data
                2. Unsupervised Learning - finding patterns in unlabeled data
                3. Reinforcement Learning - learning through interaction and feedback
                """
            },
            {
                "filename": "python_guide.txt",
                "content": """
                Python Programming Guide

                Python is a high-level programming language known for its simplicity and readability.
                It's widely used in data science, web development, and automation.

                Key features:
                - Easy to learn and use
                - Extensive library ecosystem
                - Cross-platform compatibility
                - Strong community support

                Popular libraries for data science include NumPy, Pandas, and Scikit-learn.
                """
            },
            {
                "filename": "ai_history.txt",
                "content": """
                History of Artificial Intelligence

                Artificial Intelligence has a rich history dating back to the 1950s.
                Key milestones include:

                1950s: Alan Turing proposes the Turing Test
                1956: Dartmouth Conference coins the term "Artificial Intelligence"
                1980s: Expert systems become popular
                1990s: Machine learning gains prominence
                2010s: Deep learning revolution begins
                2020s: Large language models transform NLP
                """
            }
        ]

        file_paths = []
        for doc in docs:
            file_path = docs_dir / doc["filename"]
            file_path.write_text(doc["content"])
            file_paths.append(str(file_path))

        return file_paths

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self, integration_rag_config, integration_llm_client, sample_documents):
        """Test complete RAG workflow from document ingestion to query response."""
        # Step 1: Initialize RAG engine
        rag_engine = RAGEngine(integration_rag_config, integration_llm_client)

        # Mock the document processing components
        rag_engine.processor.process_document = AsyncMock()
        rag_engine.processor.chunk_document = AsyncMock()
        rag_engine.document_store.add_document = AsyncMock()
        rag_engine.document_store.add_chunk = AsyncMock()

        # Mock document processing to return realistic documents and chunks
        async def mock_process_document(file_path, **metadata):
            from ragents.rag.types import Document
            content = Path(file_path).read_text()
            return Document(
                id=f"doc_{Path(file_path).stem}",
                content=content,
                metadata={"source": file_path, **metadata}
            )

        async def mock_chunk_document(document):
            from ragents.rag.types import Document
            # Simple chunking: split by paragraphs
            chunks = []
            paragraphs = document.content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    chunks.append(Document(
                        id=f"{document.id}_chunk_{i}",
                        content=paragraph.strip(),
                        metadata={**document.metadata, "chunk_index": i}
                    ))
            return chunks

        rag_engine.processor.process_document.side_effect = mock_process_document
        rag_engine.processor.chunk_document.side_effect = mock_chunk_document

        # Step 2: Add documents to the system
        documents = []
        for file_path in sample_documents:
            doc = await rag_engine.add_document(file_path)
            documents.append(doc)

        assert len(documents) == 3
        assert rag_engine.processor.process_document.call_count == 3

        # Step 3: Mock retrieval to return relevant chunks
        mock_retrieval_results = [
            MagicMock(
                score=0.9,
                chunk=MagicMock(content="Machine learning is a subset of artificial intelligence..."),
                metadata={"source": "ml_basics.txt"}
            ),
            MagicMock(
                score=0.8,
                chunk=MagicMock(content="Key types of machine learning include supervised, unsupervised..."),
                metadata={"source": "ml_basics.txt"}
            )
        ]
        rag_engine.retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)

        # Step 4: Query the system
        response = await rag_engine.query("What is machine learning?")

        # Verify response structure
        assert response.query == "What is machine learning?"
        assert response.answer is not None
        assert len(response.sources) == 2
        assert response.confidence > 0
        assert response.processing_time > 0

        # Verify RAG components were called
        rag_engine.retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_with_rag_integration(self, integration_rag_config, integration_llm_client, sample_documents):
        """Test agent integration with RAG system."""
        # Step 1: Set up RAG engine
        rag_engine = RAGEngine(integration_rag_config, integration_llm_client)

        # Mock RAG components
        rag_engine.processor.process_document = AsyncMock()
        rag_engine.processor.chunk_document = AsyncMock()
        rag_engine.document_store.add_document = AsyncMock()
        rag_engine.document_store.add_chunk = AsyncMock()

        # Mock query response
        mock_rag_response = MagicMock()
        mock_rag_response.answer = "Machine learning is a method of data analysis that automates analytical model building."
        mock_rag_response.sources = []
        rag_engine.query = AsyncMock(return_value=mock_rag_response)

        # Step 2: Initialize agent with RAG
        agent_config = AgentConfig(
            name="RAGAgent",
            description="An agent with RAG capabilities",
            enable_rag=True,
            enable_memory=True,
            memory_window=5
        )

        agent = SimpleAgent(agent_config, integration_llm_client, rag_engine)

        # Step 3: Process a question that should trigger RAG
        response = await agent.process_message("What is machine learning?")

        # Verify response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify RAG was called for a question
        rag_engine.query.assert_called_once()

        # Step 4: Process a non-question that shouldn't trigger RAG
        rag_engine.query.reset_mock()
        await agent.process_message("Hello there!")

        # Verify conversation state
        assert agent.state.turn_count == 2
        assert len(agent.state.memory) == 4  # 2 user messages + 2 assistant responses

    @pytest.mark.asyncio
    async def test_decision_tree_agent_workflow(self, integration_llm_client):
        """Test decision tree agent workflow."""
        # Mock the decision tree components
        with pytest.raises(ImportError):
            # DecisionTreeAgent might not be fully implemented
            # This is a placeholder for when it's available
            from ragents.agents.decision_tree import DecisionNode

    @pytest.mark.asyncio
    async def test_vector_store_integration(self, temp_working_dir):
        """Test vector store integration with different backends."""
        # Test memory vector store
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="test_collection",
            embedding_dimension=384
        )

        vector_store = create_vector_store(config)

        # Mock embedding function
        def mock_embed(texts):
            import numpy as np
            return np.random.rand(len(texts), 384).tolist()

        # Test adding and searching documents
        documents = [
            {"id": "doc1", "content": "Machine learning content", "metadata": {}},
            {"id": "doc2", "content": "Python programming content", "metadata": {}},
        ]

        # This would require actual vector store implementation
        # For now, just verify the store was created
        assert vector_store is not None
        assert hasattr(vector_store, 'add_documents')
        assert hasattr(vector_store, 'search')

    @pytest.mark.asyncio
    async def test_multimodal_processing_workflow(self, integration_rag_config, integration_llm_client):
        """Test multimodal processing workflow (text + vision)."""
        # Enable vision processing
        integration_rag_config.enable_vision = True

        rag_engine = RAGEngine(integration_rag_config, integration_llm_client)

        # Mock multimodal processor
        rag_engine.processor.process_document = AsyncMock()

        async def mock_multimodal_process(file_path, **metadata):
            from ragents.rag.types import Document
            # Simulate processing both text and image content
            if file_path.endswith('.txt'):
                content = "Text content from document"
            else:
                content = "Image description: A diagram showing ML concepts"

            return Document(
                id=f"doc_{Path(file_path).stem}",
                content=content,
                metadata={"source": file_path, "type": "multimodal", **metadata}
            )

        rag_engine.processor.process_document.side_effect = mock_multimodal_process

        # Test processing a text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample text content")
            text_file = f.name

        try:
            doc = await rag_engine.add_document(text_file)
            assert doc.content == "Text content from document"
            assert doc.metadata["type"] == "multimodal"
        finally:
            Path(text_file).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, integration_rag_config, integration_llm_client):
        """Test error handling and system resilience."""
        rag_engine = RAGEngine(integration_rag_config, integration_llm_client)

        # Test handling of LLM API errors
        integration_llm_client.acomplete.side_effect = Exception("API Error")

        # Query should handle the error gracefully
        rag_engine.retriever.retrieve = AsyncMock(return_value=[])

        with pytest.raises(Exception):
            await rag_engine.query("Test query")

        # Reset and test with retry mechanism
        integration_llm_client.acomplete.side_effect = None
        integration_llm_client.acomplete_with_retries = AsyncMock(return_value=MagicMock(content="Recovered response"))

        # This would test retry logic if implemented in the engine

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integration_rag_config, integration_llm_client, sample_documents):
        """Test concurrent document processing and querying."""
        rag_engine = RAGEngine(integration_rag_config, integration_llm_client)

        # Mock processors for concurrent operations
        rag_engine.processor.process_document = AsyncMock()
        rag_engine.processor.chunk_document = AsyncMock(return_value=[])
        rag_engine.document_store.add_document = AsyncMock()
        rag_engine.document_store.add_chunk = AsyncMock()

        # Test concurrent document addition
        tasks = [rag_engine.add_document(doc_path) for doc_path in sample_documents]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(sample_documents)
        assert rag_engine.processor.process_document.call_count == len(sample_documents)

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid RAG configuration
        with pytest.raises(ValueError):
            RAGConfig(
                chunk_size=-1,  # Invalid negative value
                top_k=0  # Invalid zero value
            )

        # Test invalid model configuration
        with pytest.raises(ValueError):
            ModelConfig(
                provider="invalid_provider",
                model_name="test-model",
                api_key="test-key"
            )

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, integration_rag_config, integration_llm_client):
        """Test performance monitoring and metrics collection."""
        rag_engine = RAGEngine(integration_rag_config, integration_llm_client)

        # Mock retrieval and generation
        rag_engine.retriever.retrieve = AsyncMock(return_value=[])
        rag_engine._generate_answer = AsyncMock(return_value=("Test answer", None))
        rag_engine._calculate_confidence = AsyncMock(return_value=0.8)

        response = await rag_engine.query("Test query")

        # Verify timing metrics are captured
        assert response.processing_time > 0
        assert response.metadata is not None
        assert "num_sources" in response.metadata