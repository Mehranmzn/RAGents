"""Integration tests for RAG pipeline components."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from ragents.rag.engine import RAGEngine
from ragents.config.rag_config import RAGConfig
from ragents.llm.client import LLMClient
from ragents.rag.types import Document, QueryContext


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline."""

    @pytest.fixture
    def pipeline_rag_config(self):
        """RAG configuration optimized for pipeline testing."""
        return RAGConfig(
            vector_store_type="memory",
            collection_name="pipeline_test",
            chunk_size=300,
            chunk_overlap=50,
            top_k=5,
            similarity_threshold=0.6,
            query_expansion=True,
            enable_caching=False
        )

    @pytest.fixture
    def mock_documents(self):
        """Mock documents for pipeline testing."""
        return [
            Document(
                id="doc1",
                content="""
                Natural Language Processing (NLP) is a field of artificial intelligence
                that focuses on the interaction between computers and human language.
                It involves developing algorithms that can understand, interpret, and
                generate human language in a valuable way.
                """,
                metadata={"title": "NLP Introduction", "category": "AI"}
            ),
            Document(
                id="doc2",
                content="""
                Machine Learning algorithms can be categorized into three main types:
                supervised learning, unsupervised learning, and reinforcement learning.
                Supervised learning uses labeled data to train models that can make
                predictions on new, unseen data.
                """,
                metadata={"title": "ML Types", "category": "Machine Learning"}
            ),
            Document(
                id="doc3",
                content="""
                Deep Learning is a subset of machine learning that uses neural networks
                with multiple layers (deep neural networks) to model and understand
                complex patterns in data. It has revolutionized fields like computer
                vision, natural language processing, and speech recognition.
                """,
                metadata={"title": "Deep Learning", "category": "AI"}
            )
        ]

    @pytest.mark.asyncio
    async def test_document_ingestion_pipeline(self, pipeline_rag_config, mock_llm_client, mock_documents):
        """Test the complete document ingestion pipeline."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock the document processing steps
        processed_docs = []
        chunks_created = []

        async def mock_add_document(doc):
            processed_docs.append(doc)

        async def mock_add_chunk(chunk):
            chunks_created.append(chunk)

        rag_engine.document_store.add_document = AsyncMock(side_effect=mock_add_document)
        rag_engine.document_store.add_chunk = AsyncMock(side_effect=mock_add_chunk)

        # Mock chunking to create realistic chunks
        async def mock_chunk_document(document):
            # Simple sentence-based chunking
            sentences = document.content.split('.')
            chunks = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunks.append(Document(
                        id=f"{document.id}_chunk_{i}",
                        content=sentence.strip() + ".",
                        metadata={**document.metadata, "chunk_index": i, "parent_id": document.id}
                    ))
            return chunks

        rag_engine.processor.chunk_document = AsyncMock(side_effect=mock_chunk_document)

        # Mock document processing to return the document as-is
        async def mock_process_document(file_path, **metadata):
            # Find matching mock document
            for doc in mock_documents:
                if doc.id in file_path:
                    return Document(
                        id=doc.id,
                        content=doc.content,
                        metadata={**doc.metadata, "source": file_path, **metadata}
                    )
            return mock_documents[0]  # Fallback

        rag_engine.processor.process_document = AsyncMock(side_effect=mock_process_document)

        # Test ingestion pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock files
            file_paths = []
            for i, doc in enumerate(mock_documents):
                file_path = Path(tmpdir) / f"{doc.id}.txt"
                file_path.write_text(doc.content)
                file_paths.append(str(file_path))

            # Ingest documents
            results = await rag_engine.add_documents_batch(file_paths)

            # Verify results
            assert len(results) == len(mock_documents)
            assert len(processed_docs) == len(mock_documents)
            assert len(chunks_created) > 0  # Should have created chunks

            # Verify each document was processed
            for doc in processed_docs:
                assert doc.id in [d.id for d in mock_documents]
                assert "source" in doc.metadata

    @pytest.mark.asyncio
    async def test_query_processing_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test the complete query processing pipeline."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock retrieval results
        mock_retrieval_results = [
            MagicMock(
                score=0.92,
                chunk=MagicMock(content="Natural Language Processing is a field of AI..."),
                metadata={"title": "NLP Introduction", "chunk_index": 0}
            ),
            MagicMock(
                score=0.87,
                chunk=MagicMock(content="Deep Learning is a subset of machine learning..."),
                metadata={"title": "Deep Learning", "chunk_index": 0}
            ),
            MagicMock(
                score=0.81,
                chunk=MagicMock(content="Machine Learning algorithms can be categorized..."),
                metadata={"title": "ML Types", "chunk_index": 0}
            )
        ]

        rag_engine.retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)

        # Mock LLM responses for different pipeline stages
        def mock_llm_response(*args, **kwargs):
            messages = args[0] if args else kwargs.get('messages', [])
            last_message = messages[-1].content if messages else ""

            if "expand" in last_message.lower() or "related" in last_message.lower():
                # Query expansion
                response = MagicMock()
                response.content = '["what is NLP", "natural language processing applications", "NLP vs ML"]'
                return response
            elif "Context Information:" in last_message:
                # Answer generation
                response = MagicMock()
                response.content = """
                Natural Language Processing (NLP) is a field of artificial intelligence that focuses on
                enabling computers to understand, interpret, and generate human language. It combines
                computational linguistics with machine learning and deep learning to process and analyze
                large amounts of natural language data.
                """
                return response
            else:
                # Default response
                response = MagicMock()
                response.content = "General AI response"
                return response

        mock_llm_client.acomplete = AsyncMock(side_effect=mock_llm_response)

        # Test query processing
        query = "What is natural language processing?"
        response = await rag_engine.query(query)

        # Verify pipeline execution
        assert response.query == query
        assert response.answer is not None
        assert len(response.sources) == 3
        assert response.confidence > 0
        assert response.processing_time > 0

        # Verify retrieval was called
        rag_engine.retriever.retrieve.assert_called_once()

        # Verify LLM was called for expansion and generation
        assert mock_llm_client.acomplete.call_count >= 2

    @pytest.mark.asyncio
    async def test_query_expansion_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test query expansion pipeline functionality."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock query expansion
        def mock_expansion_response(*args, **kwargs):
            response = MagicMock()
            response.content = '["machine learning basics", "ML algorithms", "supervised learning"]'
            return response

        mock_llm_client.acomplete = AsyncMock(side_effect=mock_expansion_response)

        # Test query expansion
        original_query = "What is machine learning?"
        context = QueryContext(original_query=original_query)

        expanded_context = await rag_engine._expand_query(
            original_query, context, use_structured_thinking=False
        )

        # Verify expansion
        assert len(expanded_context.expanded_queries) > 0
        assert "machine learning basics" in expanded_context.expanded_queries
        assert "ML algorithms" in expanded_context.expanded_queries

    @pytest.mark.asyncio
    async def test_structured_thinking_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test structured thinking pipeline."""
        from ragents.llm.types import StructuredThought

        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock structured thinking response
        mock_structured_thought = StructuredThought(
            query_analysis="This query asks about the definition and applications of machine learning",
            sources_needed=["ML definition", "ML types", "ML applications", "ML examples"],
            reasoning_steps=[
                "Define what machine learning is",
                "Explain different types of ML",
                "Provide practical examples",
                "Discuss current applications"
            ],
            final_answer="Machine learning is a method of data analysis that automates analytical model building."
        )

        mock_llm_client._instructor_async.chat.completions.create.return_value = mock_structured_thought

        # Test structured thinking expansion
        context = QueryContext(original_query="What is machine learning?")
        expanded_context = await rag_engine._expand_query(
            "What is machine learning?", context, use_structured_thinking=True
        )

        # Verify structured expansion
        assert len(expanded_context.expanded_queries) == 4
        assert "ML definition" in expanded_context.expanded_queries
        assert "ML applications" in expanded_context.expanded_queries

    @pytest.mark.asyncio
    async def test_confidence_calculation_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test confidence calculation in the pipeline."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Test different retrieval scenarios
        test_cases = [
            {
                "results": [],  # No results
                "expected_confidence": 0.0
            },
            {
                "results": [
                    MagicMock(score=0.95),
                    MagicMock(score=0.90),
                    MagicMock(score=0.85)
                ],
                "expected_confidence": lambda c: c > 0.8  # High confidence
            },
            {
                "results": [
                    MagicMock(score=0.6),
                    MagicMock(score=0.5),
                    MagicMock(score=0.4)
                ],
                "expected_confidence": lambda c: 0.3 < c < 0.7  # Medium confidence
            },
            {
                "results": [
                    MagicMock(score=0.3),
                    MagicMock(score=0.2)
                ],
                "expected_confidence": lambda c: c < 0.4  # Low confidence
            }
        ]

        context = QueryContext(original_query="test")

        for case in test_cases:
            confidence = await rag_engine._calculate_confidence(
                context, case["results"], "test answer"
            )

            if callable(case["expected_confidence"]):
                assert case["expected_confidence"](confidence), f"Confidence {confidence} doesn't match expected range"
            else:
                assert confidence == case["expected_confidence"], f"Expected {case['expected_confidence']}, got {confidence}"

    @pytest.mark.asyncio
    async def test_error_recovery_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test error recovery in the pipeline."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Test retrieval failure recovery
        rag_engine.retriever.retrieve = AsyncMock(side_effect=Exception("Retrieval failed"))

        with pytest.raises(Exception):
            await rag_engine.query("test query")

        # Test LLM failure recovery
        rag_engine.retriever.retrieve = AsyncMock(return_value=[])
        mock_llm_client.acomplete = AsyncMock(side_effect=Exception("LLM failed"))

        with pytest.raises(Exception):
            await rag_engine.query("test query")

    @pytest.mark.asyncio
    async def test_caching_pipeline(self, mock_llm_client):
        """Test caching in the pipeline."""
        # Enable caching
        config = RAGConfig(
            vector_store_type="memory",
            enable_caching=True,
            cache_ttl_hours=1
        )

        rag_engine = RAGEngine(config, mock_llm_client)

        # Mock components
        rag_engine.retriever.retrieve = AsyncMock(return_value=[])
        mock_llm_client.acomplete = AsyncMock(return_value=MagicMock(content="Cached response"))

        # This would test caching if implemented
        # For now, just verify the engine can be created with caching enabled
        assert rag_engine.config.enable_caching is True

    @pytest.mark.asyncio
    async def test_multimodal_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test multimodal processing pipeline."""
        # Enable vision processing
        pipeline_rag_config.enable_vision = True
        pipeline_rag_config.enable_table_extraction = True

        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock multimodal processor
        async def mock_multimodal_process(file_path, **metadata):
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.jpg', '.png', '.pdf']:
                content = f"Image/PDF content extracted from {file_path}"
                metadata["content_type"] = "multimodal"
            else:
                content = f"Text content from {file_path}"
                metadata["content_type"] = "text"

            return Document(
                id=f"doc_{Path(file_path).stem}",
                content=content,
                metadata=metadata
            )

        rag_engine.processor.process_document = AsyncMock(side_effect=mock_multimodal_process)
        rag_engine.processor.chunk_document = AsyncMock(return_value=[])
        rag_engine.document_store.add_document = AsyncMock()
        rag_engine.document_store.add_chunk = AsyncMock()

        # Test processing different file types
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock files
            text_file = Path(tmpdir) / "document.txt"
            text_file.write_text("Sample text content")

            # Process the file
            doc = await rag_engine.add_document(str(text_file))

            assert doc.content == f"Text content from {text_file}"
            assert doc.metadata["content_type"] == "text"

    @pytest.mark.asyncio
    async def test_reranking_pipeline_integration(self, pipeline_rag_config, mock_llm_client):
        """Test reranking pipeline integration."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock initial retrieval results
        initial_results = [
            MagicMock(
                chunk=MagicMock(content=f"Content {i}"),
                score=0.8 - i * 0.1,
                metadata={},
                document_id=f"doc{i}"
            )
            for i in range(5)
        ]

        # Mock reranking result
        from ragents.reranking.base import RetrievedDocument, RerankingResult

        reranked_docs = [
            RetrievedDocument(
                content=f"Reranked content {i}",
                metadata={},
                similarity_score=0.9 - i * 0.05,
                document_id=f"doc{i}"
            )
            for i in range(3)  # Reduced to top 3
        ]

        mock_reranking_result = RerankingResult(
            reranked_documents=reranked_docs,
            reranking_scores=[0.9, 0.85, 0.8],
            metadata={"strategy": "hybrid"}
        )

        rag_engine.reranker.rerank = AsyncMock(return_value=mock_reranking_result)

        # Test reranking application
        result = await rag_engine._apply_reranking_and_autocut("test query", initial_results)

        # Verify reranking was applied
        assert len(result) == 3  # Should be reduced by reranking
        rag_engine.reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_optimization_pipeline(self, pipeline_rag_config, mock_llm_client):
        """Test performance optimization features in the pipeline."""
        rag_engine = RAGEngine(pipeline_rag_config, mock_llm_client)

        # Mock batch processing
        rag_engine.document_store.add_document = AsyncMock()
        rag_engine.document_store.add_chunk = AsyncMock()
        rag_engine.processor.process_document = AsyncMock(return_value=Document(
            id="test", content="test", metadata={}
        ))
        rag_engine.processor.chunk_document = AsyncMock(return_value=[])

        # Test batch document processing
        file_paths = [f"doc{i}.txt" for i in range(10)]

        import time
        start_time = time.time()

        results = await rag_engine.add_documents_batch(file_paths)

        processing_time = time.time() - start_time

        # Verify batch processing completed
        assert len(results) == 10
        assert processing_time < 5.0  # Should be reasonably fast with mocking

        # Verify all documents were processed
        assert rag_engine.processor.process_document.call_count == 10