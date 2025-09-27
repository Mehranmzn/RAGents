"""Unit tests for RAG engine."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ragents.rag.engine import RAGEngine
from ragents.rag.types import QueryContext, Document
from ragents.llm.types import StructuredThought


class TestRAGEngine:
    """Test cases for RAGEngine."""

    def test_rag_engine_initialization(self, rag_config, mock_llm_client):
        """Test RAG engine initialization."""
        engine = RAGEngine(rag_config, mock_llm_client)

        assert engine.config == rag_config
        assert engine.llm_client == mock_llm_client
        assert engine.document_store is not None
        assert engine.retriever is not None
        assert engine.processor is not None

    def test_rag_engine_with_custom_components(self, rag_config, mock_llm_client):
        """Test RAG engine with custom components."""
        mock_document_store = MagicMock()
        mock_retriever = MagicMock()
        mock_processor = MagicMock()

        engine = RAGEngine(
            rag_config,
            mock_llm_client,
            document_store=mock_document_store,
            retriever=mock_retriever,
            processor=mock_processor
        )

        assert engine.document_store == mock_document_store
        assert engine.retriever == mock_retriever
        assert engine.processor == mock_processor

    @pytest.mark.asyncio
    async def test_add_document(self, rag_config, mock_llm_client):
        """Test adding a document."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock the processor and document store
        mock_document = Document(
            id="test-doc",
            content="Test content",
            metadata={"title": "Test"}
        )
        mock_chunks = [
            Document(id="chunk-1", content="Chunk 1"),
            Document(id="chunk-2", content="Chunk 2")
        ]

        engine.processor.process_document = AsyncMock(return_value=mock_document)
        engine.processor.chunk_document = AsyncMock(return_value=mock_chunks)
        engine.document_store.add_document = AsyncMock()
        engine.document_store.add_chunk = AsyncMock()

        result = await engine.add_document("test.txt", title="Test")

        assert result == mock_document
        engine.processor.process_document.assert_called_once_with("test.txt", title="Test")
        engine.document_store.add_document.assert_called_once_with(mock_document)
        assert engine.document_store.add_chunk.call_count == 2

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, rag_config, mock_llm_client):
        """Test adding multiple documents in batch."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock add_document method
        mock_docs = [
            Document(id="doc1", content="Content 1"),
            Document(id="doc2", content="Content 2")
        ]
        engine.add_document = AsyncMock(side_effect=mock_docs)

        result = await engine.add_documents_batch(["file1.txt", "file2.txt"])

        assert len(result) == 2
        assert result == mock_docs

    @pytest.mark.asyncio
    async def test_query_basic(self, rag_config, mock_llm_client):
        """Test basic query processing."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock components
        mock_retrieval_results = [
            MagicMock(score=0.9, chunk=MagicMock(content="Relevant content"))
        ]
        engine.retriever.retrieve = AsyncMock(return_value=mock_retrieval_results)
        engine._expand_query = AsyncMock(return_value=QueryContext(original_query="test"))
        engine._generate_answer = AsyncMock(return_value=("Test answer", "Test reasoning"))
        engine._calculate_confidence = AsyncMock(return_value=0.8)

        response = await engine.query("What is test?")

        assert response.query == "What is test?"
        assert response.answer == "Test answer"
        assert response.confidence == 0.8
        assert len(response.sources) == 1

    @pytest.mark.asyncio
    async def test_query_with_context(self, rag_config, mock_llm_client):
        """Test query processing with provided context."""
        engine = RAGEngine(rag_config, mock_llm_client)

        context = QueryContext(
            original_query="test",
            expanded_queries=["related test"]
        )

        # Mock components
        engine.retriever.retrieve = AsyncMock(return_value=[])
        engine._expand_query = AsyncMock(return_value=context)
        engine._generate_answer = AsyncMock(return_value=("Answer", None))
        engine._calculate_confidence = AsyncMock(return_value=0.5)

        response = await engine.query("What is test?", context=context)

        engine._expand_query.assert_called_once()
        assert response.query == "What is test?"

    @pytest.mark.asyncio
    async def test_query_without_structured_thinking(self, rag_config, mock_llm_client):
        """Test query processing without structured thinking."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock components
        engine.retriever.retrieve = AsyncMock(return_value=[])
        engine._expand_query = AsyncMock(return_value=QueryContext(original_query="test"))
        engine._generate_answer = AsyncMock(return_value=("Answer", None))
        engine._calculate_confidence = AsyncMock(return_value=0.5)

        response = await engine.query("What is test?", use_structured_thinking=False)

        assert response.reasoning is None  # No reasoning when structured thinking is off

    @pytest.mark.asyncio
    async def test_expand_query_with_structured_thinking(self, rag_config, mock_llm_client):
        """Test query expansion with structured thinking."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock structured thought response
        mock_structured_thought = StructuredThought(
            query_analysis="Analysis of the query",
            sources_needed=["source1", "source2"],
            reasoning_steps=["step1", "step2"],
            final_answer="Test answer"
        )
        mock_llm_client._instructor_async.chat.completions.create.return_value = mock_structured_thought

        context = QueryContext(original_query="test query")
        result = await engine._expand_query("test query", context, use_structured_thinking=True)

        assert "source1" in result.expanded_queries
        assert "source2" in result.expanded_queries

    @pytest.mark.asyncio
    async def test_expand_query_simple(self, rag_config, mock_llm_client):
        """Test simple query expansion."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock LLM response with JSON list
        mock_response = MagicMock()
        mock_response.content = '["expanded query 1", "expanded query 2"]'
        mock_llm_client.acomplete.return_value = mock_response

        context = QueryContext(original_query="test query")
        result = await engine._expand_query("test query", context, use_structured_thinking=False)

        assert "expanded query 1" in result.expanded_queries
        assert "expanded query 2" in result.expanded_queries

    @pytest.mark.asyncio
    async def test_expand_query_json_parse_error(self, rag_config, mock_llm_client):
        """Test query expansion with JSON parsing error."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock LLM response with invalid JSON
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON"
        mock_llm_client.acomplete.return_value = mock_response

        context = QueryContext(original_query="test query")
        result = await engine._expand_query("test query", context, use_structured_thinking=False)

        # Should not add any expanded queries on JSON error
        assert len(result.expanded_queries) == 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_structured_thinking(self, rag_config, mock_llm_client):
        """Test answer generation with structured thinking."""
        engine = RAGEngine(rag_config, mock_llm_client)

        context = QueryContext(original_query="test query")
        mock_retrieval_results = [
            MagicMock(score=0.9, chunk=MagicMock(content="Test content"))
        ]

        # Mock structured response
        mock_structured_response = StructuredThought(
            query_analysis="Query analysis",
            sources_needed=["source1"],
            reasoning_steps=["step1"],
            final_answer="Structured answer"
        )
        mock_llm_client._instructor_async.chat.completions.create.return_value = mock_structured_response

        answer, reasoning = await engine._generate_answer(
            context, mock_retrieval_results, use_structured_thinking=True
        )

        assert answer == "Structured answer"
        assert reasoning == "Query analysis"

    @pytest.mark.asyncio
    async def test_generate_answer_simple(self, rag_config, mock_llm_client):
        """Test simple answer generation."""
        engine = RAGEngine(rag_config, mock_llm_client)

        context = QueryContext(original_query="test query")
        mock_retrieval_results = [
            MagicMock(score=0.9, chunk=MagicMock(content="Test content"))
        ]

        # Mock simple response
        mock_response = MagicMock()
        mock_response.content = "Simple answer"
        mock_llm_client.acomplete.return_value = mock_response

        answer, reasoning = await engine._generate_answer(
            context, mock_retrieval_results, use_structured_thinking=False
        )

        assert answer == "Simple answer"
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_calculate_confidence_no_results(self, rag_config, mock_llm_client):
        """Test confidence calculation with no retrieval results."""
        engine = RAGEngine(rag_config, mock_llm_client)

        context = QueryContext(original_query="test")
        confidence = await engine._calculate_confidence(context, [], "answer")

        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_calculate_confidence_with_results(self, rag_config, mock_llm_client):
        """Test confidence calculation with retrieval results."""
        engine = RAGEngine(rag_config, mock_llm_client)

        context = QueryContext(original_query="test")
        mock_results = [
            MagicMock(score=0.9),
            MagicMock(score=0.8),
            MagicMock(score=0.7)
        ]

        confidence = await engine._calculate_confidence(context, mock_results, "answer")

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # Should have some confidence with good scores

    @pytest.mark.asyncio
    async def test_get_document_stats(self, rag_config, mock_llm_client):
        """Test getting document statistics."""
        engine = RAGEngine(rag_config, mock_llm_client)

        mock_stats = {"total_documents": 10, "total_chunks": 50}
        engine.document_store.get_stats = AsyncMock(return_value=mock_stats)

        stats = await engine.get_document_stats()

        assert stats == mock_stats
        engine.document_store.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_documents(self, rag_config, mock_llm_client):
        """Test clearing all documents."""
        engine = RAGEngine(rag_config, mock_llm_client)

        engine.document_store.clear = AsyncMock()

        await engine.clear_documents()

        engine.document_store.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_document(self, rag_config, mock_llm_client):
        """Test removing a specific document."""
        engine = RAGEngine(rag_config, mock_llm_client)

        engine.document_store.remove_document = AsyncMock(return_value=True)

        result = await engine.remove_document("doc-123")

        assert result is True
        engine.document_store.remove_document.assert_called_once_with("doc-123")

    def test_init_default_reranker(self, rag_config, mock_llm_client):
        """Test default reranker initialization."""
        engine = RAGEngine(rag_config, mock_llm_client)
        reranker = engine._init_default_reranker()

        assert reranker is not None
        # Should return a HybridReranker by default

    @pytest.mark.asyncio
    async def test_apply_reranking_and_autocut_empty_results(self, rag_config, mock_llm_client):
        """Test reranking with empty results."""
        engine = RAGEngine(rag_config, mock_llm_client)

        result = await engine._apply_reranking_and_autocut("query", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_apply_reranking_and_autocut_with_results(self, rag_config, mock_llm_client):
        """Test reranking with actual results."""
        engine = RAGEngine(rag_config, mock_llm_client)

        # Mock retrieval results
        mock_results = [
            MagicMock(
                chunk=MagicMock(content="Content 1"),
                score=0.9,
                metadata={},
                document_id="doc1"
            ),
            MagicMock(
                chunk=MagicMock(content="Content 2"),
                score=0.8,
                metadata={},
                document_id="doc2"
            )
        ]

        # Mock reranker
        engine.reranker.rerank = AsyncMock(return_value=MagicMock(
            reranked_documents=mock_results
        ))

        result = await engine._apply_reranking_and_autocut("query", mock_results)

        assert len(result) == 2
        engine.reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_reranking_exception_handling(self, rag_config, mock_llm_client):
        """Test reranking with exception handling."""
        engine = RAGEngine(rag_config, mock_llm_client)

        mock_results = [MagicMock(chunk=MagicMock(content="Content"), score=0.9)]

        # Mock reranker to raise exception
        engine.reranker.rerank = AsyncMock(side_effect=Exception("Reranking failed"))

        result = await engine._apply_reranking_and_autocut("query", mock_results)

        # Should return original results when reranking fails
        assert len(result) == 1