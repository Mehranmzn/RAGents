"""Main RAG engine implementation."""

import asyncio
import time

from ragents.config.rag_config import RAGConfig
from ragents.ingestion.config import IngestionConfig
from ragents.ingestion.processors import MultiModalProcessor
from ragents.llm.client import LLMClient
from ragents.llm.types import ChatMessage, MessageRole, StructuredThought
from ragents.rag.cache import CacheConfig, RAGCache
from ragents.rag.document_store import DocumentStore
from ragents.rag.retriever import Retriever
from ragents.rag.types import Document, QueryContext, RAGResponse
from ragents.reranking.autocut import AutocutFilter
from ragents.reranking.base import Reranker, RetrievedDocument
from ragents.reranking.config import RerankingConfig
from ragents.reranking.strategies import HybridReranker, SemanticReranker


class RAGEngine:
    """Main RAG engine with multimodal capabilities."""

    def __init__(
        self,
        config: RAGConfig,
        llm_client: LLMClient,
        document_store: DocumentStore | None = None,
        retriever: Retriever | None = None,
        processor: MultiModalProcessor | None = None,
        reranker: Reranker | None = None,
        autocut_filter: AutocutFilter | None = None,
        cache: RAGCache | None = None,
        cache_config: CacheConfig | None = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.document_store = document_store or DocumentStore(config)
        self.retriever = retriever or Retriever(config, self.document_store)

        # Create IngestionConfig from RAGConfig for the processor
        ingestion_config = IngestionConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_file_size_mb=config.max_file_size_mb,
            supported_formats=config.supported_formats,
            extract_tables=config.enable_table_extraction,
            extract_images=config.enable_vision,
        )
        self.processor = processor or MultiModalProcessor(ingestion_config)

        # Initialize caching layer
        self.cache = cache
        self.cache_config = cache_config
        self._cache_initialized = False

        # Initialize reranking components
        self.reranking_config = getattr(config, "reranking", RerankingConfig())
        self.reranker = reranker or self._init_default_reranker()
        self.autocut_filter = autocut_filter or AutocutFilter(
            strategy=self.reranking_config.cutoff_strategy
        )

    async def add_document(self, file_path: str, **metadata) -> Document:
        """Add a document to the RAG system."""
        # Process document through multimodal pipeline
        document = await self.processor.process_document(file_path, **metadata)

        # Store document and chunks
        await self.document_store.add_document(document)
        chunks = await self.processor.chunk_document(document)

        for chunk in chunks:
            await self.document_store.add_chunk(chunk)

        return document

    async def add_documents_batch(self, file_paths: list[str]) -> list[Document]:
        """Add multiple documents in batch."""
        tasks = [self.add_document(path) for path in file_paths]
        return await asyncio.gather(*tasks)

    async def _ensure_cache_initialized(self):
        """Ensure cache is initialized."""
        if not self._cache_initialized and self.cache_config:
            if self.cache is None:
                from .cache import get_cache

                self.cache = await get_cache(self.cache_config)
            elif not self.cache._initialized:
                await self.cache.initialize()
            self._cache_initialized = True

    async def query(
        self,
        query: str,
        context: QueryContext | None = None,
        use_structured_thinking: bool = True,
    ) -> RAGResponse:
        """Query the RAG system with optional structured thinking."""
        start_time = time.time()

        # Initialize cache if configured
        await self._ensure_cache_initialized()

        # Check cache for retrieval results
        cached_result = None
        if self.cache:
            cached_result = await self.cache.get_retrieval(query, self.config.top_k)
            if cached_result:
                # Return cached response
                return RAGResponse(**cached_result)

        if context is None:
            context = QueryContext(original_query=query)

        # Step 1: Query expansion and planning
        if self.config.query_expansion or use_structured_thinking:
            context = await self._expand_query(query, context, use_structured_thinking)

        # Step 2: Retrieve relevant chunks
        retrieval_results = await self.retriever.retrieve(context)

        # Step 2.5: Apply reranking and Autocut filtering
        if self.reranker and retrieval_results:
            retrieval_results = await self._apply_reranking_and_autocut(
                query, retrieval_results
            )

        # Step 3: Generate answer with context
        answer, reasoning = await self._generate_answer(
            context, retrieval_results, use_structured_thinking
        )

        # Step 4: Calculate confidence score
        confidence = await self._calculate_confidence(
            context, retrieval_results, answer
        )

        processing_time = time.time() - start_time

        # Extract chunks from results - handle both RetrievalResult and RetrievedDocument
        from .types import ChunkType, ContentChunk

        context_chunks = []
        for result in retrieval_results:
            if hasattr(result, "chunk"):
                # RetrievalResult type
                context_chunks.append(result.chunk)
            else:
                # RetrievedDocument type - create a ContentChunk
                chunk = ContentChunk(
                    id=result.document_id or str(hash(result.content)),
                    document_id=result.document_id or "unknown",
                    content=result.content,
                    chunk_type=ChunkType.TEXT,
                    metadata=result.metadata,
                    start_index=0,
                    end_index=len(result.content),
                    created_at=time.time(),
                )
                context_chunks.append(chunk)

        response = RAGResponse(
            query=query,
            answer=answer,
            sources=retrieval_results,
            context_chunks=context_chunks,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={
                "expanded_queries": context.expanded_queries,
                "retrieval_strategy": context.retrieval_strategy,
                "num_sources": len(retrieval_results),
            },
        )

        # Cache the result
        if self.cache:
            await self.cache.cache_retrieval(
                query, response.model_dump(), self.config.top_k
            )

        return response

    async def _expand_query(
        self, query: str, context: QueryContext, use_structured_thinking: bool
    ) -> QueryContext:
        """Expand query with related terms and sub-queries."""
        if use_structured_thinking:
            # Use structured thinking to analyze the query
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are an expert at analyzing queries and planning information retrieval. "
                        "Break down complex queries into structured thinking steps."
                    ),
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=f"Analyze this query and plan how to find relevant information: {query}",
                ),
            ]

            structured_thought = await self.llm_client.acomplete(
                messages, response_model=StructuredThought
            )

            context.expanded_queries.extend(structured_thought.sources_needed)
        else:
            # Simple query expansion
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Generate 2-3 related search queries that would help answer the user's question. "
                        "Return them as a JSON list of strings."
                    ),
                ),
                ChatMessage(role=MessageRole.USER, content=query),
            ]

            response = await self.llm_client.acomplete(messages)
            # Parse expanded queries from response
            try:
                import json

                expanded = json.loads(response.content)
                if isinstance(expanded, list):
                    context.expanded_queries.extend(expanded[:3])
            except Exception:
                pass

        return context

    async def _generate_answer(
        self, context: QueryContext, retrieval_results, use_structured_thinking: bool
    ) -> tuple[str, str | None]:
        """Generate answer from retrieved context."""
        # Prepare context from retrieval results
        # Handle both RetrievalResult and RetrievedDocument types
        context_parts = []
        for i, result in enumerate(retrieval_results[: self.config.top_k]):
            # Get score - handle both types
            score = getattr(result, "score", None) or getattr(
                result, "similarity_score", 0.0
            )
            # Get content - handle both types
            content = getattr(result, "content", None) or (
                result.chunk.content if hasattr(result, "chunk") else str(result)
            )
            context_parts.append(f"Source {i+1} (score: {score:.3f}):\n{content}")

        context_text = "\n\n".join(context_parts)

        if use_structured_thinking:
            system_prompt = (
                "You are an expert assistant that provides accurate, well-reasoned answers "
                "based on the provided context. Use structured thinking to analyze the "
                "information and provide a comprehensive response."
            )

            user_prompt = f"""
            Context Information:
            {context_text}

            Question: {context.original_query}

            Please provide a detailed answer based on the context above. If the context doesn't
            contain enough information to fully answer the question, state this clearly.
            """

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]

            structured_response = await self.llm_client.acomplete(
                messages, response_model=StructuredThought
            )

            return structured_response.final_answer, structured_response.query_analysis
        else:
            # Simple answer generation
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based on the "
                "provided context. Be accurate and concise."
            )

            user_prompt = f"""
Context:
{context_text}

Question: {context.original_query}

Answer:"""

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]

            response = await self.llm_client.acomplete(messages)
            return response.content, None

    def _init_default_reranker(self) -> Reranker:
        """Initialize default reranker based on configuration."""
        if hasattr(self.reranking_config, "strategy"):
            if self.reranking_config.strategy.value == "hybrid":
                return HybridReranker(weights=self.reranking_config.fusion_weights)
            elif self.reranking_config.strategy.value == "semantic":
                return SemanticReranker(self.reranking_config.semantic_model)
            else:
                return HybridReranker()  # Default fallback
        else:
            return HybridReranker()

    async def _apply_reranking_and_autocut(self, query: str, retrieval_results) -> list:
        """Apply reranking and Autocut filtering to retrieval results."""
        if not retrieval_results:
            return retrieval_results

        # Convert retrieval results to RetrievedDocument format
        retrieved_docs = []
        for result in retrieval_results:
            doc = RetrievedDocument(
                content=(
                    result.chunk.content if hasattr(result, "chunk") else str(result)
                ),
                metadata=getattr(result, "metadata", {}),
                similarity_score=getattr(result, "score", 0.5),
                document_id=getattr(result, "document_id", None),
                source=getattr(result, "source", None),
                chunk_index=getattr(result, "chunk_index", None),
            )
            retrieved_docs.append(doc)

        # Apply reranking
        try:
            reranking_result = await self.reranker.rerank(
                query, retrieved_docs, top_k=self.reranking_config.top_k
            )
            reranked_docs = reranking_result.reranked_documents
        except Exception as e:
            print(f"Reranking failed: {e}, using original order")
            reranked_docs = retrieved_docs

        # Apply Autocut filtering if enabled
        if self.reranking_config.enable_autocut and self.autocut_filter:
            try:
                filtered_docs, cutoff_result = self.autocut_filter.filter_documents(
                    reranked_docs, strategy=self.reranking_config.cutoff_strategy
                )
                reranked_docs = filtered_docs

                # Log cutoff results for analysis
                print(
                    f"Autocut applied: kept {cutoff_result.kept_count}, removed {cutoff_result.removed_count}"
                )

            except Exception as e:
                print(f"Autocut filtering failed: {e}, using all reranked documents")

        # Convert RetrievedDocument back to RetrievalResult for compatibility
        from .types import ChunkType, ContentChunk, RetrievalResult

        final_results = []
        for rank, doc in enumerate(
            reranked_docs[: self.reranking_config.max_documents], 1
        ):
            # Create a ContentChunk from the RetrievedDocument
            chunk = ContentChunk(
                id=doc.document_id or str(hash(doc.content)),
                document_id=doc.document_id or "unknown",
                content=doc.content,
                chunk_type=ChunkType.TEXT,
                metadata=doc.metadata,
                start_index=0,
                end_index=len(doc.content),
                created_at=time.time(),
            )
            # Create RetrievalResult
            result = RetrievalResult(
                chunk=chunk,
                score=doc.similarity_score,
                rank=rank,
                retrieval_method="reranked",
            )
            final_results.append(result)

        return final_results

    async def _calculate_confidence(
        self, context: QueryContext, retrieval_results, answer: str
    ) -> float:
        """Calculate confidence score for the generated answer."""
        if not retrieval_results:
            return 0.0

        # Get scores - handle both RetrievalResult and RetrievedDocument types
        scores = []
        for r in retrieval_results:
            # Try to get score attribute
            score = getattr(r, "score", None)
            if score is None:
                # Try similarity_score attribute
                score = getattr(r, "similarity_score", None)
            if score is None:
                score = 0.0
            scores.append(float(score))

        # Simple confidence calculation based on retrieval scores
        avg_retrieval_score = sum(scores) / len(scores) if scores else 0.0

        # Factor in number of good quality results
        high_quality_results = sum(1 for score in scores if score > 0.8)
        coverage_score = min(high_quality_results / 3, 1.0)

        # Combine scores
        confidence = (avg_retrieval_score * 0.7) + (coverage_score * 0.3)

        return max(min(confidence, 1.0), 0.0)

    async def get_document_stats(self) -> dict:
        """Get statistics about the document collection."""
        return await self.document_store.get_stats()

    async def clear_documents(self) -> None:
        """Clear all documents from the system."""
        await self.document_store.clear()

    async def remove_document(self, document_id: str) -> bool:
        """Remove a specific document."""
        return await self.document_store.remove_document(document_id)
