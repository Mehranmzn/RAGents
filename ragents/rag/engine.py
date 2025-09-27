"""Main RAG engine implementation."""

import asyncio
import time
from typing import Dict, List, Optional

from ..config.rag_config import RAGConfig
from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole, StructuredThought
from ..ingestion.processors import MultiModalProcessor
from ..reranking.base import Reranker, RetrievedDocument
from ..reranking.strategies import HybridReranker, SemanticReranker
from ..reranking.autocut import AutocutFilter, CutoffStrategy
from ..reranking.config import RerankingConfig
from .document_store import DocumentStore
from .retriever import Retriever
from .types import Document, QueryContext, RAGResponse


class RAGEngine:
    """Main RAG engine with multimodal capabilities."""

    def __init__(
        self,
        config: RAGConfig,
        llm_client: LLMClient,
        document_store: Optional[DocumentStore] = None,
        retriever: Optional[Retriever] = None,
        processor: Optional[MultiModalProcessor] = None,
        reranker: Optional[Reranker] = None,
        autocut_filter: Optional[AutocutFilter] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.document_store = document_store or DocumentStore(config)
        self.retriever = retriever or Retriever(config, self.document_store)
        self.processor = processor or MultiModalProcessor(config)

        # Initialize reranking components
        self.reranking_config = getattr(config, 'reranking', RerankingConfig())
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

    async def add_documents_batch(self, file_paths: List[str]) -> List[Document]:
        """Add multiple documents in batch."""
        tasks = [self.add_document(path) for path in file_paths]
        return await asyncio.gather(*tasks)

    async def query(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        use_structured_thinking: bool = True,
    ) -> RAGResponse:
        """Query the RAG system with optional structured thinking."""
        start_time = time.time()

        if context is None:
            context = QueryContext(original_query=query)

        # Step 1: Query expansion and planning
        if self.config.query_expansion or use_structured_thinking:
            context = await self._expand_query(query, context, use_structured_thinking)

        # Step 2: Retrieve relevant chunks
        retrieval_results = await self.retriever.retrieve(context)

        # Step 2.5: Apply reranking and Autocut filtering
        if self.reranker and retrieval_results:
            retrieval_results = await self._apply_reranking_and_autocut(query, retrieval_results)

        # Step 3: Generate answer with context
        answer, reasoning = await self._generate_answer(
            context, retrieval_results, use_structured_thinking
        )

        # Step 4: Calculate confidence score
        confidence = await self._calculate_confidence(context, retrieval_results, answer)

        processing_time = time.time() - start_time

        return RAGResponse(
            query=query,
            answer=answer,
            sources=retrieval_results,
            context_chunks=[result.chunk for result in retrieval_results],
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={
                "expanded_queries": context.expanded_queries,
                "retrieval_strategy": context.retrieval_strategy,
                "num_sources": len(retrieval_results),
            },
        )

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
            except:
                pass

        return context

    async def _generate_answer(
        self, context: QueryContext, retrieval_results, use_structured_thinking: bool
    ) -> tuple[str, Optional[str]]:
        """Generate answer from retrieved context."""
        # Prepare context from retrieval results
        context_text = "\n\n".join(
            [
                f"Source {i+1} (score: {result.score:.3f}):\n{result.chunk.content}"
                for i, result in enumerate(retrieval_results[:self.config.top_k])
            ]
        )

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
        if hasattr(self.reranking_config, 'strategy'):
            if self.reranking_config.strategy.value == "hybrid":
                return HybridReranker(weights=self.reranking_config.fusion_weights)
            elif self.reranking_config.strategy.value == "semantic":
                return SemanticReranker(self.reranking_config.semantic_model)
            else:
                return HybridReranker()  # Default fallback
        else:
            return HybridReranker()

    async def _apply_reranking_and_autocut(self, query: str, retrieval_results) -> List:
        """Apply reranking and Autocut filtering to retrieval results."""
        if not retrieval_results:
            return retrieval_results

        # Convert retrieval results to RetrievedDocument format
        retrieved_docs = []
        for result in retrieval_results:
            doc = RetrievedDocument(
                content=result.chunk.content if hasattr(result, 'chunk') else str(result),
                metadata=getattr(result, 'metadata', {}),
                similarity_score=getattr(result, 'score', 0.5),
                document_id=getattr(result, 'document_id', None),
                source=getattr(result, 'source', None),
                chunk_index=getattr(result, 'chunk_index', None)
            )
            retrieved_docs.append(doc)

        # Apply reranking
        try:
            reranking_result = await self.reranker.rerank(
                query,
                retrieved_docs,
                top_k=self.reranking_config.top_k
            )
            reranked_docs = reranking_result.reranked_documents
        except Exception as e:
            print(f"Reranking failed: {e}, using original order")
            reranked_docs = retrieved_docs

        # Apply Autocut filtering if enabled
        if self.reranking_config.enable_autocut and self.autocut_filter:
            try:
                filtered_docs, cutoff_result = self.autocut_filter.filter_documents(
                    reranked_docs,
                    strategy=self.reranking_config.cutoff_strategy
                )
                reranked_docs = filtered_docs

                # Log cutoff results for analysis
                print(f"Autocut applied: kept {cutoff_result.kept_count}, removed {cutoff_result.removed_count}")

            except Exception as e:
                print(f"Autocut filtering failed: {e}, using all reranked documents")

        # Convert back to original format (simple conversion for compatibility)
        return reranked_docs[:self.reranking_config.max_documents]

    async def _calculate_confidence(
        self, context: QueryContext, retrieval_results, answer: str
    ) -> float:
        """Calculate confidence score for the generated answer."""
        if not retrieval_results:
            return 0.0

        # Simple confidence calculation based on retrieval scores
        avg_retrieval_score = sum(r.score for r in retrieval_results) / len(
            retrieval_results
        )

        # Factor in number of good quality results
        high_quality_results = sum(1 for r in retrieval_results if r.score > 0.8)
        coverage_score = min(high_quality_results / 3, 1.0)

        # Combine scores
        confidence = (avg_retrieval_score * 0.7) + (coverage_score * 0.3)

        return min(confidence, 1.0)

    async def get_document_stats(self) -> Dict:
        """Get statistics about the document collection."""
        return await self.document_store.get_stats()

    async def clear_documents(self) -> None:
        """Clear all documents from the system."""
        await self.document_store.clear()

    async def remove_document(self, document_id: str) -> bool:
        """Remove a specific document."""
        return await self.document_store.remove_document(document_id)