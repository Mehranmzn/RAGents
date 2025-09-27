# RAG Engine Overview

The RAG (Retrieval-Augmented Generation) Engine is the core component responsible for document processing, knowledge retrieval, and context-aware response generation in RAGents.

## Architecture Overview

The RAG Engine consists of several integrated components:

- **Document Processing** - Multimodal document ingestion and chunking
- **Vector Storage** - Pluggable vector database backends
- **Retrieval Engine** - Semantic search and context retrieval
- **Reranking** - Advanced result reranking and filtering
- **Response Generation** - Context-aware answer synthesis

## Key Features

- **Multimodal Processing** - Handle text, images, PDFs, tables, and more
- **Pluggable Vector Stores** - ChromaDB, Weaviate, pgvector, Elasticsearch
- **Hybrid Search** - Combine semantic and keyword search
- **Intelligent Chunking** - Context-aware document segmentation
- **Autocut Filtering** - Smart result filtering based on relevance
- **Batch Processing** - Efficient bulk document handling
- **Caching** - Intelligent result and embedding caching

## Basic Usage

### Simple RAG Setup

```python
import asyncio
from ragents import RAGEngine, RAGConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def basic_rag_example():
    # Initialize configuration
    rag_config = RAGConfig.from_env()
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create RAG engine
    rag_engine = RAGEngine(rag_config, llm_client)

    # Add documents
    await rag_engine.add_document("path/to/document.pdf")
    await rag_engine.add_text("Additional context information")

    # Query the knowledge base
    response = await rag_engine.query("What are the main findings?")
    print(response.answer)
    print(f"Sources: {response.sources}")

asyncio.run(basic_rag_example())
```

### RAG with Custom Vector Store

```python
from ragents.vector_stores import create_vector_store, VectorStoreConfig, VectorStoreType

async def custom_vector_store_rag():
    # Configure vector store
    vector_config = VectorStoreConfig(
        store_type=VectorStoreType.WEAVIATE,
        url="http://localhost:8080",
        collection_name="my_knowledge_base"
    )
    vector_store = create_vector_store(vector_config)

    # Configure RAG with custom vector store
    rag_config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        vector_store=vector_store
    )

    rag_engine = RAGEngine(rag_config, llm_client)

    # Process documents
    documents = [
        "research_paper.pdf",
        "technical_manual.docx",
        "presentation.pptx"
    ]

    for doc in documents:
        result = await rag_engine.add_document(doc)
        print(f"Processed {doc}: {result.chunks_created} chunks")

    # Query with enhanced retrieval
    response = await rag_engine.query(
        "Compare the methodologies across the documents",
        retrieval_strategy="hybrid",
        rerank_results=True
    )

    print(response.answer)
```

## Document Processing

### Supported Formats

The RAG Engine supports various document formats:

- **Text Files** - .txt, .md, .csv
- **Office Documents** - .pdf, .docx, .pptx, .xlsx
- **Images** - .png, .jpg, .jpeg (with OCR)
- **Web Content** - HTML, URLs
- **Code Files** - .py, .js, .java, .cpp, etc.
- **Structured Data** - JSON, XML, YAML

### Document Ingestion

```python
# Single document
result = await rag_engine.add_document(
    "document.pdf",
    metadata={"author": "John Doe", "category": "research"},
    chunk_strategy="semantic"  # or "fixed", "paragraph", "sentence"
)

# Batch processing
documents = [
    {"path": "doc1.pdf", "metadata": {"type": "manual"}},
    {"path": "doc2.docx", "metadata": {"type": "report"}},
]

results = await rag_engine.add_documents_batch(
    documents,
    batch_size=5,
    show_progress=True
)

# URL processing
web_result = await rag_engine.add_url(
    "https://example.com/article",
    extract_links=True,
    max_depth=2
)

# Direct text
text_result = await rag_engine.add_text(
    "Important information to remember",
    metadata={"source": "user_input", "timestamp": "2024-01-01"}
)
```

### Chunking Strategies

```python
from ragents.rag.chunking import ChunkingStrategy

# Semantic chunking (recommended)
rag_config = RAGConfig(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=1000,
    chunk_overlap=200,
    semantic_similarity_threshold=0.8
)

# Fixed-size chunking
rag_config = RAGConfig(
    chunking_strategy=ChunkingStrategy.FIXED,
    chunk_size=512,
    chunk_overlap=50
)

# Paragraph-based chunking
rag_config = RAGConfig(
    chunking_strategy=ChunkingStrategy.PARAGRAPH,
    min_chunk_size=100,
    max_chunk_size=2000
)

# Custom chunking
from ragents.rag.chunking import CustomChunker

class CustomChunker:
    async def chunk_text(self, text: str, metadata: dict) -> list:
        # Your custom chunking logic
        chunks = []
        # ... implementation
        return chunks

rag_config = RAGConfig(custom_chunker=CustomChunker())
```

## Retrieval Strategies

### Basic Retrieval

```python
# Simple similarity search
results = await rag_engine.retrieve(
    query="machine learning algorithms",
    top_k=5
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content[:200]}...")
    print(f"Metadata: {result.metadata}")
```

### Advanced Retrieval

```python
# Hybrid search (semantic + keyword)
results = await rag_engine.retrieve(
    query="deep learning optimization techniques",
    strategy="hybrid",
    top_k=10,
    keyword_weight=0.3,
    semantic_weight=0.7
)

# MMR (Maximal Marginal Relevance) for diversity
results = await rag_engine.retrieve(
    query="neural network architectures",
    strategy="mmr",
    top_k=8,
    diversity_lambda=0.5
)

# Filtered retrieval
results = await rag_engine.retrieve(
    query="computer vision",
    top_k=5,
    filters={
        "metadata.category": "research",
        "metadata.year": {"$gte": 2020}
    }
)
```

### Query Expansion

```python
# Enable query expansion
rag_config = RAGConfig(
    enable_query_expansion=True,
    expansion_strategy="synonym",  # or "related_terms", "llm_expansion"
    max_expanded_terms=5
)

# Custom query expansion
from ragents.query_rewriting import QueryExpander

class CustomQueryExpander(QueryExpander):
    async def expand_query(self, query: str) -> str:
        # Your custom expansion logic
        expanded = f"{query} OR related_terms OR synonyms"
        return expanded

rag_config = RAGConfig(query_expander=CustomQueryExpander())
```

## Response Generation

### Basic Generation

```python
# Generate response with retrieved context
response = await rag_engine.query(
    "What are the benefits of renewable energy?",
    include_sources=True,
    max_tokens=500
)

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)} documents used")
print(f"Confidence: {response.confidence}")
```

### Advanced Generation

```python
# Custom generation with templates
response = await rag_engine.query(
    "Summarize the main points",
    response_template="""
    Based on the retrieved information:

    **Main Points:**
    {main_points}

    **Supporting Evidence:**
    {evidence}

    **Conclusion:**
    {conclusion}

    **Sources:** {sources}
    """,
    structured_output=True
)

# Streaming responses
async for chunk in rag_engine.query_stream(
    "Explain quantum computing",
    chunk_size=50
):
    print(chunk, end="", flush=True)
```

### Citation and Attribution

```python
# Automatic citation generation
rag_config = RAGConfig(
    enable_citations=True,
    citation_style="apa",  # or "mla", "chicago", "numbered"
    min_citation_confidence=0.7
)

response = await rag_engine.query(
    "What is the impact of climate change?",
    include_citations=True
)

print(response.answer)
print("\n**Citations:**")
for citation in response.citations:
    print(f"[{citation.id}] {citation.formatted_citation}")
```

## Multimodal Processing

### Image Processing

```python
# Enable vision processing
rag_config = RAGConfig(
    enable_vision=True,
    vision_model="gpt-4-vision",  # or "claude-3-vision"
    ocr_enabled=True
)

# Process images
image_result = await rag_engine.add_document(
    "chart.png",
    document_type="image",
    extract_text=True,
    describe_image=True
)

# Query with image context
response = await rag_engine.query(
    "What trends are shown in the chart?",
    include_visual_context=True
)
```

### Table Processing

```python
# Process tables and spreadsheets
table_result = await rag_engine.add_document(
    "data.xlsx",
    table_processing=True,
    extract_column_headers=True,
    create_table_summaries=True
)

# Query tabular data
response = await rag_engine.query(
    "What is the average sales figure?",
    table_aware=True
)
```

## Performance Optimization

### Caching

```python
# Enable intelligent caching
rag_config = RAGConfig(
    enable_caching=True,
    cache_embeddings=True,
    cache_responses=True,
    cache_ttl=3600,  # 1 hour
    cache_strategy="lru"  # or "lfu", "ttl"
)

# Precompute embeddings
await rag_engine.precompute_embeddings(
    documents=["doc1.pdf", "doc2.pdf"],
    batch_size=10
)
```

### Batch Processing

```python
# Efficient batch processing
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

results = await rag_engine.add_documents_batch(
    documents,
    batch_size=5,
    parallel_processing=True,
    max_workers=4
)

# Batch queries
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "Define artificial intelligence"
]

responses = await rag_engine.query_batch(
    queries,
    batch_size=3,
    deduplicate_context=True
)
```

### Memory Management

```python
# Configure memory usage
rag_config = RAGConfig(
    max_memory_usage="2GB",
    chunk_batch_size=100,
    embedding_batch_size=50,
    cleanup_interval=3600  # seconds
)

# Monitor memory usage
stats = await rag_engine.get_memory_stats()
print(f"Memory usage: {stats.memory_usage_mb}MB")
print(f"Cached embeddings: {stats.cached_embeddings}")
print(f"Active chunks: {stats.active_chunks}")
```

## Monitoring and Analytics

### Performance Metrics

```python
from ragents.observability import RAGMetrics

metrics = RAGMetrics(rag_engine)

# Get retrieval statistics
retrieval_stats = metrics.get_retrieval_stats()
print(f"Average retrieval time: {retrieval_stats.avg_time}ms")
print(f"Cache hit rate: {retrieval_stats.cache_hit_rate}")
print(f"Average relevance score: {retrieval_stats.avg_relevance}")

# Get document statistics
doc_stats = metrics.get_document_stats()
print(f"Total documents: {doc_stats.total_documents}")
print(f"Total chunks: {doc_stats.total_chunks}")
print(f"Average chunk size: {doc_stats.avg_chunk_size}")
```

### Query Analytics

```python
# Track query patterns
query_analytics = metrics.get_query_analytics()
print(f"Most common queries: {query_analytics.popular_queries}")
print(f"Query performance: {query_analytics.performance_trends}")

# Custom metrics
@metrics.track_metric("custom_retrieval_quality")
async def track_retrieval_quality(query: str, results: list):
    # Your custom quality assessment
    quality_score = assess_result_quality(results)
    return {"quality": quality_score, "query_length": len(query)}
```

## Error Handling

### Graceful Degradation

```python
try:
    response = await rag_engine.query("Complex query")
except DocumentProcessingError as e:
    print(f"Document processing failed: {e}")
    # Fallback to basic LLM response
    fallback_response = await llm_client.complete(query)

except RetrievalError as e:
    print(f"Retrieval failed: {e}")
    # Try with different strategy
    response = await rag_engine.query(
        query,
        strategy="keyword_only",
        fallback_to_llm=True
    )
```

### Robust Configuration

```python
rag_config = RAGConfig(
    # Retry configuration
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True,

    # Fallback options
    fallback_to_llm=True,
    fallback_vector_store="memory",
    fallback_embedding_model="sentence-transformers",

    # Validation
    validate_documents=True,
    check_document_size=True,
    max_document_size="10MB"
)
```

## Best Practices

### Document Organization

1. **Consistent Metadata** - Use structured metadata across documents
2. **Logical Grouping** - Organize documents by topic or source
3. **Regular Updates** - Keep knowledge base current
4. **Quality Control** - Validate document quality before ingestion

### Retrieval Optimization

1. **Right Chunk Size** - Balance context and relevance
2. **Appropriate Top-K** - Don't retrieve too many irrelevant results
3. **Use Filters** - Narrow down search when possible
4. **Monitor Performance** - Track retrieval quality metrics

### Response Quality

1. **Context Validation** - Ensure retrieved context is relevant
2. **Citation Accuracy** - Verify source attribution
3. **Response Consistency** - Test with similar queries
4. **User Feedback** - Collect and incorporate user feedback

## Next Steps

- **[Document Processing](document-processing.md)** - Deep dive into document handling
- **[Vector Stores](vector-stores.md)** - Explore vector database options
- **[Retrieval Strategies](retrieval.md)** - Advanced retrieval techniques
- **[Reranking](reranking.md)** - Improve result relevance