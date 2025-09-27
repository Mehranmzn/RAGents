# Configuration

RAGents uses environment-driven configuration with sensible defaults. This guide covers all configuration options and how to customize your setup.

## Configuration Overview

RAGents follows a hierarchical configuration approach:

1. **Environment Variables** - Primary configuration method
2. **Configuration Files** - For complex setups
3. **Runtime Configuration** - Programmatic overrides
4. **Defaults** - Fallback values

## Environment Variables

### Required Configuration

```bash
# LLM Provider (choose one)
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Core RAG Settings

```bash
# LLM Provider Selection
export RAGENTS_LLM_PROVIDER="openai"          # Options: openai, anthropic

# Vector Store Configuration
export RAGENTS_VECTOR_STORE_TYPE="chromadb"   # Options: chromadb, weaviate, pgvector, elasticsearch

# Document Processing
export RAGENTS_CHUNK_SIZE="1000"              # Document chunk size in characters
export RAGENTS_CHUNK_OVERLAP="200"            # Overlap between chunks
export RAGENTS_ENABLE_VISION="false"          # Enable image processing

# Retrieval Settings
export RAGENTS_TOP_K="5"                      # Number of documents to retrieve
export RAGENTS_RETRIEVAL_STRATEGY="hybrid"    # Options: similarity, mmr, hybrid

# Performance
export RAGENTS_ENABLE_CACHING="true"          # Enable result caching
export RAGENTS_BATCH_SIZE="10"                # Batch size for processing

# Working Directories
export RAGENTS_WORKING_DIR="./output"         # Working directory for outputs
```

### Advanced Settings

```bash
# Model Configuration
export RAGENTS_MODEL_NAME="gpt-4"             # Default model name
export RAGENTS_TEMPERATURE="0.1"              # Model temperature
export RAGENTS_MAX_TOKENS="2048"              # Maximum tokens per response

# Logical LLM
export RAGENTS_ENABLE_LOGICAL_REASONING="true" # Enable Logic-LLM integration
export RAGENTS_CONSTRAINT_TIMEOUT="30"         # Constraint solving timeout

# Observability
export RAGENTS_ENABLE_TRACING="true"          # Enable distributed tracing
export RAGENTS_TRACE_LEVEL="info"             # Trace level: debug, info, warn, error

# Query Rewriting
export RAGENTS_ENABLE_QUERY_REWRITING="true"  # Enable query optimization
export RAGENTS_REWRITING_STRATEGY="contextual" # Options: cot, few_shot, contextual

# Evaluation
export RAGENTS_ENABLE_EVALUATION="false"      # Enable automatic evaluation
export RAGENTS_EVALUATION_METRICS="faithfulness,relevance" # Comma-separated metrics
```

## Vector Store Configuration

### ChromaDB (Default)

```bash
export RAGENTS_VECTOR_STORE_TYPE="chromadb"
export RAGENTS_CHROMADB_PATH="./chroma_db"    # Local storage path
export RAGENTS_CHROMADB_HOST="localhost"      # For client mode
export RAGENTS_CHROMADB_PORT="8000"           # For client mode
```

### Weaviate

```bash
export RAGENTS_VECTOR_STORE_TYPE="weaviate"
export RAGENTS_WEAVIATE_URL="http://localhost:8080"
export RAGENTS_WEAVIATE_API_KEY="your-api-key"      # For cloud instances
export RAGENTS_WEAVIATE_ADDITIONAL_HEADERS='{"X-Custom": "value"}'
```

### PostgreSQL with pgvector

```bash
export RAGENTS_VECTOR_STORE_TYPE="pgvector"
export RAGENTS_PGVECTOR_HOST="localhost"
export RAGENTS_PGVECTOR_PORT="5432"
export RAGENTS_PGVECTOR_DATABASE="ragents"
export RAGENTS_PGVECTOR_USER="postgres"
export RAGENTS_PGVECTOR_PASSWORD="password"
export RAGENTS_PGVECTOR_TABLE="embeddings"
```

### Elasticsearch

```bash
export RAGENTS_VECTOR_STORE_TYPE="elasticsearch"
export RAGENTS_ELASTICSEARCH_URL="http://localhost:9200"
export RAGENTS_ELASTICSEARCH_API_KEY="your-api-key"
export RAGENTS_ELASTICSEARCH_INDEX="ragents_embeddings"
```

## Programmatic Configuration

### Using Configuration Classes

```python
from ragents.config import RAGConfig, VectorStoreConfig, LLMConfig
from ragents.vector_stores import VectorStoreType

# RAG Configuration
rag_config = RAGConfig(
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5,
    enable_vision=True,
    enable_caching=True
)

# Vector Store Configuration
vector_config = VectorStoreConfig(
    store_type=VectorStoreType.CHROMADB,
    collection_name="my_collection",
    persist_directory="./vector_db"
)

# LLM Configuration
llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.1,
    max_tokens=2048
)
```

### Agent Configuration

```python
from ragents import AgentConfig

agent_config = AgentConfig(
    name="My Agent",
    description="A helpful assistant",
    enable_rag=True,
    enable_memory=True,
    enable_tools=True,
    enable_reasoning=True,
    memory_window_size=10,
    temperature=0.1
)
```

## Configuration Validation

RAGents validates all configuration at startup:

```python
from ragents.config import validate_config

# Validate current configuration
try:
    config = validate_config()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Configuration Profiles

### Development Profile

```bash
# .env.development
RAGENTS_LLM_PROVIDER=openai
RAGENTS_VECTOR_STORE_TYPE=chromadb
RAGENTS_ENABLE_CACHING=true
RAGENTS_TRACE_LEVEL=debug
RAGENTS_WORKING_DIR=./dev_output
```

### Production Profile

```bash
# .env.production
RAGENTS_LLM_PROVIDER=openai
RAGENTS_VECTOR_STORE_TYPE=weaviate
RAGENTS_ENABLE_CACHING=true
RAGENTS_ENABLE_TRACING=true
RAGENTS_TRACE_LEVEL=info
RAGENTS_WORKING_DIR=/app/output
```

### Testing Profile

```bash
# .env.testing
RAGENTS_LLM_PROVIDER=openai
RAGENTS_VECTOR_STORE_TYPE=memory
RAGENTS_ENABLE_CACHING=false
RAGENTS_TRACE_LEVEL=warn
RAGENTS_WORKING_DIR=./test_output
```

## Configuration Best Practices

### Security

1. **Never commit API keys** to version control
2. **Use environment-specific configs** for different deployments
3. **Rotate API keys regularly**
4. **Use secure credential management** in production

### Performance

1. **Enable caching** for better performance
2. **Optimize chunk size** based on your document types
3. **Choose appropriate vector stores** for your scale
4. **Monitor resource usage** with observability

### Development

1. **Use development profiles** for local work
2. **Enable debug tracing** during development
3. **Use in-memory stores** for testing
4. **Validate configurations** early

## Troubleshooting Configuration

### Common Issues

**Configuration Not Found:**
```bash
# Check if environment variables are set
env | grep RAGENTS

# Verify configuration loading
python -c "from ragents.config import RAGConfig; print(RAGConfig.from_env())"
```

**Invalid Vector Store:**
```bash
# Test vector store connection
python -c "
from ragents.vector_stores import create_vector_store, VectorStoreConfig
config = VectorStoreConfig.from_env()
store = create_vector_store(config)
print('Vector store created successfully')
"
```

**LLM Connection Issues:**
```bash
# Test LLM client
python -c "
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env
client = LLMClient(get_llm_config_from_env())
print('LLM client created successfully')
"
```

## Next Steps

- **[Quick Start](quickstart.md)** - Get started with basic usage
- **[Agents Overview](../agents/overview.md)** - Learn about agent types
- **[Deployment](../deployment/docker.md)** - Deploy RAGents to production