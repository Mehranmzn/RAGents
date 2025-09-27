# Quick Start

Get up and running with RAGents in minutes.

## Prerequisites

- Python 3.10 or higher
- One of the following API keys:
  - OpenAI API key (recommended)
  - Anthropic API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragents.git
cd ragents

# Basic installation
pip install -e .

# Or install with all features
pip install -e ".[all]"
```

## Environment Setup

Set up your environment variables:

```bash
# Required: Set your LLM provider API key
export OPENAI_API_KEY="your-openai-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: Configure RAG settings
export RAGENTS_LLM_PROVIDER="openai"  # or "anthropic"
export RAGENTS_VECTOR_STORE_TYPE="chromadb"
export RAGENTS_CHUNK_SIZE="1000"
export RAGENTS_TOP_K="5"
```

## Basic Usage

### 1. Simple Agent Chat

```python
import asyncio
from ragents import DecisionTreeAgent, AgentConfig
from ragents.config.environment import get_llm_config_from_env
from ragents.llm.client import LLMClient

async def simple_chat():
    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create agent
    agent_config = AgentConfig(name="Assistant")
    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client
    )

    # Chat with the agent
    response = await agent.process_message("Hello! What can you help me with?")
    print(response)

# Run the example
asyncio.run(simple_chat())
```

### 2. RAG-Enabled Agent

```python
import asyncio
from ragents import (
    DecisionTreeAgent, AgentConfig, RAGConfig, RAGEngine,
    create_vector_store, VectorStoreConfig, VectorStoreType
)
from ragents.config.environment import get_llm_config_from_env
from ragents.llm.client import LLMClient

async def rag_example():
    # Initialize configuration
    rag_config = RAGConfig.from_env()
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Set up vector store
    vector_config = VectorStoreConfig(
        store_type=VectorStoreType.CHROMADB,
        collection_name="my_documents"
    )
    vector_store = create_vector_store(vector_config)

    # Create RAG engine
    rag_engine = RAGEngine(rag_config, llm_client)

    # Create agent with RAG capabilities
    agent_config = AgentConfig(
        name="RAG Assistant",
        enable_rag=True,
        enable_reasoning=True
    )

    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Add a document to the knowledge base
    await rag_engine.add_document("path/to/your/document.pdf")

    # Ask questions about the document
    response = await agent.process_message("What are the key findings in the document?")
    print(response)

# Run the example
asyncio.run(rag_example())
```

### 3. Run the Interactive Demo

```bash
# Run the main interactive demo
python main.py

# Or try specific examples
python examples/basic_usage.py
python examples/logical_llm_demo.py
```

## What's Next?

- **[Configuration Guide](configuration.md)** - Learn about all configuration options
- **[Agents Overview](../agents/overview.md)** - Explore different agent types
- **[RAG Engine](../rag/overview.md)** - Understand document processing and retrieval
- **[Examples](../examples/basic.md)** - See more advanced use cases

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you installed the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

**API Key Issues:**
```bash
# Verify your API key is set
echo $OPENAI_API_KEY

# Or check if using Anthropic
echo $ANTHROPIC_API_KEY
```

**Vector Store Issues:**
```bash
# ChromaDB is the default and requires no setup
# For other stores, see the configuration guide
```

## Performance Tips

1. **Use async/await** - RAGents is built for async operations
2. **Enable caching** - Set `RAGENTS_ENABLE_CACHING=true`
3. **Optimize chunk size** - Adjust `RAGENTS_CHUNK_SIZE` based on your documents
4. **Choose the right vector store** - ChromaDB for development, others for production

Ready to build intelligent agents? Let's go! 🚀