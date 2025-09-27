<div align="center">
  <img src="assets/logo.svg" alt="RAGents Logo" width="200" height="200" />
  <h1>RAGents 🤖</h1>
  <p><strong>Production-Ready Agentic RAG Framework with Intelligent Decision Trees and Logic-LLM Integration</strong></p>

  <p>
    <a href="https://pypi.org/project/ragents"><img alt="PyPI" src="https://img.shields.io/pypi/v/ragents?color=blue"></a>
    <a href="https://github.com/yourusername/ragents/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/yourusername/ragents?color=green"></a>
    <a href="https://github.com/yourusername/ragents"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/yourusername/ragents?style=social"></a>
    <a href="https://codecov.io/gh/yourusername/ragents"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/yourusername/ragents"></a>
  </p>

  <p>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
    <a href="https://github.com/jxnl/instructor"><img alt="Instructor" src="https://img.shields.io/badge/powered%20by-instructor-orange"></a>
    <a href="https://docs.pydantic.dev/"><img alt="Pydantic" src="https://img.shields.io/badge/typed%20with-pydantic-purple"></a>
    <a href="https://github.com/chroma-core/chroma"><img alt="ChromaDB" src="https://img.shields.io/badge/vectors-chromadb-red"></a>
  </p>

  <p>
    <img alt="Async" src="https://img.shields.io/badge/async-first-green">
    <img alt="Multimodal" src="https://img.shields.io/badge/multimodal-rag-blueviolet">
    <img alt="Type Safe" src="https://img.shields.io/badge/type-safe-brightgreen">
    <img alt="Logic LLM" src="https://img.shields.io/badge/logic--llm-enabled-yellow">
    <img alt="Observability" src="https://img.shields.io/badge/observability-ready-lightblue">
    <img alt="Production" src="https://img.shields.io/badge/production-ready-success">
  </p>

  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="#core-features">Features</a> •
    <a href="#intelligent-agents">Agents</a> •
    <a href="#examples">Examples</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>

## 🔎 Overview

RAGents represents the next generation of Retrieval-Augmented Generation frameworks, specifically designed for production environments where intelligent reasoning, multimodal processing, and enterprise-grade reliability are paramount. Unlike traditional RAG systems that rely on simple retrieval patterns, RAGents introduces sophisticated agent architectures that can reason through complex problems, make decisions based on configurable logic trees, and optimize their own performance through advanced techniques.

The framework distinguishes itself through its integration of Logic-LLM capabilities, which dramatically reduce token consumption while improving reasoning accuracy. This approach combines neural language models with symbolic reasoning, enabling the system to solve mathematical problems, handle logical constraints, and perform step-by-step deduction with significantly lower computational overhead than traditional approaches.

RAGents provides multiple agent architectures tailored for different use cases. The Decision Tree Agent excels in scenarios requiring structured, rule-based reasoning, making it ideal for customer support, content moderation, and workflow automation. The ReAct Agent implements the Reasoning plus Acting pattern, perfect for research tasks and complex problem-solving that requires iterative tool usage. The Graph Planner Agent brings sophisticated project management capabilities, enabling complex task orchestration with dependency resolution and resource optimization.

Built with enterprise requirements in mind, RAGents features comprehensive observability through OpenInference protocol integration, enabling detailed tracing and monitoring of agent behavior in production environments. The framework supports multiple vector database backends including ChromaDB, Weaviate, PostgreSQL with pgvector, and Elasticsearch, allowing organizations to choose the storage solution that best fits their scalability and performance requirements.

## 🌟 Core Features

### Intelligent Agent Architecture

RAGents implements three distinct agent types, each optimized for specific reasoning patterns and use cases. The Decision Tree Agent provides configurable decision logic that can be customized for domain-specific workflows, allowing organizations to encode their business rules directly into the agent's reasoning process. This agent type excels in scenarios where consistent, rule-based responses are required, such as customer service automation or compliance checking.

The ReAct Agent brings sophisticated iterative reasoning capabilities, combining step-by-step analysis with dynamic tool selection. This agent can break down complex problems into manageable steps, execute appropriate tools based on context, and adjust its approach based on intermediate results. The implementation includes self-correction mechanisms and transparent reasoning traces, making it particularly valuable for research assistance and analytical tasks.

The Graph Planner Agent introduces advanced project management capabilities through graph-based task modeling. It can analyze complex requirements, identify dependencies between subtasks, optimize resource allocation, and generate executable plans with built-in monitoring and adaptation capabilities. This agent type is especially powerful for workflow automation and multi-step process orchestration.

### Logic-LLM Integration for Enhanced Reasoning

The framework incorporates a sophisticated Logic-LLM module based on the Logic-LLM research, which fundamentally changes how language models approach reasoning tasks. This system identifies logical patterns in queries and applies symbolic reasoning techniques to solve problems more efficiently than traditional neural approaches alone. For mathematical calculations, logical deductions, and constraint satisfaction problems, this approach can reduce token usage by 60-80% while improving accuracy.

The Logic-LLM implementation includes specialized components for different reasoning types. The Constraint Engine handles rule-based validation and satisfaction problems, while the Symbolic Solver manages mathematical computations and algebraic manipulations. The Query Clarifier provides interactive refinement capabilities, helping users formulate clearer, more answerable questions through guided clarification processes.

Pattern recognition capabilities enable the system to identify recurring logical structures in queries and apply appropriate reasoning templates. This includes support for common patterns like modus ponens, syllogistic reasoning, and causal chains, allowing the system to handle complex logical arguments with mathematical precision.

### Advanced RAG Engine with Multimodal Processing

The RAG engine represents a significant advancement over traditional retrieval systems, offering true multimodal document processing capabilities. The system can intelligently process PDFs with embedded images, extract data from tables and charts, perform OCR on image documents, and maintain semantic relationships between different content types within the same document.

The chunking strategy goes beyond simple text splitting to implement semantic-aware segmentation that preserves context boundaries and maintains logical document structure. The system supports multiple retrieval strategies including semantic similarity search, Maximal Marginal Relevance for diversity, and hybrid approaches that combine dense and sparse retrieval methods.

Query processing includes sophisticated expansion and rewriting capabilities that can reformulate ambiguous queries into more precise search terms. The system maintains query history and can perform contextual rewriting based on conversation flow, significantly improving retrieval accuracy for follow-up questions and complex information requests.

### Production-Grade Vector Database Support

RAGents provides native integration with multiple vector database backends, each optimized for different deployment scenarios. ChromaDB integration offers excellent development and prototyping capabilities with its embedded mode, while also supporting client-server deployments for production environments. The Weaviate integration leverages advanced schema management and hybrid search capabilities for enterprise-scale deployments.

PostgreSQL integration through pgvector enables organizations to leverage existing database infrastructure while adding vector search capabilities. This approach is particularly valuable for enterprises with significant PostgreSQL investments who want to add AI capabilities without introducing new database technologies. The Elasticsearch integration provides powerful hybrid search combining traditional text search with vector similarity.

Each vector store integration includes comprehensive configuration options for index optimization, batch processing, and performance tuning. The system automatically handles connection pooling, retry logic, and failover scenarios to ensure reliable operation in production environments.

### Enterprise Observability and Monitoring

The framework includes comprehensive observability capabilities designed for production monitoring and debugging. OpenInference protocol integration provides detailed tracing of agent decision-making processes, RAG retrieval operations, and LLM interactions. This enables operators to understand exactly how the system arrives at specific responses and identify potential issues or optimization opportunities.

Structured logging provides tamper-proof audit trails of agent reasoning, including decision rationales, confidence scores, and processing times. The system generates detailed execution traces that can be analyzed for performance optimization or used for compliance and auditing purposes in regulated industries.

Performance metrics collection includes detailed statistics on response times, success rates, token usage, and resource utilization. The system provides Prometheus-compatible metrics endpoints and supports custom metric collection for domain-specific monitoring requirements.

### Type-Safe Development with Instructor Integration

RAGents leverages the Instructor library to provide type-safe interactions with language models, ensuring that all LLM responses conform to predefined Pydantic schemas. This approach eliminates the parsing errors and inconsistencies common in traditional LLM applications, while providing automatic validation and error handling for structured outputs.

The type system extends throughout the framework, with comprehensive Pydantic models for configuration, agent responses, document metadata, and evaluation results. This design enables robust error handling, automatic API documentation generation, and seamless integration with modern Python development workflows.

Response validation includes multiple fallback strategies for handling edge cases where LLM outputs don't conform to expected schemas. The system can automatically retry with refined prompts, apply heuristic corrections, or gracefully degrade to alternative response formats while maintaining type safety.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragents.git
cd ragents

# Basic installation
pip install -e .

# With all features including vector stores and observability
pip install -e ".[all]"

# Specific feature sets
pip install -e ".[weaviate,vision,evaluation,observability]"
```

### Environment Setup

```bash
# Required: Set your LLM provider API key
export OPENAI_API_KEY="your-openai-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: Configure RAG settings
export RAGENTS_LLM_PROVIDER="openai"  # or "anthropic"
export RAGENTS_VECTOR_STORE_TYPE="chromadb"  # or "weaviate", "pgvector", "elasticsearch"
export RAGENTS_CHUNK_SIZE="1000"
export RAGENTS_TOP_K="5"
export RAGENTS_ENABLE_VISION="true"
```

### Basic Usage

```python
import asyncio
from ragents import (
    AgentConfig, DecisionTreeAgent, RAGConfig, RAGEngine,
    create_vector_store, VectorStoreConfig, VectorStoreType
)
from ragents.config.environment import get_llm_config_from_env
from ragents.llm.client import LLMClient

async def main():
    # Initialize configuration
    rag_config = RAGConfig.from_env()
    llm_config = get_llm_config_from_env()

    # Create LLM client with type safety
    llm_client = LLMClient(llm_config)

    # Set up vector store
    vector_config = VectorStoreConfig(
        store_type=VectorStoreType.CHROMADB,
        collection_name="my_documents"
    )
    vector_store = create_vector_store(vector_config)

    # Set up RAG engine
    rag_engine = RAGEngine(rag_config, llm_client)

    # Create intelligent agent
    agent_config = AgentConfig(
        name="My Assistant",
        enable_rag=True,
        enable_reasoning=True,
        enable_logical_reasoning=True
    )

    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Add documents to knowledge base
    await rag_engine.add_document("path/to/document.pdf")

    # Interact with the agent
    response = await agent.process_message("What are the key findings in the document?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 🎬 Live Demo

```bash
# Run the interactive demo
python main.py

# Or try the examples
python examples/basic_usage.py
python examples/multimodal_rag.py
python examples/logical_llm_demo.py
```

The interactive demo showcases real-time RAG capabilities with document upload, multimodal processing including image analysis, agent comparison across different reasoning approaches, live evaluation metrics, and comprehensive observability features including distributed tracing visualization.

### ⚡ Logical LLM Integration

RAGents includes a powerful Logical LLM module based on Logic-LLM research that enhances reasoning capabilities while reducing token usage through symbolic reasoning and constraint satisfaction. The system identifies logical patterns in queries and applies appropriate symbolic reasoning techniques, dramatically reducing computational overhead for mathematical problems, logical deductions, and constraint satisfaction tasks.

The implementation includes specialized components for different reasoning types including constraint engines for rule-based validation, symbolic solvers for mathematical computations, query clarifiers for interactive refinement, and pattern matchers for logical structure recognition. This integration enables the system to handle complex logical arguments with mathematical precision while maintaining the flexibility of neural language models.

## 📖 Advanced Usage

### Multiple Agent Types

```python
from ragents import DecisionTreeAgent, GraphPlannerAgent, ReActAgent

# Decision Tree Agent for structured reasoning
dt_agent = DecisionTreeAgent(config, llm_client, rag_engine)

# Graph Planner Agent for complex task orchestration
gp_agent = GraphPlannerAgent(config, llm_client, rag_engine)

# ReAct Agent for iterative problem solving
react_agent = ReActAgent(config, llm_client, rag_engine)
```

### Vector Store Backends

```python
from ragents import create_vector_store, VectorStoreConfig, VectorStoreType

# ChromaDB for development and small-scale deployment
chroma_config = VectorStoreConfig(
    store_type=VectorStoreType.CHROMADB,
    persist_directory="./chroma_db"
)

# Weaviate for enterprise-scale deployment
weaviate_config = VectorStoreConfig(
    store_type=VectorStoreType.WEAVIATE,
    url="https://your-cluster.weaviate.network",
    api_key="your-api-key"
)

# PostgreSQL with pgvector for existing database integration
pgvector_config = VectorStoreConfig(
    store_type=VectorStoreType.PGVECTOR,
    database_url="postgresql://user:pass@localhost/db"
)

vector_store = create_vector_store(config)
```

### Evaluation Framework

```python
from ragents import RAGEvaluator, create_sample_dataset

# Create evaluator with RAGAS-style metrics
evaluator = RAGEvaluator(llm_client, rag_engine)

# Use built-in datasets or create custom ones
dataset = create_sample_dataset("science")

# Evaluate RAG system performance
results = await evaluator.evaluate_batch(dataset.data_points)

# Generate comprehensive evaluation report
report = evaluator.create_evaluation_report(results)
print(report)
```

## 🗺️ Roadmap

### ✅ Current Version (v0.1.0)

RAGents v0.1.0 provides a comprehensive foundation with Decision Tree, Graph Planner, and ReAct agent implementations. The framework includes robust support for ChromaDB, Weaviate, pgvector, and Elasticsearch vector databases, along with type-safe LLM interactions through Instructor integration. Logic-LLM capabilities are fully integrated for token optimization and enhanced reasoning performance.

The current release features complete RAGAS-style evaluation metrics for assessing system performance, OpenInference observability integration for production monitoring, and comprehensive multimodal document processing capabilities. The system includes sophisticated query rewriting and logical reasoning modules that significantly improve both efficiency and accuracy compared to traditional RAG implementations.

### 🚧 Next Release (v0.2.0)

The upcoming release will focus on advanced multi-agent collaboration systems that enable multiple specialized agents to work together on complex tasks. This includes sophisticated communication protocols, task delegation mechanisms, and result synthesis capabilities that allow organizations to deploy agent teams for comprehensive problem-solving scenarios.

Distributed RAG capabilities will enable deployment across multiple nodes with intelligent load balancing and data consistency management. The system will support edge deployment scenarios with optimized models and caching strategies designed for resource-constrained environments while maintaining full functionality.

Enhanced security and privacy features will include advanced encryption for data at rest and in transit, comprehensive audit logging, and privacy-preserving techniques for sensitive document processing. Cloud-native deployment options will provide seamless integration with major cloud platforms including automated scaling, managed service integration, and comprehensive monitoring dashboards.

### Observability

```python
from ragents.observability import get_tracer, setup_openinference_tracing

# Setup distributed tracing
tracer = get_tracer()
openinference = setup_openinference_tracing(tracer)

# Instrument components for comprehensive monitoring
instrumented_llm = openinference.instrument_llm_client(llm_client)
instrumented_rag = openinference.instrument_rag_engine(rag_engine)

# Manual tracing for custom operations
with tracer.trace("my_operation") as trace:
    with tracer.span("processing", SpanType.DOCUMENT_PROCESSING) as span:
        # Your code here with automatic performance tracking
        if span:
            span.add_tag("documents", 5)
```

## 🔧 Configuration

RAGents provides comprehensive configuration management through environment variables, configuration files, and programmatic interfaces. The system supports different deployment profiles for development, staging, and production environments with appropriate defaults and validation for each scenario.

Vector store configuration includes optimization parameters for different backends, enabling fine-tuning of performance characteristics based on specific deployment requirements. The LLM provider configuration supports multiple models and providers with automatic fallback and load balancing capabilities for high-availability deployments.

Observability configuration enables detailed control over tracing, logging, and metrics collection with support for different monitoring backends and custom metric definitions. Security configuration includes comprehensive options for authentication, authorization, encryption, and audit logging to meet enterprise compliance requirements.

## 🤝 Contributing

We welcome contributions to RAGents from the community. The project follows standard open-source contribution practices with comprehensive documentation for developers interested in extending the framework or contributing improvements.

Development setup includes complete testing infrastructure with automated test suites, code quality tools including Black and Ruff for formatting, MyPy for type checking, and comprehensive pre-commit hooks to ensure code quality. The project uses modern Python development practices with full async support and extensive type annotations.

Contributors can focus on various areas including new agent types, additional vector store integrations, enhanced reasoning algorithms, improved evaluation metrics, or deployment tooling. The modular architecture makes it straightforward to add new capabilities while maintaining backward compatibility and system stability.

## 📄 License & Acknowledgments

This project is licensed under the BSD-3-Clause License, providing flexibility for both open-source and commercial usage while ensuring proper attribution and liability protection.

### Acknowledgments

RAGents builds upon excellent research and open-source work from the broader AI community. The Elysia rag project was our first insipration for the agent trees. The Logic-LLM integration is inspired by research from Logic-LLM project, implemented independently to provide similar capabilities while respecting the original BSD-3 license. The framework incorporates multimodal processing patterns inspired by RAG-Anything research, providing comprehensive document handling capabilities.

The system is powered by Instructor for type-safe LLM interactions, ensuring robust and reliable language model integration. Observability capabilities are designed to be compatible with OpenInference standards, enabling seamless integration with existing monitoring infrastructure. All external patterns and concepts are implemented independently to respect original licenses while providing enhanced functionality tailored for production environments.

## 🔗 Links

- [Documentation](https://ragents.readthedocs.io) *(coming soon)*
- [PyPI Package](https://pypi.org/project/ragents) *(coming soon)*
- [GitHub Repository](https://github.com/yourusername/ragents)
- [Issues](https://github.com/yourusername/ragents/issues)

---

**RAGents** - Build intelligent agents that think, reason, and learn with production-grade reliability and performance. 🤖✨