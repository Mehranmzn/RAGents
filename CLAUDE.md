# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGents is an advanced agentic RAG framework written in Python 3.10+. It combines intelligent decision-tree agents with multimodal RAG capabilities and type-safe LLM interactions using the instructor package.

## Commands

### Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Run the main demo
python main.py

# Run examples
python examples/basic_usage.py

# Run tests
pytest

# Format code
black ragents/
ruff check ragents/

# Type checking
mypy ragents/
```

### Package Installation
```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[weaviate,vision]"
```

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Core Components
- `ragents/agents/`: Agent implementations with decision tree capabilities
- `ragents/rag/`: RAG engine with multimodal processing
- `ragents/llm/`: Type-safe LLM client using instructor for structured outputs
- `ragents/config/`: Environment-driven configuration management
- `ragents/processors/`: Document and content processors

### Key Patterns
- **Async-first design**: All main operations support async/await
- **Type safety**: Extensive use of Pydantic models and instructor for LLM interactions
- **Mixin architecture**: Extensible agent capabilities through mixins
- **Environment configuration**: Configuration through environment variables with fallbacks
- **Structured outputs**: All LLM interactions use typed responses

### Dependencies
- `instructor`: Type-safe LLM interactions
- `pydantic`: Data validation and settings management
- `openai`/`anthropic`: LLM provider clients
- `chromadb`: Vector database for embeddings
- `sentence-transformers`: Text embeddings

## Development Guidelines

### Adding New Features
1. Follow the existing async patterns
2. Use Pydantic models for data validation
3. Add appropriate type hints
4. Create corresponding test files
5. Update configuration classes if needed

### Agent Development
- Extend `Agent` base class or use `DecisionTreeAgent`
- Define decision nodes for complex behaviors
- Use structured thinking patterns with instructor

### LLM Integration
- Always use the `LLMClient` with structured outputs
- Define Pydantic models for expected responses
- Implement retry logic for robustness

### Configuration
- Add new settings to appropriate config classes
- Support environment variable overrides
- Provide sensible defaults

## Environment Variables

Required:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: LLM provider authentication

Optional configuration:
- `RAGENTS_LLM_PROVIDER`: openai/anthropic (default: openai)
- `RAGENTS_CHUNK_SIZE`: Document chunk size (default: 1000)
- `RAGENTS_TOP_K`: Retrieval results count (default: 5)
- `RAGENTS_ENABLE_VISION`: Enable image processing (default: false)
- `RAGENTS_WORKING_DIR`: Working directory (default: ./output)