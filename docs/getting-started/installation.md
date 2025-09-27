# Installation

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

Install RAGents using pip:

```bash
pip install ragents
```

Or using uv (recommended):

```bash
uv add ragents
```

## Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/ragents.git
cd ragents
pip install -e ".[dev]"
```

## Optional Dependencies

RAGents supports various optional features through extra dependencies:

### Vector Store Backends

```bash
# Weaviate support
pip install "ragents[weaviate]"

# PostgreSQL with pgvector
pip install "ragents[pgvector]"

# Elasticsearch
pip install "ragents[elasticsearch]"

# All vector stores
pip install "ragents[all-vector-stores]"
```

### AI/ML Extensions

```bash
# Vision capabilities (image processing)
pip install "ragents[vision]"

# Evaluation tools
pip install "ragents[evaluation]"
```

### Deployment & Observability

```bash
# Production deployment tools
pip install "ragents[deployment]"

# Observability and monitoring
pip install "ragents[observability]"
```

### Complete Installation

Install everything:

```bash
pip install "ragents[all]"
```

## Environment Variables

Set up your environment variables:

```bash
# Required: At least one LLM provider
export OPENAI_API_KEY="your-openai-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional configuration
export RAGENTS_LLM_PROVIDER="openai"  # or "anthropic"
export RAGENTS_CHUNK_SIZE="1000"
export RAGENTS_TOP_K="5"
export RAGENTS_ENABLE_VISION="false"
export RAGENTS_WORKING_DIR="./output"
```

## Verification

Verify your installation:

```bash
python -c "import ragents; print(ragents.__version__)"
```

Run the demo:

```bash
python -m ragents.demo
```

## Docker Installation

Use the official Docker image:

```bash
docker run -e OPENAI_API_KEY=your-key ragents:latest
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration](configuration.md)