# Multi-stage Docker build for RAGents

# Development stage
FROM python:3.11-slim as development

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --dev

# Copy source code
COPY . .

# Install the package in development mode
RUN uv pip install -e .

# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Install system dependencies for production
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install only production dependencies
RUN uv sync --no-dev

# Copy source code
COPY ragents/ ragents/
COPY main.py ./

# Install the package
RUN uv pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash ragents
USER ragents

# Set environment variables
ENV PYTHONPATH="/app"
ENV RAGENTS_WORKING_DIR="/app/data"

# Create data directory
RUN mkdir -p /app/data /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "ragents.server"]

# GPU stage for vision models
FROM production as gpu

USER root

# Install CUDA dependencies
RUN apt-get update && apt-get install -y \
    nvidia-driver-470 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install vision dependencies
RUN uv pip install "ragents[vision]"

USER ragents