# Docker Deployment

RAGents provides Docker images for easy deployment and scaling.

## Quick Start

Run RAGents in a container:

```bash
docker run -e OPENAI_API_KEY=your-key -p 8000:8000 ragents:latest
```

## Building from Source

Build the Docker image:

```bash
docker build -t ragents:local .
```

## Docker Compose

Use Docker Compose for multi-service deployments:

```yaml
version: '3.8'
services:
  ragents:
    image: ragents:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RAGENTS_LLM_PROVIDER=openai
    volumes:
      - ./data:/app/data
    depends_on:
      - chromadb
      - redis

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chromadb_data:/chroma/chroma

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  chromadb_data:
```

## Environment Variables

Configure your container with environment variables:

```bash
docker run \
  -e OPENAI_API_KEY=your-key \
  -e RAGENTS_LLM_PROVIDER=openai \
  -e RAGENTS_CHUNK_SIZE=1000 \
  -e RAGENTS_TOP_K=5 \
  -p 8000:8000 \
  ragents:latest
```

## Production Considerations

### Resource Limits

Set appropriate resource limits:

```bash
docker run \
  --memory=2g \
  --cpus=1.0 \
  -e OPENAI_API_KEY=your-key \
  ragents:latest
```

### Health Checks

The container includes health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Persistent Storage

Mount volumes for persistent data:

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e OPENAI_API_KEY=your-key \
  ragents:latest
```

## Multi-Stage Builds

The Dockerfile uses multi-stage builds for optimization:

- **Development**: Full development environment
- **Production**: Minimal runtime environment
- **GPU**: CUDA support for vision models

## Next Steps

- [Kubernetes Deployment](kubernetes.md)
- [Kubeflow Integration](kubeflow.md)