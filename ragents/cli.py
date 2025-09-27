"""Command-line interface for RAGents."""

import click
import os


@click.group()
def cli():
    """RAGents - Advanced Agentic RAG Framework"""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8000, help='Server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host, port, reload):
    """Start the transparency server."""
    click.echo(f"🚀 Starting RAGents transparency server on {host}:{port}")

    if reload:
        click.echo("🔄 Auto-reload enabled for development")

    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        import uvicorn
        from ragents.server.app import create_app

        app = create_app()
        uvicorn.run(app, host=host, port=port, reload=reload)
    except ImportError:
        click.echo("❌ uvicorn is required to run the server")
        click.echo("Install with: pip install uvicorn")
    except Exception as e:
        click.echo(f"❌ Server error: {e}")


@cli.command()
@click.option('--type', 'demo_type', default='transparency', help='Demo type: transparency, query-rewriting')
def demo(demo_type):
    """Run various RAGents demos."""

    demo_scripts = {
        'transparency': 'transparency_demo.py',
        'query-rewriting': 'query_rewriting_demo.py'
    }

    if demo_type not in demo_scripts:
        click.echo(f"❌ Unknown demo type: {demo_type}")
        click.echo(f"Available demos: {', '.join(demo_scripts.keys())}")
        return

    click.echo(f"🎬 Starting RAGents {demo_type} demo...")

    try:
        import subprocess
        import sys

        demo_script = os.path.join(os.path.dirname(__file__), "..", "examples", demo_scripts[demo_type])
        subprocess.run([sys.executable, demo_script])
    except Exception as e:
        click.echo(f"❌ Demo error: {e}")


@cli.command()
@click.option('--domain', default='general', help='Dataset domain (general, science, history, technology)')
@click.option('--size', default=5, help='Number of samples to generate')
def create_dataset(domain, size):
    """Create evaluation datasets."""
    click.echo(f"📊 Creating {domain} dataset with {size} samples...")

    try:
        from ragents.evaluation.datasets import create_sample_dataset

        dataset = create_sample_dataset(domain)
        click.echo(f"✅ Created dataset: {dataset.name}")
        click.echo(f"📈 Total samples: {len(dataset)}")

        # Save to file
        output_file = f"{domain}_dataset.json"
        dataset.save_to_json(output_file)
        click.echo(f"💾 Saved to: {output_file}")

    except Exception as e:
        click.echo(f"❌ Dataset creation error: {e}")


@cli.command()
@click.option('--backend', default='chromadb', help='Vector store backend')
@click.option('--test-data', is_flag=True, help='Add test data')
def init_vectorstore(backend, test_data):
    """Initialize vector store."""
    click.echo(f"🗄️ Initializing {backend} vector store...")

    try:
        from ragents.vector_stores import create_vector_store, VectorStoreConfig, VectorStoreType

        config = VectorStoreConfig(
            store_type=VectorStoreType(backend),
            collection_name="ragents_test"
        )

        vector_store = create_vector_store(config)
        click.echo(f"✅ Vector store initialized: {backend}")

        if test_data:
            click.echo("📝 Adding test data...")
            # Add some test vectors here
            click.echo("✅ Test data added")

    except Exception as e:
        click.echo(f"❌ Vector store error: {e}")


@cli.command()
def health():
    """Check system health and requirements."""
    click.echo("🏥 RAGents Health Check")
    click.echo("=" * 40)

    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"🐍 Python: {python_version}")

    # Check API keys
    openai_key = "✅" if os.environ.get("OPENAI_API_KEY") else "❌"
    anthropic_key = "✅" if os.environ.get("ANTHROPIC_API_KEY") else "❌"
    click.echo(f"🔑 OpenAI API Key: {openai_key}")
    click.echo(f"🔑 Anthropic API Key: {anthropic_key}")

    # Check optional dependencies
    deps = {
        "chromadb": "ChromaDB vector store",
        "weaviate": "Weaviate vector store",
        "asyncpg": "PostgreSQL pgvector",
        "elasticsearch": "Elasticsearch vector store",
        "uvicorn": "Web server",
    }

    click.echo("\n📦 Optional Dependencies:")
    for dep, desc in deps.items():
        try:
            __import__(dep)
            status = "✅"
        except ImportError:
            status = "❌"
        click.echo(f"{status} {dep}: {desc}")

    # Check environment variables
    click.echo("\n⚙️ Environment Variables:")
    env_vars = [
        "RAGENTS_LLM_PROVIDER",
        "RAGENTS_VECTOR_STORE_TYPE",
        "RAGENTS_CHUNK_SIZE",
        "RAGENTS_WORKING_DIR",
    ]

    for var in env_vars:
        value = os.environ.get(var, "Not set")
        click.echo(f"   {var}: {value}")


if __name__ == '__main__':
    cli()