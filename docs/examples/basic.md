# Basic Examples

Get started with RAGents through practical examples that demonstrate core functionality.

## Quick Start Examples

### 1. Simple Chat Agent

The most basic RAGents implementation - a conversational agent without RAG capabilities.

```python
import asyncio
from ragents import DecisionTreeAgent, AgentConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def simple_chat_agent():
    """Create a basic conversational agent."""

    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create agent configuration
    config = AgentConfig(
        name="Simple Assistant",
        description="A helpful conversational assistant",
        enable_memory=True,
        memory_window_size=10
    )

    # Create agent
    agent = DecisionTreeAgent(
        config=config,
        llm_client=llm_client
    )

    # Have a conversation
    print("Chat Agent Ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        response = await agent.process_message(user_input)
        print(f"Assistant: {response}")

# Run the example
if __name__ == "__main__":
    asyncio.run(simple_chat_agent())
```

### 2. RAG-Enabled Q&A System

A knowledge-based question answering system using RAG.

```python
import asyncio
from ragents import (
    DecisionTreeAgent, AgentConfig, RAGEngine, RAGConfig,
    create_vector_store, VectorStoreConfig, VectorStoreType
)
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def rag_qa_system():
    """Create a RAG-enabled Q&A system."""

    # Initialize components
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Configure vector store
    vector_config = VectorStoreConfig(
        store_type=VectorStoreType.CHROMADB,
        collection_name="knowledge_base",
        persist_directory="./chroma_db"
    )
    vector_store = create_vector_store(vector_config)

    # Configure RAG engine
    rag_config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        enable_caching=True
    )
    rag_engine = RAGEngine(rag_config, llm_client)

    # Create RAG-enabled agent
    agent_config = AgentConfig(
        name="Knowledge Assistant",
        enable_rag=True,
        enable_reasoning=True,
        enable_memory=True
    )

    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Add documents to knowledge base
    print("Adding documents to knowledge base...")
    documents = [
        "path/to/manual.pdf",
        "path/to/faq.txt",
        "path/to/documentation.md"
    ]

    for doc in documents:
        try:
            result = await rag_engine.add_document(doc)
            print(f"Added {doc}: {result.chunks_created} chunks")
        except FileNotFoundError:
            print(f"File not found: {doc}")

    # Interactive Q&A
    print("\nKnowledge-based Q&A Ready! Type 'quit' to exit.")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'quit':
            break

        response = await agent.process_message(question)
        print(f"Answer: {response}")

        # Show sources if available
        if hasattr(response, 'sources') and response.sources:
            print("\nSources:")
            for i, source in enumerate(response.sources[:3], 1):
                print(f"{i}. {source.metadata.get('filename', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(rag_qa_system())
```

### 3. Multi-Modal Document Processing

Process different types of documents including images and PDFs.

```python
import asyncio
from ragents import RAGEngine, RAGConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def multimodal_processing():
    """Process multiple document types including images."""

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Configure RAG with vision capabilities
    rag_config = RAGConfig(
        enable_vision=True,
        chunk_size=1000,
        top_k=5
    )
    rag_engine = RAGEngine(rag_config, llm_client)

    # Process different document types
    documents = {
        "pdf": "research_paper.pdf",
        "image": "chart.png",
        "text": "notes.txt",
        "presentation": "slides.pptx"
    }

    print("Processing multimodal documents...")

    for doc_type, file_path in documents.items():
        try:
            print(f"\nProcessing {doc_type}: {file_path}")

            if doc_type == "image":
                # Special handling for images
                result = await rag_engine.add_document(
                    file_path,
                    document_type="image",
                    extract_text=True,  # OCR
                    describe_image=True  # Generate description
                )
                print(f"  - Text extracted: {result.text_extracted}")
                print(f"  - Image described: {result.image_described}")

            else:
                # Standard document processing
                result = await rag_engine.add_document(file_path)

            print(f"  - Chunks created: {result.chunks_created}")
            print(f"  - Processing time: {result.processing_time:.2f}s")

        except FileNotFoundError:
            print(f"  - File not found: {file_path}")
        except Exception as e:
            print(f"  - Error processing {file_path}: {e}")

    # Query the multimodal knowledge base
    queries = [
        "What are the main findings in the research paper?",
        "What trends are shown in the chart?",
        "Summarize the key points from the presentation"
    ]

    print("\nQuerying multimodal knowledge base:")
    for query in queries:
        print(f"\nQuery: {query}")
        response = await rag_engine.query(
            query,
            include_visual_context=True,
            max_tokens=200
        )
        print(f"Answer: {response.answer}")

if __name__ == "__main__":
    asyncio.run(multimodal_processing())
```

## Agent Examples

### 4. ReAct Agent with Tools

Demonstrate the ReAct (Reasoning + Acting) pattern with tool usage.

```python
import asyncio
from ragents import ReActAgent, AgentConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env
from ragents.tools import tool

# Define custom tools
@tool(name="calculator", description="Perform mathematical calculations")
def calculate(expression: str) -> str:
    """Safe calculator for basic math operations."""
    try:
        # Simple eval (use a proper math parser in production)
        result = eval(expression.replace("^", "**"))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool(name="unit_converter", description="Convert between units")
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units."""
    conversions = {
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x / 3.28084,
        ("kg", "pounds"): lambda x: x * 2.20462,
        ("pounds", "kg"): lambda x: x / 2.20462,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported"

async def react_agent_example():
    """Demonstrate ReAct agent with tool usage."""

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create ReAct agent
    config = AgentConfig(
        name="ReAct Tool Agent",
        enable_tools=True,
        enable_reasoning=True,
        max_iterations=5,
        reasoning_detail_level="verbose"
    )

    agent = ReActAgent(
        config=config,
        llm_client=llm_client
    )

    # Test queries that require reasoning and tool use
    test_queries = [
        "What is 25% of 480?",
        "Convert 100 degrees Fahrenheit to Celsius",
        "If I have a rectangular room that is 12 feet by 15 feet, how many square meters is that?",
        "Calculate the compound interest on $1000 at 5% annual rate for 3 years"
    ]

    print("ReAct Agent with Tools Demo")
    print("=" * 40)

    for query in test_queries:
        print(f"\n🤔 Query: {query}")
        print("-" * 30)

        response = await agent.process_message(query)

        print(f"🔧 Final Answer: {response.content}")

        if hasattr(response, 'reasoning_steps'):
            print("\n🧠 Reasoning Steps:")
            for i, step in enumerate(response.reasoning_steps, 1):
                print(f"  {i}. {step.thought}")
                if step.action:
                    print(f"     Action: {step.action}")
                if step.observation:
                    print(f"     Result: {step.observation}")

        print("\n" + "=" * 40)

if __name__ == "__main__":
    asyncio.run(react_agent_example())
```

### 5. Custom Decision Tree Agent

Create a specialized agent with custom decision logic.

```python
import asyncio
from ragents import DecisionTreeAgent, AgentConfig
from ragents.agents.decision_tree import DecisionNode, ActionNode
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def custom_decision_tree_agent():
    """Create an agent with custom decision tree for customer support."""

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Define custom decision tree for customer support
    decision_tree = [
        DecisionNode(
            id="intent_classifier",
            condition="classify_customer_intent(message)",
            branches={
                "technical_issue": "handle_technical",
                "billing_question": "handle_billing",
                "general_inquiry": "handle_general",
                "complaint": "handle_complaint",
                "default": "escalate_to_human"
            }
        ),

        ActionNode(
            id="handle_technical",
            action="search_technical_docs",
            tools=["rag_search"],
            response_template="Based on our technical documentation: {context}",
            next_node="follow_up_check"
        ),

        ActionNode(
            id="handle_billing",
            action="provide_billing_info",
            response_template="For billing questions, please check your account at portal.company.com or contact billing@company.com",
            next_node="follow_up_check"
        ),

        ActionNode(
            id="handle_general",
            action="general_assistance",
            use_llm=True,
            system_prompt="You are a helpful customer service assistant. Be friendly and informative.",
            next_node="follow_up_check"
        ),

        ActionNode(
            id="handle_complaint",
            action="escalate_complaint",
            response_template="I understand your concern. Let me escalate this to our customer care team. Reference number: {ticket_id}",
            next_node=None
        ),

        ActionNode(
            id="follow_up_check",
            action="check_satisfaction",
            response_template="Was this helpful? Is there anything else I can assist you with?",
            next_node=None
        )
    ]

    # Create agent with custom decision tree
    config = AgentConfig(
        name="Customer Support Agent",
        decision_tree=decision_tree,
        enable_memory=True,
        enable_reasoning=True
    )

    agent = DecisionTreeAgent(
        config=config,
        llm_client=llm_client
    )

    # Test scenarios
    test_scenarios = [
        "My login isn't working and I can't access my account",
        "I was charged twice for my subscription last month",
        "What are your business hours?",
        "This service is terrible and I want a refund!"
    ]

    print("Customer Support Agent Demo")
    print("=" * 40)

    for scenario in test_scenarios:
        print(f"\n📧 Customer: {scenario}")
        print("-" * 30)

        response = await agent.process_message(scenario)
        print(f"🎧 Support: {response}")

        print("\n" + "=" * 40)

if __name__ == "__main__":
    asyncio.run(custom_decision_tree_agent())
```

## RAG Examples

### 6. Document Analysis Pipeline

Batch process and analyze multiple documents.

```python
import asyncio
from ragents import RAGEngine, RAGConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def document_analysis_pipeline():
    """Batch process and analyze documents."""

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    rag_config = RAGConfig(
        chunk_size=1200,
        chunk_overlap=200,
        top_k=8,
        enable_caching=True
    )
    rag_engine = RAGEngine(rag_config, llm_client)

    # Documents to analyze
    documents = [
        {"path": "quarterly_report_q1.pdf", "category": "financial"},
        {"path": "quarterly_report_q2.pdf", "category": "financial"},
        {"path": "market_analysis.pdf", "category": "research"},
        {"path": "competitor_review.pdf", "category": "research"}
    ]

    # Batch process documents
    print("Processing documents...")
    processed_docs = []

    for doc in documents:
        try:
            result = await rag_engine.add_document(
                doc["path"],
                metadata={"category": doc["category"]},
                chunk_strategy="semantic"
            )
            processed_docs.append({
                "path": doc["path"],
                "category": doc["category"],
                "chunks": result.chunks_created,
                "processing_time": result.processing_time
            })
            print(f"✓ {doc['path']}: {result.chunks_created} chunks")

        except Exception as e:
            print(f"✗ {doc['path']}: Error - {e}")

    # Analysis queries
    analysis_queries = [
        {
            "query": "What are the key financial metrics across quarters?",
            "category_filter": "financial",
            "analysis_type": "financial_trends"
        },
        {
            "query": "What market opportunities were identified?",
            "category_filter": "research",
            "analysis_type": "market_opportunities"
        },
        {
            "query": "Compare our performance with competitors",
            "category_filter": None,  # Search all categories
            "analysis_type": "competitive_analysis"
        }
    ]

    print("\nPerforming document analysis...")

    analysis_results = {}
    for analysis in analysis_queries:
        print(f"\n📊 Analysis: {analysis['analysis_type']}")
        print(f"Query: {analysis['query']}")

        # Apply category filter if specified
        filters = {}
        if analysis["category_filter"]:
            filters["metadata.category"] = analysis["category_filter"]

        response = await rag_engine.query(
            analysis["query"],
            filters=filters,
            include_sources=True,
            max_tokens=300
        )

        analysis_results[analysis["analysis_type"]] = {
            "query": analysis["query"],
            "answer": response.answer,
            "sources": len(response.sources),
            "confidence": response.confidence
        }

        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.sources)} documents")
        print(f"Confidence: {response.confidence:.2f}")

    # Generate summary report
    print("\n" + "=" * 50)
    print("DOCUMENT ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"Documents processed: {len(processed_docs)}")
    total_chunks = sum(doc["chunks"] for doc in processed_docs)
    print(f"Total chunks created: {total_chunks}")

    print("\nAnalysis Results:")
    for analysis_type, result in analysis_results.items():
        print(f"\n{analysis_type.upper()}:")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Sources used: {result['sources']}")

if __name__ == "__main__":
    asyncio.run(document_analysis_pipeline())
```

### 7. Interactive Knowledge Explorer

Create an interactive system for exploring a knowledge base.

```python
import asyncio
from ragents import RAGEngine, RAGConfig, DecisionTreeAgent, AgentConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def interactive_knowledge_explorer():
    """Interactive knowledge base exploration system."""

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Setup RAG engine
    rag_config = RAGConfig(
        chunk_size=800,
        top_k=6,
        enable_query_expansion=True,
        enable_caching=True
    )
    rag_engine = RAGEngine(rag_config, llm_client)

    # Setup knowledge agent
    agent_config = AgentConfig(
        name="Knowledge Explorer",
        enable_rag=True,
        enable_memory=True,
        enable_reasoning=True,
        memory_window_size=20
    )

    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Add knowledge base
    knowledge_sources = [
        "encyclopedia.pdf",
        "technical_manual.pdf",
        "research_papers/*.pdf"
    ]

    print("🔍 Knowledge Explorer - Setting up knowledge base...")

    for source in knowledge_sources:
        try:
            if "*" in source:
                # Handle wildcards (simplified)
                print(f"Processing directory: {source}")
            else:
                result = await rag_engine.add_document(source)
                print(f"✓ Added {source}")
        except Exception as e:
            print(f"⚠ Could not add {source}: {e}")

    print("\n" + "=" * 50)
    print("🧠 KNOWLEDGE EXPLORER")
    print("=" * 50)
    print("Commands:")
    print("  'explore <topic>' - Deep dive into a topic")
    print("  'summarize <topic>' - Get a summary")
    print("  'compare <A> vs <B>' - Compare concepts")
    print("  'history' - Show conversation history")
    print("  'sources' - Show available sources")
    print("  'quit' - Exit")
    print("=" * 50)

    conversation_history = []

    while True:
        user_input = input("\n🔍 Explorer> ").strip()

        if user_input.lower() == 'quit':
            break

        elif user_input.lower() == 'history':
            print("\n📚 Conversation History:")
            for i, item in enumerate(conversation_history[-10:], 1):
                print(f"{i}. Q: {item['question'][:60]}...")
            continue

        elif user_input.lower() == 'sources':
            stats = await rag_engine.get_stats()
            print(f"\n📊 Knowledge Base Stats:")
            print(f"Documents: {stats.get('documents', 0)}")
            print(f"Chunks: {stats.get('chunks', 0)}")
            continue

        # Process query
        try:
            print("🤔 Searching knowledge base...")

            response = await agent.process_message(user_input)

            print(f"\n💡 {response}")

            # Show sources if available
            if hasattr(response, 'sources') and response.sources:
                print(f"\n📖 Sources ({len(response.sources)}):")
                for i, source in enumerate(response.sources[:3], 1):
                    filename = source.metadata.get('filename', 'Unknown')
                    print(f"  {i}. {filename}")

            # Save to history
            conversation_history.append({
                "question": user_input,
                "answer": str(response)[:200],
                "sources": len(getattr(response, 'sources', []))
            })

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Thank you for exploring the knowledge base!")

if __name__ == "__main__":
    asyncio.run(interactive_knowledge_explorer())
```

## Integration Examples

### 8. FastAPI Web Service

Integrate RAGents with a web API.

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from ragents import DecisionTreeAgent, AgentConfig, RAGEngine, RAGConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 500
    include_sources: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    processing_time: float

class DocumentUpload(BaseModel):
    content: str
    filename: str
    metadata: Optional[dict] = {}

# Initialize RAGents components
app = FastAPI(title="RAGents API", version="1.0.0")

# Global components (in production, use dependency injection)
llm_client = None
rag_engine = None
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAGents components on startup."""
    global llm_client, rag_engine, agent

    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Initialize RAG engine
    rag_config = RAGConfig.from_env()
    rag_engine = RAGEngine(rag_config, llm_client)

    # Initialize agent
    agent_config = AgentConfig(
        name="API Agent",
        enable_rag=True,
        enable_memory=False,  # Stateless for API
        enable_reasoning=True
    )

    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    print("✅ RAGents API initialized successfully")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base."""
    import time

    start_time = time.time()

    try:
        response = await agent.process_message(request.query)

        # Extract information
        answer = str(response)
        sources = []
        confidence = 0.8  # Default confidence

        if hasattr(response, 'sources'):
            sources = [
                source.metadata.get('filename', 'Unknown')
                for source in response.sources[:5]
            ]

        if hasattr(response, 'confidence'):
            confidence = response.confidence

        processing_time = time.time() - start_time

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_document(document: DocumentUpload, background_tasks: BackgroundTasks):
    """Add a document to the knowledge base."""

    def process_document():
        # Process document in background
        asyncio.run(rag_engine.add_text(
            document.content,
            metadata={
                "filename": document.filename,
                **document.metadata
            }
        ))

    background_tasks.add_task(process_document)

    return {"message": f"Document '{document.filename}' queued for processing"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test LLM connection
        await llm_client.complete("test", max_tokens=1)
        return {"status": "healthy", "components": ["llm", "rag", "agent"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics."""
    try:
        stats = await rag_engine.get_stats()
        return {
            "documents": stats.get("documents", 0),
            "chunks": stats.get("chunks", 0),
            "last_updated": stats.get("last_updated", "Unknown")
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Running the Examples

### Setup Instructions

1. **Install RAGents:**
```bash
pip install -e .
# or
pip install -e ".[all]"
```

2. **Set Environment Variables:**
```bash
export OPENAI_API_KEY="your-openai-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
```

3. **Prepare Sample Documents:**
```bash
mkdir sample_docs
# Add your PDF, text, and other documents to this directory
```

4. **Run Examples:**
```bash
python examples/basic_chat.py
python examples/rag_qa.py
python examples/react_tools.py
```

### Customization Tips

1. **Modify Agent Behavior** - Adjust `AgentConfig` parameters
2. **Add Custom Tools** - Use the `@tool` decorator
3. **Change Vector Store** - Update `VectorStoreConfig`
4. **Tune RAG Settings** - Modify `RAGConfig` parameters
5. **Customize Prompts** - Override system prompts in agent configs

## Next Steps

- **[Advanced Examples](advanced.md)** - More complex use cases
- **[Integration Examples](integrations.md)** - External system integrations
- **[Agent Customization](../agents/custom.md)** - Build specialized agents
- **[RAG Optimization](../rag/retrieval.md)** - Improve retrieval performance