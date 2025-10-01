"""Enhanced Chainlit application for RAGents with authentication, streaming, and observability."""

import asyncio
import os
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiofiles
from datetime import datetime
import time

import chainlit as cl

from ragents import (
    AgentConfig, RAGConfig, RAGEngine, LLMClient
)
from ragents.agents import SimpleAgent
from ragents.observability import (
    RAGTracer, SpanType, get_tracer, MetricsCollector, get_metrics_collector,
    StructuredLogger
)

try:
    from ragents.agents import DecisionTreeAgent, GraphPlannerAgent, ReActAgent
    ADVANCED_AGENTS_AVAILABLE = True
except ImportError:
    ADVANCED_AGENTS_AVAILABLE = False
from ragents.config.environment import get_llm_config_from_env
from ragents.ingestion.pipeline import IngestionPipeline


# Configuration
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize observability
tracer = get_tracer()
metrics = get_metrics_collector()
logger = StructuredLogger()

# Global state management
user_sessions: Dict[str, Dict[str, Any]] = {}

# Demo users for authentication (in production, use proper auth)
DEMO_USERS = {
    "admin": {"password": "admin123", "role": "admin", "name": "Administrator"},
    "user": {"password": "user123", "role": "user", "name": "Demo User"},
    "tenant1": {"password": "tenant123", "role": "tenant", "name": "Tenant 1"},
    "tenant2": {"password": "tenant456", "role": "tenant", "name": "Tenant 2"}
}


class AgentType:
    DECISION_TREE = "decision_tree"
    GRAPH_PLANNER = "graph_planner"
    REACT = "react"


def get_user_session(user_id: str) -> Dict[str, Any]:
    """Get or create user session data with tracing."""
    with tracer.span("get_user_session", SpanType.AGENT, user_id=user_id):
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                "rag_engine": None,
                "agent": None,
                "agent_type": None,
                "documents": [],
                "processed_documents": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            metrics.increment_counter("user_sessions_created", {"user_id": user_id})

        user_sessions[user_id]["last_activity"] = datetime.now().isoformat()
        return user_sessions[user_id]


def get_action_buttons() -> List[cl.Action]:
    """Generate standard action buttons for the UI."""
    actions = [
        cl.Action(name="upload_files", value="upload", label="📁 Upload Documents", payload={}),
        cl.Action(name="view_history", value="history", label="📋 View File History", payload={}),
        cl.Action(name="view_metrics", value="metrics", label="📊 View Metrics", payload={}),
    ]

    # Add agent selection actions
    if ADVANCED_AGENTS_AVAILABLE:
        actions.extend([
            cl.Action(name="agent_decision_tree", value="decision_tree", label="🌳 Decision Tree Agent", payload={}),
            cl.Action(name="agent_graph_planner", value="graph_planner", label="🗺️ Graph Planner Agent", payload={}),
            cl.Action(name="agent_react", value="react", label="🔄 ReAct Agent", payload={}),
        ])
    else:
        actions.append(cl.Action(name="agent_simple", value="simple", label="🤖 Simple Agent", payload={}))

    return actions


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Handle user authentication."""
    with tracer.span("authenticate_user", SpanType.AGENT, username=username):
        if username in DEMO_USERS and DEMO_USERS[username]["password"] == password:
            user_data = DEMO_USERS[username]
            metrics.increment_counter("authentication_success", {"username": username})

            return cl.User(
                identifier=username,
                metadata={
                    "role": user_data["role"],
                    "name": user_data["name"],
                    "login_time": datetime.now().isoformat()
                }
            )

        metrics.increment_counter("authentication_failed", {"username": username})
        return None


@cl.on_chat_start
async def start():
    """Initialize the chat session with observability."""
    with tracer.trace("chat_start", user_authentication=True):
        user = cl.user_session.get("user")
        if not user:
            await cl.Message(
                content="❌ Authentication required. Please log in to continue."
            ).send()
            return

        user_id = user.identifier
        session = get_user_session(user_id)

        # Track user activity
        metrics.increment_counter("chat_sessions_started", {"user_id": user_id})
        metrics.record_active_connections(len(user_sessions))

        # Get standard action buttons
        actions = get_action_buttons()

        # Create welcome message with all content
        welcome_content = f"""# 🤖 Welcome to RAGents, {user.metadata.get('name', user_id)}!

**Advanced Agentic RAG Framework with Full Observability**

👤 **User**: {user_id} ({user.metadata.get('role', 'user')})
🔐 **Session**: Authenticated
📊 **Documents**: {len(session.get('processed_documents', []))} processed
🤖 **Active Agent**: {(session.get('agent_type') or 'None').replace('_', ' ').title()}

### Getting Started:
1. **Upload Documents**: Click the '📁 Upload Documents' button
2. **Select Agent**: Choose which agent type to activate
3. **Ask Questions**: Start asking questions about your documents
4. **Monitor**: View metrics and observability data

*Ready to help you with document analysis and intelligent responses!*
"""

        # Send welcome message with actions
        await cl.Message(
            content=welcome_content,
            actions=actions
        ).send()


@cl.action_callback("upload_files")
async def on_upload_files(action):
    """Handle file upload with observability."""
    with tracer.span("file_upload_request", SpanType.DOCUMENT_PROCESSING):
        res = await cl.AskFileMessage(
            content="📤 **Upload Your Documents**\n\nPlease select one or more files to upload:",
            accept=["text/plain", "application/pdf", "text/markdown", "text/csv",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_size_mb=100,
            max_files=10
        ).send()

        if res:
            await handle_file_uploads(res)


@cl.action_callback("view_history")
async def on_view_history(action):
    """Display file upload history with observability."""
    with tracer.span("view_file_history", SpanType.RETRIEVAL):
        user = cl.user_session.get("user")
        user_id = user.identifier if user else "demo_user"
        session = get_user_session(user_id)

        processed_docs = session.get("processed_documents", [])
        metrics.increment_counter("history_views", {"user_id": user_id})

        if not processed_docs:
            await cl.Message(
                content="📋 **File History**\n\n*No files have been uploaded yet.*\n\nUse the '📁 Upload Documents' button to get started!"
            ).send()
            return

        # Build complete history content
        history_content = "📋 **File History**\n\n"

        for i, doc in enumerate(processed_docs, 1):
            status_icon = "✅" if doc.get("status") == "success" else "❌"
            file_size = doc.get("size", 0)
            size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"

            doc_info = f"{i}. {status_icon} **{doc['name']}**\n"
            doc_info += f"   - Size: {size_str}\n"
            doc_info += f"   - Uploaded: {doc.get('uploaded_at', 'Unknown')}\n"
            doc_info += f"   - Status: {doc.get('status', 'Unknown')}\n\n"

            history_content += doc_info

        # Send complete history
        await cl.Message(content=history_content).send()


@cl.action_callback("view_metrics")
async def on_view_metrics(action):
    """Display observability metrics."""
    with tracer.span("view_metrics", SpanType.EVALUATION):
        user = cl.user_session.get("user")
        user_id = user.identifier if user else "demo_user"

        # Get metrics summary
        summary = metrics.get_summary()

        metrics_content = f"""📊 **RAGents Observability Dashboard**

**System Metrics:**
- Uptime: {summary['uptime_seconds']:.1f} seconds
- Total Metrics: {summary['metrics_count']}
- Active Sessions: {len(user_sessions)}

**User Activity:**
- Current User: {user_id}
- Session Count: {len(user_sessions)}

**Recent Metrics:**
"""

        # Stream metrics data
        metrics_msg = cl.Message(content=metrics_content)
        await metrics_msg.send()

        for name, data in summary.get('metrics', {}).items():
            if data['latest_value'] is not None:
                metric_line = f"- **{name}**: {data['latest_value']:.2f} ({data['type']})\n"
                metrics_content += metric_line
                metrics_msg.content = metrics_content
                await metrics_msg.update()
                await asyncio.sleep(0.02)


# Agent selection callbacks with observability
@cl.action_callback("agent_decision_tree")
async def on_agent_decision_tree(action):
    await setup_agent(AgentType.DECISION_TREE)

@cl.action_callback("agent_graph_planner")
async def on_agent_graph_planner(action):
    await setup_agent(AgentType.GRAPH_PLANNER)

@cl.action_callback("agent_react")
async def on_agent_react(action):
    await setup_agent(AgentType.REACT)

@cl.action_callback("agent_simple")
async def on_agent_simple(action):
    await setup_agent("simple")


async def setup_agent(agent_type: str):
    """Setup the selected agent type with full observability."""
    with tracer.span("setup_agent", SpanType.AGENT, agent_type=agent_type):
        user = cl.user_session.get("user")
        user_id = user.identifier if user else "demo_user"
        session = get_user_session(user_id)

        start_time = time.time()

        try:
            # Setup LLM config
            llm_config = get_llm_config_from_env()
            llm_client = LLMClient(config=llm_config)

            # Setup RAG engine (always initialize it, documents can be added later)
            rag_config = RAGConfig()
            rag_engine = RAGEngine(config=rag_config, llm_client=llm_client)

            # If documents were already uploaded, ingest them into the RAG engine
            if session["processed_documents"]:
                for doc_info in session["processed_documents"]:
                    if doc_info.get("status") == "success" and doc_info.get("path"):
                        try:
                            await rag_engine.add_document(doc_info["path"], source=doc_info["name"])
                        except Exception as e:
                            print(f"Error re-ingesting document {doc_info['name']}: {e}")

            # Create agent based on type with proper configuration
            agent_name = agent_type.replace('_', ' ').title()
            agent_config = AgentConfig(
                name=agent_name,
                description=f"An intelligent {agent_name} agent for document analysis and question answering"
            )

            if agent_type == AgentType.DECISION_TREE and ADVANCED_AGENTS_AVAILABLE:
                agent = DecisionTreeAgent(config=agent_config, llm_client=llm_client, rag_engine=rag_engine)
            elif agent_type == AgentType.GRAPH_PLANNER and ADVANCED_AGENTS_AVAILABLE:
                agent = GraphPlannerAgent(config=agent_config, llm_client=llm_client, rag_engine=rag_engine)
            elif agent_type == AgentType.REACT and ADVANCED_AGENTS_AVAILABLE:
                agent = ReActAgent(config=agent_config, llm_client=llm_client, rag_engine=rag_engine)
            else:
                # Fallback to SimpleAgent
                agent = SimpleAgent(config=agent_config, llm_client=llm_client, rag_engine=rag_engine)
                agent_type = "simple"

            # Update session
            session["agent"] = agent
            session["agent_type"] = agent_type
            session["rag_engine"] = rag_engine

            # Record metrics
            duration = time.time() - start_time
            metrics.record_agent_decision(agent_type, duration, 1.0)
            metrics.increment_counter("agents_created", {"agent_type": agent_type})

            # Stream success message
            agent_name = agent_type.replace('_', ' ').title()
            success_msg = cl.Message(content="")
            await success_msg.send()

            success_parts = [
                f"✅ **Agent Activated: {agent_name}**\n\n",
                f"🤖 Your {agent_name} is now ready to help!\n",
                f"📊 Setup time: {duration:.2f}s\n",
                f"📚 Documents available: {len(session.get('processed_documents', []))}\n\n",
                "*Ask me anything about your documents or start a conversation!*"
            ]

            current_content = ""
            for part in success_parts:
                current_content += part
                success_msg.content = current_content
                await success_msg.update()
                await asyncio.sleep(0.1)

        except Exception as e:
            metrics.increment_counter("agent_setup_errors", {"agent_type": agent_type})
            await cl.Message(
                content=f"❌ **Error setting up agent**: {str(e)}\n\nPlease try again or contact support."
            ).send()


async def handle_file_uploads(files: List):
    """Process uploaded files with detailed progress tracking and observability."""
    with tracer.span("file_uploads", SpanType.DOCUMENT_PROCESSING, file_count=len(files)):
        user = cl.user_session.get("user")
        user_id = user.identifier if user else "demo_user"
        session = get_user_session(user_id)

        # Track upload metrics
        total_size = sum(getattr(f, 'size', 0) for f in files)
        metrics.record_document_processing("upload_batch", 0, len(files))

        # Show upload progress with streaming
        progress_msg = cl.Message(content="📤 **Processing Files...**\n\n🔄 Starting upload process...")
        await progress_msg.send()

        processed_files = []
        total_files = len(files)

        for i, file in enumerate(files, 1):
            with tracer.span("process_single_file", SpanType.DOCUMENT_PROCESSING,
                           file_name=file.name, file_index=i):
                start_time = time.time()

                try:
                    # Update progress with streaming
                    progress_msg.content = (
                        f"📤 **Processing Files...** ({i}/{total_files})\n\n"
                        f"🔄 Processing: **{file.name}**\n📊 Progress: {i}/{total_files}"
                    )
                    await progress_msg.update()

                    # Save file to upload directory
                    file_path = UPLOAD_DIR / f"{user_id}_{uuid.uuid4().hex}_{file.name}"

                    # Copy file content with progress tracking
                    async with aiofiles.open(file_path, 'wb') as f:
                        if hasattr(file, 'content'):
                            await f.write(file.content)
                        else:
                            # For file objects, read from path
                            async with aiofiles.open(file.path, 'rb') as src:
                                content = await src.read()
                                await f.write(content)

                    # Actually ingest into RAG engine if it exists
                    rag_engine = session.get("rag_engine")
                    if rag_engine:
                        # Chunking process
                        progress_msg.content = (
                            f"📤 **Processing Files...** ({i}/{total_files})\n\n"
                            f"✂️ Chunking: **{file.name}**\n📊 Progress: {i}/{total_files}"
                        )
                        await progress_msg.update()

                        # Embedding process - actually add to RAG engine
                        progress_msg.content = (
                            f"📤 **Processing Files...** ({i}/{total_files})\n\n"
                            f"🧠 Embedding: **{file.name}**\n📊 Progress: {i}/{total_files}"
                        )
                        await progress_msg.update()

                        # Add document to RAG engine
                        await rag_engine.add_document(str(file_path), source=file.name)
                    else:
                        # No RAG engine yet - just simulate progress
                        await asyncio.sleep(0.6)

                    # Track processed file
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    duration = time.time() - start_time

                    file_info = {
                        "name": file.name,
                        "path": str(file_path),
                        "size": file_size,
                        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "success",
                        "processing_time": duration
                    }

                    processed_files.append(file_info)
                    session["processed_documents"].append(file_info)

                    # Record metrics
                    metrics.record_document_processing(
                        file.name.split('.')[-1] if '.' in file.name else "unknown",
                        duration,
                        1  # chunk count (simplified)
                    )

                except Exception as e:
                    duration = time.time() - start_time
                    error_info = {
                        "name": file.name,
                        "path": "",
                        "size": 0,
                        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": f"failed: {str(e)}",
                        "processing_time": duration
                    }
                    processed_files.append(error_info)
                    metrics.increment_counter("file_processing_errors", {"error": str(e)})

        # Final success message with streaming
        success_count = sum(1 for f in processed_files if f["status"] == "success")
        total_size = sum(f["size"] for f in processed_files if f["status"] == "success")
        size_str = f"{total_size:,} bytes" if total_size < 1024 else f"{total_size/1024:.1f} KB"
        avg_time = sum(f.get("processing_time", 0) for f in processed_files) / len(processed_files)

        final_content = f"""✅ **Upload Complete!**

📊 **Summary:**
- Files processed: {success_count}/{total_files}
- Total size: {size_str}
- Average processing time: {avg_time:.2f}s
- Status: {'All files processed successfully!' if success_count == total_files else f'{total_files - success_count} files failed'}

🎯 **What's Next:**
1. Select an agent using the buttons below
2. Start asking questions about your documents
3. View file history anytime
4. Monitor performance with metrics

*Your documents are now ready for intelligent analysis!*
"""

        # Update final message and add action buttons
        progress_msg.content = final_content
        progress_msg.actions = get_action_buttons()
        await progress_msg.update()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with streaming responses and full observability."""
    with tracer.trace("message_processing", message_length=len(message.content)):
        user = cl.user_session.get("user")
        user_id = user.identifier if user else "demo_user"
        session = get_user_session(user_id)

        # Track message metrics
        metrics.increment_counter("messages_received", {"user_id": user_id})
        metrics.record_gauge("message_length", len(message.content))

        # Check for file uploads in message
        if message.elements:
            await handle_file_uploads(message.elements)
            return

        # Check if agent is set up
        if not session.get("agent"):
            await cl.Message(
                content="⚠️ **No agent selected!**\n\nPlease select an agent first using one of the buttons above."
            ).send()
            return

        # Process message with agent using streaming response
        try:
            with tracer.span("agent_processing", SpanType.AGENT,
                           agent_type=session.get("agent_type")):
                agent = session["agent"]
                start_time = time.time()

                # Create streaming response message
                response_msg = cl.Message(content="")
                await response_msg.send()

                # Show thinking indicator with streaming
                thinking_steps = ["🤔 Thinking", "🤔 Thinking.", "🤔 Thinking..", "🤔 Thinking..."]
                for step in thinking_steps:
                    response_msg.content = step
                    await response_msg.update()
                    await asyncio.sleep(0.3)

                # Process with agent (simplified - in real implementation, use actual streaming)
                try:
                    if hasattr(agent, 'process_message'):
                        response = await agent.process_message(message.content)
                    elif hasattr(agent, 'process_async'):
                        response = await agent.process_async(message.content)
                    elif hasattr(agent, 'process'):
                        response = agent.process(message.content)
                    else:
                        response = "Agent does not have a valid processing method."
                except Exception as agent_error:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Agent error: {error_details}")
                    response = f"I apologize, but I encountered an error processing your message: {str(agent_error)}"

                # Stream the response for better UX
                if isinstance(response, str):
                    # Simulate streaming by breaking response into chunks
                    words = response.split()
                    current_response = ""

                    for i, word in enumerate(words):
                        current_response += word + " "
                        if i % 3 == 0 or i == len(words) - 1:  # Update every 3 words
                            response_msg.content = current_response.strip()
                            await response_msg.update()
                            await asyncio.sleep(0.05)

                # Record metrics
                duration = time.time() - start_time
                metrics.record_rag_query(duration, len(session.get("processed_documents", [])), 0.8)
                metrics.increment_counter("messages_processed", {
                    "user_id": user_id,
                    "agent_type": session.get("agent_type", "unknown")
                })

        except Exception as e:
            metrics.increment_counter("message_processing_errors", {"error": str(e)})
            await cl.Message(
                content=f"❌ **Error processing message**: {str(e)}\n\nPlease try again."
            ).send()


if __name__ == "__main__":
    import subprocess
    import sys

    # Ensure required environment variables
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    print("🚀 Starting Enhanced RAGents Chainlit App...")
    print("🔐 Authentication enabled - Demo users: admin/admin123, user/user123")
    print("📊 Full observability enabled - metrics, tracing, and logging")
    print("🌊 Streaming responses enabled for better UX")
    print("📖 Access the app at: http://localhost:8000")

    subprocess.run([
        "chainlit", "run", __file__, "--port", "8000"
    ])
