"""FastAPI application with SSE streaming for agent transparency."""

import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..config.environment import get_llm_config_from_env
from ..config.rag_config import RAGConfig
from ..llm.client import LLMClient
from ..rag.engine import RAGEngine
from .events import EventStreamer, StreamingAgentWrapper, get_event_streamer
from .transparency import TransparencyEngine


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    agent_type: str = "decision_tree"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    session_id: str


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    id: str
    filename: str
    status: str
    message: str


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str
    filename: str
    size: int
    status: str
    uploaded_at: str
    user_id: str


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="RAGents Transparency Server",
        description="Real-time visualization of agent reasoning and decision making",
        version="0.1.0",
    )

    # Add CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize components
    streamer = get_event_streamer()
    transparency_engine = TransparencyEngine(streamer)

    # Store active sessions and documents
    active_sessions: Dict[str, Dict] = {}
    uploaded_documents: Dict[str, DocumentInfo] = {}
    document_content_store: Dict[str, Dict] = {}  # Separate store for document content

    # Mount static files and templates
    try:
        app.mount("/static", StaticFiles(directory="ragents/frontend/static"), name="static")
        templates = Jinja2Templates(directory="ragents/frontend/templates")
    except Exception:
        # For development, create basic structure
        templates = None

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Main dashboard page."""
        if templates:
            return templates.TemplateResponse("dashboard.html", {"request": request})
        else:
            # Return basic HTML if templates not available
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RAGents Transparency Dashboard</title>
                <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
                <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
                <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .events { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
                    .event { background: white; margin: 10px 0; padding: 15px; border-radius: 4px; border-left: 4px solid #007acc; }
                    .chat-container { display: flex; gap: 20px; }
                    .chat-input { flex: 1; }
                    .visualization { flex: 2; }
                    button { background: #007acc; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                    input, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 RAGents Transparency Dashboard</h1>
                    <div id="app"></div>
                </div>
                <script type="text/babel">
                    function App() {
                        const [sessionId, setSessionId] = React.useState('');
                        const [message, setMessage] = React.useState('');
                        const [events, setEvents] = React.useState([]);
                        const [isConnected, setIsConnected] = React.useState(false);

                        React.useEffect(() => {
                            if (sessionId) {
                                const eventSource = new EventSource(`/events/${sessionId}`);
                                eventSource.onmessage = (event) => {
                                    const data = JSON.parse(event.data);
                                    setEvents(prev => [...prev, data]);
                                };
                                eventSource.onopen = () => setIsConnected(true);
                                eventSource.onerror = () => setIsConnected(false);
                                return () => eventSource.close();
                            }
                        }, [sessionId]);

                        const sendMessage = async () => {
                            if (!message.trim()) return;

                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    message,
                                    session_id: sessionId || undefined
                                })
                            });

                            const result = await response.json();
                            setSessionId(result.session_id);
                            setMessage('');
                        };

                        return (
                            <div className="chat-container">
                                <div className="chat-input">
                                    <h2>Chat Interface</h2>
                                    <div style={{marginBottom: '10px'}}>
                                        <label>Session ID: </label>
                                        <input
                                            value={sessionId}
                                            onChange={(e) => setSessionId(e.target.value)}
                                            placeholder="Auto-generated or enter custom"
                                        />
                                    </div>
                                    <textarea
                                        value={message}
                                        onChange={(e) => setMessage(e.target.value)}
                                        placeholder="Ask the agent something..."
                                        rows={4}
                                    />
                                    <button onClick={sendMessage}>Send Message</button>
                                    <p>Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}</p>
                                </div>

                                <div className="visualization">
                                    <h2>Live Agent Reasoning</h2>
                                    <div className="events">
                                        {events.map((event, i) => (
                                            <div key={i} className="event">
                                                <strong>{event.type}</strong> - {new Date(event.timestamp * 1000).toLocaleTimeString()}
                                                <pre>{JSON.stringify(event.data, null, 2)}</pre>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        );
                    }

                    ReactDOM.render(<App />, document.getElementById('app'));
                </script>
            </body>
            </html>
            """)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Handle chat requests with streaming agent transparency."""
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        try:
            # Use shared RAG engine with uploaded documents
            rag_engine = get_rag_engine()
            if not rag_engine:
                # Fallback: create new instances
                rag_config = RAGConfig.from_env()
                llm_config = get_llm_config_from_env()
                llm_client = LLMClient(llm_config)
                rag_engine = RAGEngine(rag_config, llm_client)
            else:
                # Use the shared LLM client from the RAG engine
                llm_client = rag_engine.llm_client

            # Create agent based on type
            from ..agents.base import AgentConfig
            from ..agents.decision_tree import DecisionTreeAgent
            from ..agents.graph_planner import GraphPlannerAgent
            from ..agents.react_agent import ReActAgent

            agent_config = AgentConfig(
                name=f"{request.agent_type}_agent",
                enable_rag=True,
                enable_reasoning=True,
            )

            # Create a simple RAG engine that can search uploaded documents
            class SimpleDocumentRAG:
                def __init__(self, content_store):
                    self.content_store = content_store

                async def query(self, query_text):
                    # Simple search through uploaded document content
                    results = []
                    for doc_id, doc_data in self.content_store.items():
                        try:
                            content = doc_data['content']
                            filename = doc_data['filename']

                            # Check if content appears to be binary/corrupted
                            if self._is_binary_content(content):
                                print(f"Warning: Document {filename} contains binary/unreadable content")
                                continue

                            # Simple keyword matching
                            query_words = query_text.lower().split()
                            content_lower = content.lower()

                            matches = sum(1 for word in query_words if word in content_lower)
                            if matches > 0:
                                # Find relevant excerpt
                                lines = content.split('\n')
                                relevant_lines = []
                                for line in lines:
                                    if any(word in line.lower() for word in query_words):
                                        relevant_lines.append(line.strip())
                                        if len(relevant_lines) >= 3:  # Max 3 relevant lines
                                            break

                                if relevant_lines:
                                    excerpt = '\n'.join(relevant_lines)
                                    results.append(f"From {filename}:\n{excerpt}")
                        except Exception as e:
                            print(f"Error processing document {doc_data.get('filename', 'unknown')}: {e}")
                            continue

                    from types import SimpleNamespace
                    if results:
                        return SimpleNamespace(answer='\n\n'.join(results))
                    else:
                        readable_docs = [doc for doc in self.content_store.values()
                                       if not self._is_binary_content(doc.get('content', ''))]
                        if len(readable_docs) == 0:
                            return SimpleNamespace(answer="The uploaded documents appear to be in unreadable formats. Please re-upload them - the document processing has been fixed.")
                        else:
                            return SimpleNamespace(answer="No relevant information found in uploaded documents.")

                def _is_binary_content(self, content):
                    """Check if content appears to be binary data."""
                    if not content:
                        return False

                    # Check for null bytes (common in binary files)
                    if '\x00' in content:
                        return True

                    # Check for high ratio of non-printable characters
                    printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
                    if len(content) > 100 and printable_chars / len(content) < 0.7:
                        return True

                    return False

            simple_rag = SimpleDocumentRAG(document_content_store)

            if request.agent_type == "decision_tree":
                agent = DecisionTreeAgent(agent_config, llm_client, simple_rag)
            elif request.agent_type == "graph_planner":
                agent = GraphPlannerAgent(agent_config, llm_client, simple_rag)
            elif request.agent_type == "react":
                agent = ReActAgent(agent_config, llm_client, simple_rag)
            else:
                agent = DecisionTreeAgent(agent_config, llm_client, simple_rag)

            # Wrap agent for streaming
            streaming_agent = StreamingAgentWrapper(agent, streamer, session_id)

            # Store session
            active_sessions[session_id] = {
                "agent": streaming_agent,
                "transparency_engine": transparency_engine,
            }

            # Process message with streaming
            response = await streaming_agent.process_message_with_streaming(request.message)

            return ChatResponse(response=response, session_id=session_id)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/events/{session_id}")
    async def stream_events(session_id: str):
        """Stream Server-Sent Events for a session."""
        return StreamingResponse(
            streamer.subscribe(session_id),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    @app.get("/sessions/{session_id}/summary")
    async def get_session_summary(session_id: str):
        """Get summary of a session."""
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        transparency_engine = active_sessions[session_id]["transparency_engine"]
        return transparency_engine.create_transparency_report(session_id)

    @app.delete("/sessions/{session_id}")
    async def clear_session(session_id: str):
        """Clear a session."""
        await streamer.clear_session(session_id)
        if session_id in active_sessions:
            del active_sessions[session_id]
        return {"message": "Session cleared"}

    # Global RAG engine instance for document processing
    global_rag_engine = None

    def get_rag_engine():
        nonlocal global_rag_engine
        if global_rag_engine is None:
            try:
                rag_config = RAGConfig.from_env()
                llm_config = get_llm_config_from_env()
                llm_client = LLMClient(llm_config)
                global_rag_engine = RAGEngine(rag_config, llm_client)
            except Exception as e:
                print(f"Warning: Could not initialize RAG engine: {e}")
                global_rag_engine = None
        return global_rag_engine

    @app.post("/upload", response_model=DocumentUploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        user_id: str = Form(...)
    ):
        """Upload and process a document through RAG pipeline."""
        import tempfile
        import os

        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())

            # Read file content
            content = await file.read()

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Get RAG engine
                rag_engine = get_rag_engine()

                if rag_engine:
                    try:
                        # Try to process document through RAG pipeline
                        print(f"Processing document {file.filename} through RAG pipeline...")

                        # Process the document through the proper RAG pipeline
                        document = await rag_engine.add_document(temp_file_path, user_id=user_id)

                        # Store the processed document content in the global content store
                        document_content_store[doc_id] = {
                            'content': document.content,
                            'filename': file.filename,
                            'user_id': user_id,
                            'metadata': document.metadata
                        }

                        # Create document info
                        doc_info = DocumentInfo(
                            id=doc_id,
                            filename=file.filename or "unknown.txt",
                            size=len(content),
                            status="completed",
                            uploaded_at=str(uuid.uuid4()),  # Simplified timestamp
                            user_id=user_id
                        )

                        # Store document
                        uploaded_documents[doc_id] = doc_info

                        return DocumentUploadResponse(
                            id=doc_id,
                            filename=doc_info.filename,
                            status="completed",
                            message="Document uploaded and content stored successfully"
                        )
                    except Exception as e:
                        print(f"RAG processing failed: {e}")

                        # Determine if this is a specific format issue
                        error_msg = str(e).lower()
                        if "python-docx" in error_msg or "docx" in file.filename.lower():
                            error_detail = "DOCX processing failed - python-docx library issue"
                        elif "pymupdf" in error_msg or "pdf" in file.filename.lower():
                            error_detail = "PDF processing failed - PyMuPDF library issue"
                        elif any(word in error_msg for word in ["encrypt", "password", "protected"]):
                            error_detail = "Document is password protected or encrypted"
                        elif any(word in error_msg for word in ["corrupt", "invalid", "malformed"]):
                            error_detail = "Document appears to be corrupted or malformed"
                        else:
                            error_detail = f"Document processing failed: {str(e)}"

                        return DocumentUploadResponse(
                            id=doc_id,
                            filename=file.filename or "unknown",
                            status="failed",
                            message=error_detail
                        )
                else:
                    # Fallback: use basic processors without full RAG pipeline
                    try:
                        from pathlib import Path
                        from ..ingestion.processors import get_processor_for_file
                        from ..ingestion.config import IngestionConfig

                        processor = get_processor_for_file(Path(temp_file_path))
                        if processor:
                            config = IngestionConfig()
                            document = await processor.process(Path(temp_file_path), config)

                            document_content_store[doc_id] = {
                                'content': document.content,
                                'filename': file.filename,
                                'user_id': user_id,
                                'metadata': document.metadata
                            }
                        else:
                            # Last resort: basic text decoding for supported text files only
                            if any(temp_file_path.lower().endswith(ext) for ext in ['.txt', '.md']):
                                content_text = content.decode('utf-8', errors='ignore')
                            else:
                                content_text = f"Unsupported file type: {file.filename}. Please ensure you have uploaded a supported document format."

                            document_content_store[doc_id] = {
                                'content': content_text,
                                'filename': file.filename,
                                'user_id': user_id
                            }
                    except Exception as e:
                        print(f"Fallback processing failed: {e}")
                        # Final fallback with clear error message
                        document_content_store[doc_id] = {
                            'content': f"Document processing failed for {file.filename}. Error: {str(e)}",
                            'filename': file.filename,
                            'user_id': user_id
                        }

                    doc_info = DocumentInfo(
                        id=doc_id,
                        filename=file.filename or "unknown.txt",
                        size=len(content),
                        status="completed",
                        uploaded_at=str(uuid.uuid4()),
                        user_id=user_id
                    )

                    uploaded_documents[doc_id] = doc_info

                    return DocumentUploadResponse(
                        id=doc_id,
                        filename=doc_info.filename,
                        status="completed",
                        message="Document uploaded (RAG processing unavailable)"
                    )

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    @app.get("/documents")
    async def get_documents(user_id: str) -> List[DocumentInfo]:
        """Get all documents for a user."""
        user_docs = [doc for doc in uploaded_documents.values() if doc.user_id == user_id]
        return user_docs

    @app.get("/debug/documents")
    async def debug_documents():
        """Debug endpoint to check document store status."""
        debug_info = {
            "total_documents": len(uploaded_documents),
            "total_content_entries": len(document_content_store),
            "documents": {}
        }

        for doc_id, content_data in document_content_store.items():
            content = content_data.get('content', '')

            # Check if binary
            is_binary = False
            if content:
                if '\x00' in content:
                    is_binary = True
                elif len(content) > 100:
                    printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
                    if printable_chars / len(content) < 0.7:
                        is_binary = True

            debug_info["documents"][doc_id] = {
                "filename": content_data.get('filename', 'unknown'),
                "content_length": len(content),
                "is_binary": is_binary,
                "content_preview": content[:100] if not is_binary else "[BINARY DATA]"
            }

        return debug_info

    @app.post("/debug/clear-corrupted")
    async def clear_corrupted_documents():
        """Clear documents with corrupted/binary content."""
        removed_docs = []

        # Create a copy of keys to avoid modification during iteration
        doc_ids_to_check = list(document_content_store.keys())

        for doc_id in doc_ids_to_check:
            content_data = document_content_store[doc_id]
            content = content_data.get('content', '')

            # Check if binary/corrupted
            is_corrupted = False
            if content:
                if '\x00' in content:
                    is_corrupted = True
                elif len(content) > 100:
                    printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
                    if printable_chars / len(content) < 0.7:
                        is_corrupted = True

            if is_corrupted:
                filename = content_data.get('filename', 'unknown')
                removed_docs.append({"id": doc_id, "filename": filename})

                # Remove from both stores
                del document_content_store[doc_id]
                if doc_id in uploaded_documents:
                    del uploaded_documents[doc_id]

        return {"message": f"Removed {len(removed_docs)} corrupted documents", "removed": removed_docs}

    @app.delete("/documents/{document_id}")
    async def delete_document(document_id: str):
        """Delete a document."""
        if document_id not in uploaded_documents:
            raise HTTPException(status_code=404, detail="Document not found")

        del uploaded_documents[document_id]
        return {"message": "Document deleted successfully"}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "active_sessions": len(active_sessions), "documents": len(uploaded_documents)}

    return app


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the transparency server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)