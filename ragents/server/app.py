"""FastAPI application with SSE streaming for agent transparency."""

import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
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


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="RAGents Transparency Server",
        description="Real-time visualization of agent reasoning and decision making",
        version="0.1.0",
    )

    # Initialize components
    streamer = get_event_streamer()
    transparency_engine = TransparencyEngine(streamer)

    # Store active sessions
    active_sessions: Dict[str, Dict] = {}

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
            # Initialize RAG components
            rag_config = RAGConfig.from_env()
            llm_config = get_llm_config_from_env()
            llm_client = LLMClient(llm_config)
            rag_engine = RAGEngine(rag_config, llm_client)

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

            if request.agent_type == "decision_tree":
                agent = DecisionTreeAgent(agent_config, llm_client, rag_engine)
            elif request.agent_type == "graph_planner":
                agent = GraphPlannerAgent(agent_config, llm_client, rag_engine)
            elif request.agent_type == "react":
                agent = ReActAgent(agent_config, llm_client, rag_engine)
            else:
                agent = DecisionTreeAgent(agent_config, llm_client, rag_engine)

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

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "active_sessions": len(active_sessions)}

    return app


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the transparency server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)