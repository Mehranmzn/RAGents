"""Demonstration of RAGents transparency features with live visualization."""

import asyncio
import threading
import time
import webbrowser

from ragents.server.app import start_server


def start_transparency_server():
    """Start the transparency server in a separate thread."""
    def run_server():
        print("🚀 Starting RAGents Transparency Server...")
        print("📊 Dashboard will be available at: http://localhost:8000")
        print("🔴 Press Ctrl+C to stop the server")
        start_server(host="0.0.0.0", port=8000)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread


async def demo_transparency_workflow():
    """Demonstrate the full transparency workflow."""
    print("🎯 RAGents Transparency Demo")
    print("=" * 60)

    # Start the server
    server_thread = start_transparency_server()

    # Wait a moment for server to start
    await asyncio.sleep(3)

    print("\n✅ Server started successfully!")
    print("\n📋 Demo Instructions:")
    print("1. Open http://localhost:8000 in your browser")
    print("2. Try these example queries to see transparency in action:")
    print("   • 'What are the benefits of renewable energy?'")
    print("   • 'How does machine learning work?'")
    print("   • 'Explain photosynthesis step by step'")
    print("3. Watch the real-time visualization of:")
    print("   • Decision tree traversal")
    print("   • Tool execution events")
    print("   • RAG query processing")
    print("   • Adaptive data rendering")
    print("\n🔍 Features to observe:")
    print("   • Live events log with timestamps")
    print("   • Decision tree node progression")
    print("   • Automatic data format detection")
    print("   • Different renderers (tables, cards, charts)")

    # Try to open browser automatically
    try:
        webbrowser.open("http://localhost:8000")
        print("\n🌐 Browser opened automatically!")
    except Exception:
        print("\n💻 Please manually open: http://localhost:8000")

    print("\n⌨️  Press Enter when ready to see example API usage...")
    input()

    # Demonstrate API usage
    await demo_api_integration()

    print("\n🎉 Demo completed! Server is still running.")
    print("💡 Keep exploring the dashboard to see more transparency features!")

    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Demo ended. Thanks for trying RAGents!")


async def demo_api_integration():
    """Demonstrate programmatic API integration."""
    import httpx

    print("\n🔧 API Integration Demo")
    print("-" * 30)

    base_url = "http://localhost:8000"

    try:
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            health_response = await client.get(f"{base_url}/health")
            print(f"📊 Server Health: {health_response.json()}")

            # Send a test message
            print("\n💬 Sending test message to agent...")
            chat_response = await client.post(
                f"{base_url}/chat",
                json={
                    "message": "What is artificial intelligence and how does it work?",
                    "agent_type": "decision_tree"
                }
            )

            if chat_response.status_code == 200:
                result = chat_response.json()
                session_id = result["session_id"]
                print(f"✅ Response received! Session ID: {session_id[:8]}...")
                print(f"🤖 Agent Response: {result['response'][:100]}...")

                # Wait a moment for events to process
                await asyncio.sleep(2)

                # Get session summary
                summary_response = await client.get(f"{base_url}/sessions/{session_id}/summary")
                if summary_response.status_code == 200:
                    summary = summary_response.json()
                    metrics = summary.get("transparency_metrics", {})
                    print(f"\n📈 Transparency Metrics:")
                    print(f"   • Total decisions: {metrics.get('total_decisions', 0)}")
                    print(f"   • Tools used: {metrics.get('tools_used', 0)}")
                    print(f"   • Reasoning steps: {metrics.get('reasoning_steps', 0)}")
                    print(f"   • Data displays: {metrics.get('data_displays', 0)}")
                    print(f"   • Session duration: {metrics.get('session_duration', 0):.2f}s")

            else:
                print(f"❌ Error: {chat_response.status_code}")

    except Exception as e:
        print(f"❌ API Demo Error: {e}")
        print("💡 Make sure the server is running and try again")


def create_sample_data_demo():
    """Create sample data to demonstrate different renderers."""
    return {
        "table_data": [
            {"name": "Solar", "efficiency": 0.22, "cost_per_kwh": 0.08, "co2_reduction": 0.85},
            {"name": "Wind", "efficiency": 0.35, "cost_per_kwh": 0.06, "co2_reduction": 0.90},
            {"name": "Hydro", "efficiency": 0.80, "cost_per_kwh": 0.05, "co2_reduction": 0.95},
        ],
        "key_value_data": {
            "total_renewable_capacity": "2,500 GW",
            "annual_growth_rate": "12%",
            "investment_2023": "$1.8 trillion",
            "jobs_created": "13.7 million",
        },
        "chart_data": [45, 67, 89, 76, 92, 88, 95],
        "complex_data": {
            "regions": {
                "north_america": {"capacity": 500, "projects": 150},
                "europe": {"capacity": 800, "projects": 220},
                "asia": {"capacity": 1200, "projects": 350},
            },
            "technologies": ["solar", "wind", "hydro", "geothermal"],
            "timeline": ["2020", "2021", "2022", "2023", "2024"],
        },
    }


if __name__ == "__main__":
    print("🌟 Welcome to RAGents Transparency Demo!")
    print("This demo showcases real-time agent reasoning visualization.")
    print("\n🚨 Requirements:")
    print("• Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
    print("• Ensure port 8000 is available")
    print("• Modern web browser for best experience")

    # Check for API keys
    import os
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        exit(1)

    print("\n🎬 Starting demo...")

    try:
        asyncio.run(demo_transparency_workflow())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Check the console for more details")