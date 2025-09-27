"""Advanced LangGraph workflow examples for RAGents."""

import asyncio
import os
from typing import Dict, Any, List
from pathlib import Path

from ragents import (
    LLMClient,
    RAGEngine,
    RAGConfig,
    LangGraphAgent,
    LangGraphReActAgent,
    LangGraphMultiAgent,
    AgentConfig,
    AgentRole,
    AgentDefinition,
)
from ragents.llm.types import ModelConfig, ModelProvider
from ragents.tools.base import ToolRegistry


class DocumentAnalysisWorkflow:
    """Custom workflow for document analysis using LangGraph agents."""

    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.setup_agents()

    def setup_agents(self):
        """Setup specialized agents for document analysis."""
        # Document extractor agent
        extractor_config = AgentConfig(
            name="DocumentExtractor",
            description="Extracts key information and concepts from documents",
            enable_rag=True,
            enable_reasoning=True
        )
        self.extractor = LangGraphAgent(
            extractor_config, self.llm_client, self.rag_engine
        )

        # Summarizer agent
        summarizer_config = AgentConfig(
            name="DocumentSummarizer",
            description="Creates concise summaries of document content",
            enable_reasoning=True,
            reasoning_depth=3
        )
        self.summarizer = LangGraphAgent(
            summarizer_config, self.llm_client
        )

        # Analyzer agent (ReAct for deep analysis)
        analyzer_config = AgentConfig(
            name="DocumentAnalyzer",
            description="Performs deep analysis of document themes and patterns",
            enable_rag=True,
            enable_tools=True,
            max_iterations=8
        )
        self.analyzer = LangGraphReActAgent(
            analyzer_config, self.llm_client, self.rag_engine
        )

    async def analyze_document(self, document_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        results = {}

        # Step 1: Extract key information
        extraction_query = f"Extract the main topics, key concepts, and important details from the document at {document_path}"
        extraction_result = await self.extractor.process_message(
            extraction_query, thread_id=f"extract_{Path(document_path).stem}"
        )
        results["extraction"] = {
            "content": extraction_result.response,
            "metadata": extraction_result.execution_metadata
        }

        # Step 2: Create summary
        summary_query = f"Based on the extracted information, create a concise summary of the document's main points"
        summary_result = await self.summarizer.process_message(
            summary_query, thread_id=f"summary_{Path(document_path).stem}"
        )
        results["summary"] = {
            "content": summary_result.response,
            "metadata": summary_result.execution_metadata
        }

        # Step 3: Deep analysis (if comprehensive)
        if analysis_type == "comprehensive":
            analysis_query = f"Perform a detailed analysis of the document themes, arguments, and conclusions. Look for patterns, biases, and connections to other concepts."
            analysis_result = await self.analyzer.process_message(
                analysis_query, thread_id=f"analyze_{Path(document_path).stem}"
            )
            results["analysis"] = {
                "content": analysis_result.response,
                "metadata": analysis_result.execution_metadata,
                "reasoning_steps": analysis_result.state.get("action_history", [])
            }

        return results


class ResearchPipelineWorkflow:
    """Multi-agent research pipeline using LangGraph."""

    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.setup_research_team()

    def setup_research_team(self):
        """Setup a comprehensive research team."""
        coordinator_config = AgentConfig(
            name="ResearchCoordinator",
            description="Coordinates complex research projects"
        )
        self.coordinator = LangGraphMultiAgent(coordinator_config, self.llm_client)

        # Information gatherer (ReAct agent)
        gatherer_config = AgentConfig(
            name="InformationGatherer",
            description="Gathers information from multiple sources",
            enable_rag=True,
            enable_tools=True,
            max_iterations=12
        )
        self.coordinator.register_agent(AgentDefinition(
            name="info_gatherer",
            role=AgentRole.RESEARCHER,
            config=gatherer_config,
            llm_client=self.llm_client,
            rag_engine=self.rag_engine
        ))

        # Data analyst
        analyst_config = AgentConfig(
            name="DataAnalyst",
            description="Analyzes research data and identifies patterns",
            enable_reasoning=True,
            reasoning_depth=5
        )
        self.coordinator.register_agent(AgentDefinition(
            name="data_analyst",
            role=AgentRole.ANALYST,
            config=analyst_config,
            llm_client=self.llm_client,
            rag_engine=self.rag_engine
        ))

        # Synthesis specialist
        synthesis_config = AgentConfig(
            name="SynthesisSpecialist",
            description="Synthesizes research findings into coherent insights",
            enable_reasoning=True
        )
        self.coordinator.register_agent(AgentDefinition(
            name="synthesizer",
            role=AgentRole.SPECIALIST,
            config=synthesis_config,
            llm_client=self.llm_client
        ))

        # Research writer
        writer_config = AgentConfig(
            name="ResearchWriter",
            description="Creates well-structured research reports"
        )
        self.coordinator.register_agent(AgentDefinition(
            name="research_writer",
            role=AgentRole.WRITER,
            config=writer_config,
            llm_client=self.llm_client
        ))

    async def conduct_research(self, research_question: str, depth: str = "standard") -> Dict[str, Any]:
        """Conduct comprehensive research on a topic."""
        # Adjust the research approach based on depth
        if depth == "deep":
            enhanced_question = f"""
            Conduct an in-depth research investigation on: {research_question}

            Requirements:
            - Gather comprehensive information from multiple perspectives
            - Analyze data patterns and trends
            - Identify key insights and implications
            - Synthesize findings into actionable conclusions
            - Create a detailed research report
            """
        else:
            enhanced_question = f"""
            Research the following topic: {research_question}

            Provide a well-researched analysis with key findings and insights.
            """

        result = await self.coordinator.process_message(
            enhanced_question,
            thread_id=f"research_{hash(research_question) % 10000}"
        )

        return {
            "research_report": result.response,
            "methodology": {
                "routing_decision": result.execution_metadata.get("routing_decision"),
                "agents_used": result.execution_metadata.get("agents_used", []),
                "collaboration_rounds": result.execution_metadata.get("collaboration_rounds", 0)
            },
            "execution_metadata": result.execution_metadata
        }


class CodeAnalysisWorkflow:
    """Specialized workflow for code analysis and review."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.setup_code_agents()

    def setup_code_agents(self):
        """Setup agents specialized for code analysis."""
        # Code reviewer
        reviewer_config = AgentConfig(
            name="CodeReviewer",
            description="Reviews code for quality, bugs, and best practices",
            enable_reasoning=True,
            reasoning_depth=4
        )
        self.reviewer = LangGraphAgent(reviewer_config, self.llm_client)

        # Security analyst
        security_config = AgentConfig(
            name="SecurityAnalyst",
            description="Analyzes code for security vulnerabilities",
            enable_reasoning=True,
            reasoning_depth=5
        )
        self.security_analyst = LangGraphAgent(security_config, self.llm_client)

        # Performance analyzer (ReAct for systematic analysis)
        performance_config = AgentConfig(
            name="PerformanceAnalyzer",
            description="Analyzes code performance and optimization opportunities",
            max_iterations=10
        )
        self.performance_analyzer = LangGraphReActAgent(
            performance_config, self.llm_client
        )

    async def analyze_code(self, code: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive code analysis."""
        if analysis_types is None:
            analysis_types = ["quality", "security", "performance"]

        results = {}
        thread_base = f"code_analysis_{hash(code) % 10000}"

        # Quality review
        if "quality" in analysis_types:
            quality_query = f"""
            Review this code for quality, readability, and best practices:

            ```
            {code}
            ```

            Provide feedback on:
            - Code structure and organization
            - Naming conventions
            - Error handling
            - Documentation
            - Adherence to best practices
            """
            quality_result = await self.reviewer.process_message(
                quality_query, thread_id=f"{thread_base}_quality"
            )
            results["quality_review"] = {
                "feedback": quality_result.response,
                "metadata": quality_result.execution_metadata
            }

        # Security analysis
        if "security" in analysis_types:
            security_query = f"""
            Analyze this code for security vulnerabilities:

            ```
            {code}
            ```

            Look for:
            - Input validation issues
            - SQL injection risks
            - XSS vulnerabilities
            - Authentication/authorization flaws
            - Data exposure risks
            """
            security_result = await self.security_analyst.process_message(
                security_query, thread_id=f"{thread_base}_security"
            )
            results["security_analysis"] = {
                "findings": security_result.response,
                "metadata": security_result.execution_metadata
            }

        # Performance analysis
        if "performance" in analysis_types:
            performance_query = f"""
            Analyze this code for performance issues and optimization opportunities:

            ```
            {code}
            ```

            Investigate:
            - Algorithmic complexity
            - Memory usage patterns
            - I/O operations
            - Caching opportunities
            - Bottlenecks and optimization points
            """
            performance_result = await self.performance_analyzer.process_message(
                performance_query, thread_id=f"{thread_base}_performance"
            )
            results["performance_analysis"] = {
                "recommendations": performance_result.response,
                "metadata": performance_result.execution_metadata,
                "analysis_steps": performance_result.state.get("action_history", [])
            }

        return results


class CustomerSupportWorkflow:
    """Customer support workflow using LangGraph agents."""

    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.setup_support_agents()

    def setup_support_agents(self):
        """Setup customer support agent team."""
        # First-line support agent
        frontline_config = AgentConfig(
            name="FrontlineSupport",
            description="Handles initial customer inquiries and common issues",
            enable_rag=True,  # Access to knowledge base
            enable_reasoning=True
        )
        self.frontline = LangGraphAgent(
            frontline_config, self.llm_client, self.rag_engine
        )

        # Technical specialist (ReAct for complex troubleshooting)
        tech_config = AgentConfig(
            name="TechnicalSpecialist",
            description="Handles complex technical issues requiring investigation",
            enable_rag=True,
            enable_tools=True,
            max_iterations=15
        )
        # Add debugging and diagnostic tools to tool registry if available
        self.tech_specialist = LangGraphReActAgent(
            tech_config, self.llm_client, self.rag_engine
        )

        # Escalation coordinator
        escalation_config = AgentConfig(
            name="EscalationCoordinator",
            description="Coordinates escalated issues and specialist routing"
        )
        self.escalation_coordinator = LangGraphMultiAgent(escalation_config, self.llm_client)

    async def handle_support_request(self, customer_message: str, priority: str = "normal") -> Dict[str, Any]:
        """Handle a customer support request."""
        # Classify the request complexity
        classification_prompt = f"""
        Classify this customer support request:

        Customer Message: {customer_message}

        Classify as:
        - "simple": Can be handled with standard responses/knowledge base
        - "technical": Requires technical investigation or troubleshooting
        - "escalation": Needs specialist attention or is highly complex

        Also identify the main issue category (e.g., billing, technical, account, etc.)
        """

        classification_result = await self.frontline.process_message(
            classification_prompt, thread_id=f"classify_{hash(customer_message) % 10000}"
        )

        classification = classification_result.response.lower()

        # Route based on classification
        if "simple" in classification:
            # Handle with frontline agent
            response_result = await self.frontline.process_message(
                f"Provide a helpful response to this customer inquiry: {customer_message}",
                thread_id=f"frontline_{hash(customer_message) % 10000}"
            )

            return {
                "response": response_result.response,
                "handling_agent": "frontline",
                "classification": "simple",
                "metadata": response_result.execution_metadata
            }

        elif "technical" in classification:
            # Handle with technical specialist
            technical_response = await self.tech_specialist.process_message(
                f"Investigate and resolve this technical issue: {customer_message}",
                thread_id=f"tech_{hash(customer_message) % 10000}"
            )

            return {
                "response": technical_response.response,
                "handling_agent": "technical_specialist",
                "classification": "technical",
                "investigation_steps": technical_response.state.get("action_history", []),
                "metadata": technical_response.execution_metadata
            }

        else:
            # Handle escalation
            escalation_response = await self.escalation_coordinator.process_message(
                f"Coordinate resolution for this escalated customer issue: {customer_message}",
                thread_id=f"escalation_{hash(customer_message) % 10000}"
            )

            return {
                "response": escalation_response.response,
                "handling_agent": "escalation_team",
                "classification": "escalation",
                "coordination_details": escalation_response.execution_metadata,
                "metadata": escalation_response.execution_metadata
            }


async def demo_document_analysis():
    """Demonstrate document analysis workflow."""
    print("=== Document Analysis Workflow Demo ===")

    # Mock setup (replace with actual implementations)
    llm_client = LLMClient(ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key")
    ))

    rag_config = RAGConfig(vector_store_type="memory", collection_name="docs")
    rag_engine = RAGEngine(rag_config, llm_client)

    workflow = DocumentAnalysisWorkflow(llm_client, rag_engine)

    try:
        # Simulate document analysis
        results = await workflow.analyze_document(
            "/path/to/document.pdf",
            analysis_type="comprehensive"
        )

        print("Document Analysis Results:")
        print(f"- Extraction: {len(results.get('extraction', {}).get('content', ''))} characters")
        print(f"- Summary: {len(results.get('summary', {}).get('content', ''))} characters")
        if 'analysis' in results:
            print(f"- Deep Analysis: {len(results['analysis']['content'])} characters")
            print(f"- Reasoning Steps: {len(results['analysis']['reasoning_steps'])}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")


async def demo_research_pipeline():
    """Demonstrate research pipeline workflow."""
    print("=== Research Pipeline Workflow Demo ===")

    # Mock setup
    llm_client = LLMClient(ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key")
    ))

    rag_config = RAGConfig(vector_store_type="memory", collection_name="research")
    rag_engine = RAGEngine(rag_config, llm_client)

    workflow = ResearchPipelineWorkflow(llm_client, rag_engine)

    try:
        results = await workflow.conduct_research(
            "What are the environmental impacts of artificial intelligence?",
            depth="deep"
        )

        print("Research Results:")
        print(f"- Report Length: {len(results['research_report'])} characters")
        print(f"- Methodology: {results['methodology']}")
        print(f"- Agents Used: {results['methodology']['agents_used']}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")


async def main():
    """Run workflow demonstrations."""
    print("🔬 RAGents Advanced LangGraph Workflows")
    print("=" * 50)

    workflows = [
        demo_document_analysis,
        demo_research_pipeline,
    ]

    for workflow_demo in workflows:
        try:
            await workflow_demo()
        except Exception as e:
            print(f"Workflow {workflow_demo.__name__} failed: {e}")
        print("-" * 30)

    print("✅ All workflow demos completed!")


if __name__ == "__main__":
    asyncio.run(main())