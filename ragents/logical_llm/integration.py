"""Integration layer for Logical LLM with RAGents agents and workflows."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .logical_reasoner import LogicalReasoner
from .models import LogicalQuery, SearchDirective
from .query_clarifier import QueryClarifier, ClarificationRequest, ClarificationResponse
from .constraint_engine import ConstraintEngine, ConstraintViolation
from .logic_patterns import BuiltinPatterns, PatternMatcher, analyze_query_complexity

from ..llm.client import LLMClient
from ..rag.engine import RAGEngine
from ..rag.types import QueryContext
from ..agents.base import Agent, AgentConfig


@dataclass
class LogicalProcessingResult:
    """Result from logical processing of a query."""

    original_query: str
    logical_query: LogicalQuery
    clarification_requests: List[ClarificationRequest]
    constraint_violations: List[ConstraintViolation]
    optimized_search_query: str
    search_directive: SearchDirective
    estimated_token_reduction: float
    processing_confidence: float
    should_proceed: bool
    interactive_mode: bool = False


class LogicalLLMIntegration:
    """Integration layer for Logical LLM with RAGents framework."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logical_reasoner = LogicalReasoner(llm_client)
        self.query_clarifier = QueryClarifier(llm_client, self.logical_reasoner)
        self.constraint_engine = ConstraintEngine(
            llm_client, self.logical_reasoner.domain_registry
        )
        self.pattern_matcher = BuiltinPatterns.create_pattern_matcher()

    async def process_query(self, query: str, interactive: bool = True) -> LogicalProcessingResult:
        """Process a query through the logical LLM pipeline."""
        # Step 1: Analyze query complexity and patterns
        patterns = BuiltinPatterns.create_all_patterns()
        complexity_analysis = analyze_query_complexity(query, patterns)

        # Step 2: Perform logical analysis
        logical_query = await self.logical_reasoner.analyze_query(query)

        # Step 3: Generate clarification requests if needed
        clarification_requests = []
        if not logical_query.is_complete() and interactive:
            _, clarifications = await self.query_clarifier.analyze_and_clarify(query)
            clarification_requests = clarifications

        # Step 4: Apply constraint engine
        optimized_query, violations = await self.constraint_engine.process_logical_query(logical_query)

        # Step 5: Generate optimized search directive
        search_directive = self.logical_reasoner.generate_search_directive(optimized_query)
        search_query = search_directive.query_text

        # Step 6: Determine if we should proceed or need clarification
        should_proceed = self._should_proceed_with_query(optimized_query, clarification_requests, violations)

        return LogicalProcessingResult(
            original_query=query,
            logical_query=optimized_query,
            clarification_requests=clarification_requests,
            constraint_violations=violations,
            optimized_search_query=search_query,
            search_directive=search_directive,
            estimated_token_reduction=complexity_analysis["estimated_token_reduction"],
            processing_confidence=optimized_query.confidence_score,
            should_proceed=should_proceed,
            interactive_mode=interactive
        )

    async def handle_clarification_response(self,
                                          result: LogicalProcessingResult,
                                          clarification_responses: Dict[str, str]) -> LogicalProcessingResult:
        """Handle user responses to clarification requests."""
        if not clarification_responses:
            return result

        # Process each clarification response
        additional_info = {}
        for field_name, user_response in clarification_responses.items():
            # Find corresponding clarification request
            request = next((req for req in result.clarification_requests
                          if req.field_name == field_name), None)

            if request:
                clarification_response = await self.query_clarifier.process_clarification_response(
                    request, user_response
                )
                additional_info[field_name] = clarification_response.extracted_value

        # Refine the logical query with additional information
        refined_query = await self.logical_reasoner.refine_query(
            result.logical_query, additional_info
        )

        # Reprocess through constraint engine
        optimized_query, violations = await self.constraint_engine.process_logical_query(refined_query)

        # Generate new optimized search query
        search_directive = self.logical_reasoner.generate_search_directive(optimized_query)
        search_query = search_directive.query_text

        # Update result
        return LogicalProcessingResult(
            original_query=result.original_query,
            logical_query=optimized_query,
            clarification_requests=[],  # Should be resolved now
            constraint_violations=violations,
            optimized_search_query=search_query,
            search_directive=search_directive,
            estimated_token_reduction=result.estimated_token_reduction,
            processing_confidence=optimized_query.confidence_score,
            should_proceed=True,
            interactive_mode=result.interactive_mode
        )

    def _should_proceed_with_query(self,
                                 logical_query: LogicalQuery,
                                 clarifications: List[ClarificationRequest],
                                 violations: List[ConstraintViolation]) -> bool:
        """Determine if query processing should proceed or needs clarification."""
        # Don't proceed if there are critical violations
        critical_violations = [v for v in violations if v.severity.value <= 2]
        if critical_violations:
            return False

        # Don't proceed if confidence is too low
        if logical_query.confidence_score < 0.3:
            return False

        # Don't proceed if there are high-priority clarification requests
        high_priority_clarifications = [c for c in clarifications if c.priority <= 2]
        if high_priority_clarifications:
            return False

        return True


class LogicalAgent(Agent):
    """Agent enhanced with logical LLM capabilities."""

    def __init__(self,
                 config: AgentConfig,
                 llm_client: LLMClient,
                 rag_engine: Optional[RAGEngine] = None,
                 enable_logical_processing: bool = True):
        super().__init__(config, llm_client, rag_engine)
        self.enable_logical_processing = enable_logical_processing

        if enable_logical_processing:
            self.logical_integration = LogicalLLMIntegration(llm_client)

    async def process_message(self, message: str) -> str:
        """Process a message with logical LLM enhancement."""
        if not self.enable_logical_processing:
            return await self._standard_processing(message)

        # Step 1: Process through logical LLM
        logical_result = await self.logical_integration.process_query(message, interactive=False)

        # Step 2: Handle different scenarios
        if not logical_result.should_proceed:
            # Generate clarification response
            return await self._generate_clarification_response(logical_result)

        elif logical_result.estimated_token_reduction > 0.2:
            # Use optimized query for RAG search
            return await self._optimized_processing(logical_result)

        else:
            # Use standard processing with slight optimizations
            return await self._enhanced_processing(logical_result)

    async def _generate_clarification_response(self, result: LogicalProcessingResult) -> str:
        """Generate a response asking for clarification."""
        if result.clarification_requests:
            # Use the first high-priority clarification request
            primary_request = result.clarification_requests[0]
            return primary_request.to_user_prompt()

        elif result.constraint_violations:
            # Address constraint violations
            violation = result.constraint_violations[0]
            if violation.suggested_fix:
                return f"I need more specific information: {violation.suggested_fix}"
            else:
                return f"I need clarification: {violation.message}"

        else:
            return "I need more specific information to provide an accurate answer. Could you please provide more details?"

    async def _optimized_processing(self, result: LogicalProcessingResult) -> str:
        """Process using optimized query for significant token reduction."""
        if self.rag_engine:
            # Use the optimized search directive instead of original
            context = self._build_query_context(result.search_directive, result.original_query)
            rag_response = await self.rag_engine.query(
                result.search_directive.query_text, context=context
            )

            # Generate response with context
            context_prompt = f"""
            Based on this specific query: {result.original_query}

            I found the following relevant information using optimized search:
            {rag_response.answer}

            Please provide a comprehensive answer that directly addresses the user's question.
            """

            from ..llm.types import ChatMessage, MessageRole
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=await self._get_system_prompt()),
                ChatMessage(role=MessageRole.USER, content=context_prompt)
            ]

            response = await self.llm_client.acomplete(messages)
            return response.content

        else:
            # Generate response using logical structure
            return await self._generate_logical_response(result)

    async def _enhanced_processing(self, result: LogicalProcessingResult) -> str:
        """Process with logical enhancements but standard flow."""
        if self.rag_engine:
            # Use both original and optimized queries for comprehensive search
            original_response = await self.rag_engine.query(result.original_query)

            if result.optimized_search_query != result.original_query:
                optimized_context = self._build_query_context(
                    result.search_directive, result.original_query
                )
                optimized_response = await self.rag_engine.query(
                    result.search_directive.query_text, context=optimized_context
                )

                # Combine responses
                combined_context = f"""
                General search results: {original_response.answer}

                Focused search results: {optimized_response.answer}
                """
            else:
                combined_context = original_response.answer

            # Generate enhanced response
            from ..llm.types import ChatMessage, MessageRole
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=await self._get_system_prompt()),
                ChatMessage(role=MessageRole.USER, content=f"Query: {result.original_query}\n\nContext: {combined_context}")
            ]

            response = await self.llm_client.acomplete(messages)
            return response.content

        else:
            return await self._generate_logical_response(result)

    async def _standard_processing(self, message: str) -> str:
        """Standard processing without logical LLM."""
        if self.rag_engine and await self._should_query_rag(message):
            rag_response = await self.rag_engine.query(message)
            context_info = f"\n\nRelevant Information:\n{rag_response.answer}"
        else:
            context_info = ""

        from ..llm.types import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=await self._get_system_prompt()),
            ChatMessage(role=MessageRole.USER, content=message + context_info)
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content

    async def _generate_logical_response(self, result: LogicalProcessingResult) -> str:
        """Generate response based on logical structure."""
        logical_query = result.logical_query

        # Build structured response
        response_parts = []

        if logical_query.parameters:
            param_info = []
            for param, value in logical_query.parameters.items():
                param_info.append(f"{param}: {value}")

            response_parts.append(f"Based on your query about {', '.join(param_info)}")

        if logical_query.domain:
            response_parts.append(f"in the {logical_query.domain} domain")

        # Add logical reasoning
        if logical_query.logical_form:
            response_parts.append(f"(logical form: {logical_query.logical_form})")

        # Combine and generate natural response
        structured_info = " ".join(response_parts)

        from ..llm.types import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM,
                       content="Generate a helpful response based on the structured query analysis."),
            ChatMessage(role=MessageRole.USER,
                       content=f"Original query: {result.original_query}\nAnalysis: {structured_info}")
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content

    async def _should_query_rag(self, message: str) -> bool:
        """Determine if we should query the RAG system for this message."""
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        return any(word in message.lower() for word in question_words) or "?" in message

    def _build_query_context(
        self, directive: Optional[SearchDirective], original_query: str
    ) -> QueryContext:
        """Create a QueryContext object from the search directive."""
        if directive is None:
            return QueryContext(original_query=original_query)

        filters = dict(directive.filters) if directive.filters else {}
        if directive.graph_entry_points:
            filters.setdefault("_graph_entry_points", directive.graph_entry_points)
        if directive.graph_query:
            filters.setdefault("_graph_query", directive.graph_query)
        if directive.hybrid_alpha is not None:
            filters.setdefault("_hybrid_alpha", directive.hybrid_alpha)

        context = QueryContext(
            original_query=original_query,
            filters=filters,
            retrieval_strategy=directive.mode.value,
        )
        return context


# Utility functions for easy integration

async def enhance_agent_with_logical_llm(agent: Agent, enable: bool = True) -> Agent:
    """Enhance an existing agent with logical LLM capabilities."""
    if hasattr(agent, 'enable_logical_processing'):
        agent.enable_logical_processing = enable
        if enable and not hasattr(agent, 'logical_integration'):
            agent.logical_integration = LogicalLLMIntegration(agent.llm_client)

    return agent


def create_logical_agent(config: AgentConfig,
                        llm_client: LLMClient,
                        rag_engine: Optional[RAGEngine] = None) -> LogicalAgent:
    """Create a new logical agent with LLM enhancement."""
    return LogicalAgent(config, llm_client, rag_engine, enable_logical_processing=True)


async def analyze_token_reduction_potential(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """Analyze the potential token reduction for a query."""
    integration = LogicalLLMIntegration(llm_client)
    result = await integration.process_query(query, interactive=False)

    original_tokens = len(query.split()) * 1.3  # Rough token estimate
    optimized_tokens = len(result.optimized_search_query.split()) * 1.3

    return {
        "original_query": query,
        "optimized_query": result.optimized_search_query,
        "estimated_original_tokens": int(original_tokens),
        "estimated_optimized_tokens": int(optimized_tokens),
        "token_reduction": max(0, original_tokens - optimized_tokens),
        "token_reduction_percentage": max(0, (original_tokens - optimized_tokens) / original_tokens * 100),
        "confidence_score": result.processing_confidence,
        "logical_domain": result.logical_query.domain,
        "requires_clarification": not result.should_proceed,
        "retrieval_mode": result.search_directive.mode.value,
        "graph_entry_points": result.search_directive.graph_entry_points,
    }
