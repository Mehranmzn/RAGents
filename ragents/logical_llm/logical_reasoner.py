"""Core logical reasoning engine for intelligent query processing."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole
from .domain_definitions import DomainDefinition, build_builtin_domains
from .models import (
    LogicalConstraint,
    LogicalOperator,
    LogicalQuery,
    RetrievalMode,
    SearchDirective,
)


class LogicalReasoner:
    """Core logical reasoning engine for query processing."""

    def __init__(
        self,
        llm_client: LLMClient,
        domain_registry: Optional[Dict[str, DomainDefinition]] = None,
    ):
        self.llm_client = llm_client
        self.domain_registry = domain_registry or build_builtin_domains()
        self.default_domain = "general"

    async def analyze_query(self, query: str) -> LogicalQuery:
        """Analyze a natural language query to extract logical structure."""
        domain = await self._detect_domain(query)
        intent = await self._extract_intent(query)
        context_summary, topic_tags = await self._summarize_context(query, domain)

        parameters = await self._extract_parameters(query, domain)
        constraints = self._get_domain_constraints(domain)
        missing_parameters = self._identify_missing_parameters(parameters, constraints)

        logical_form = await self._generate_logical_form(query, parameters, constraints)
        confidence_score = self._calculate_confidence(
            parameters, constraints, missing_parameters
        )
        refinement_suggestions = await self._generate_refinement_suggestions(
            query, missing_parameters, domain
        )

        logical_query = LogicalQuery(
            original_query=query,
            domain=domain,
            intent=intent,
            parameters=parameters,
            constraints=constraints,
            missing_parameters=missing_parameters,
            logical_form=logical_form,
            confidence_score=confidence_score,
            refinement_suggestions=refinement_suggestions,
            context_summary=context_summary,
            topic_tags=topic_tags,
        )
        logical_query.retrieval_directive = self.generate_search_directive(logical_query)
        return logical_query

    async def _detect_domain(self, query: str) -> str:
        """Detect the domain of the query using heuristics and LLM analysis."""
        keyword_scores = self._score_domains_by_keywords(query)
        best_keyword_domain = max(keyword_scores, key=keyword_scores.get) if keyword_scores else self.default_domain

        ranked_domains = sorted(
            keyword_scores.items(), key=lambda item: item[1], reverse=True
        )
        domain_options = [name for name, score in ranked_domains if score > 0]
        if not domain_options:
            domain_options = list(self.domain_registry.keys())

        llm_domain = await self._llm_domain_vote(query, domain_options)

        if llm_domain in self.domain_registry:
            return llm_domain
        return best_keyword_domain

    def _score_domains_by_keywords(self, query: str) -> Dict[str, float]:
        query_lower = query.lower()
        scores: Dict[str, float] = {}

        for name, definition in self.domain_registry.items():
            if not definition.keywords:
                scores[name] = 0.1  # small base score so general domain is an option
                continue

            match_count = sum(1 for keyword in definition.keywords if keyword in query_lower)
            scores[name] = match_count / len(definition.keywords)

            # Boost scores when domain-specific fields appear explicitly
            for field in definition.required_fields:
                if re.search(rf"\b{re.escape(field)}\b", query_lower):
                    scores[name] += 0.15

        return scores

    async def _llm_domain_vote(self, query: str, domain_candidates: Iterable[str]) -> str:
        """Ask the LLM to pick the most suitable domain from candidates."""
        candidate_text = "\n".join(f"- {name}" for name in domain_candidates)
        prompt = f"""
        Analyze the user's query and pick the best matching domain.

        Query: "{query}"

        Valid domain options:
        {candidate_text}

        Respond with a single domain name.
        """

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You classify queries into reasoning domains. Reply with only the domain name.",
            ),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            prediction = response.content.strip().lower()
            normalized = next(
                (name for name in self.domain_registry if name.lower() == prediction),
                None,
            )
            return normalized or self.default_domain
        except Exception:
            return self.default_domain

    async def _extract_intent(self, query: str) -> str:
        """Extract the intent/goal of the query."""
        intent_prompt = f"""
        Identify the user's primary intent for this query.

        Query: "{query}"

        Common intents: retrieve, compare, analyze, explain, predict, summarize, plan.
        Respond with a single intent keyword.
        """

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You extract concise intents from user queries.",
            ),
            ChatMessage(role=MessageRole.USER, content=intent_prompt),
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            return response.content.strip().lower()
        except Exception:
            return "retrieve"

    async def _summarize_context(self, query: str, domain: str) -> Tuple[Optional[str], List[str]]:
        """Generate a short context summary and topic tags for downstream use."""
        prompt = f"""
        Provide a very short (max 2 sentences) summary of the core context for this query
        so a retrieval system can stay on-topic. Then list 3-5 comma-separated topic tags.

        Format:
        Summary: <summary>
        Tags: tag1, tag2, tag3

        Query: "{query}"
        Domain hint: {domain}
        """

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You distill queries into compact summaries and topic tags.",
            ),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            summary = None
            tags: List[str] = []
            for line in response.content.splitlines():
                if line.lower().startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.lower().startswith("tags:"):
                    tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            return summary, tags
        except Exception:
            return None, []

    async def _extract_parameters(self, query: str, domain: str) -> Dict[str, Any]:
        definition = self.domain_registry.get(domain, self.domain_registry[self.default_domain])
        fields = list(dict.fromkeys(definition.required_fields + definition.optional_fields))
        if not fields:
            return {}

        field_text = "\n".join(f"- {field}" for field in fields)
        prompt = f"""
        Extract values for the listed fields from the query.

        Query: "{query}"
        Domain: {definition.name}

        Fields to check:
        {field_text}

        Reply with "field: value" for each field. Use MISSING when unknown.
        """

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You pull structured parameters from natural language queries.",
            ),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            parameters: Dict[str, Any] = {}
            for line in response.content.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value and value.upper() != "MISSING":
                    parameters[key] = value
            return parameters
        except Exception:
            return {}

    def _get_domain_constraints(self, domain: str) -> List[LogicalConstraint]:
        definition = self.domain_registry.get(domain, self.domain_registry[self.default_domain])
        return list(definition.constraints)

    def _identify_missing_parameters(
        self, parameters: Dict[str, Any], constraints: List[LogicalConstraint]
    ) -> List[str]:
        missing = []
        for constraint in constraints:
            if constraint.required and constraint.field_name not in parameters:
                missing.append(constraint.field_name)
        return missing

    async def _generate_logical_form(
        self,
        query: str,
        parameters: Dict[str, Any],
        constraints: List[LogicalConstraint],
    ) -> str:
        logical_elements: List[str] = []
        for param, value in parameters.items():
            logical_elements.append(f"{param}({value})")
        for constraint in constraints:
            if constraint.field_name in parameters:
                logical_elements.append(constraint.to_symbolic())
        if logical_elements:
            return " ∧ ".join(logical_elements)
        return f"Query({query})"

    def _calculate_confidence(
        self,
        parameters: Dict[str, Any],
        constraints: List[LogicalConstraint],
        missing_parameters: List[str],
    ) -> float:
        total_required = sum(1 for c in constraints if c.required)
        if total_required == 0:
            return 0.8 if parameters else 0.5

        satisfied = total_required - len(missing_parameters)
        base_confidence = satisfied / total_required
        extra_params = len(parameters) - satisfied
        bonus = min(0.2, extra_params * 0.05)
        return min(1.0, base_confidence + bonus)

    async def _generate_refinement_suggestions(
        self, query: str, missing_parameters: List[str], domain: str
    ) -> List[str]:
        if not missing_parameters:
            return []

        definition = self.domain_registry.get(domain, self.domain_registry[self.default_domain])
        suggestions: List[str] = []

        for parameter in missing_parameters:
            template = definition.clarification_templates.get(parameter)
            if template and template.questions:
                suggestions.append(template.questions[0])
            else:
                suggestions.append(f"Please provide more detail about {parameter}.")

        if len(suggestions) >= 3:
            return suggestions[:3]

        prompt = f"""
        The user asked: "{query}"
        Missing details: {', '.join(missing_parameters)}
        Domain: {domain}

        Suggest up to 3 concise follow-up questions that would gather the missing information.
        """

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You craft polite clarification questions to resolve missing details.",
            ),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            llm_suggestions = [line.strip("- ") for line in response.content.splitlines() if line.strip()]
            for suggestion in llm_suggestions:
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
            return suggestions[:5]
        except Exception:
            return suggestions[:3]

    async def refine_query(
        self, logical_query: LogicalQuery, additional_info: Dict[str, Any]
    ) -> LogicalQuery:
        updated_parameters = {**logical_query.parameters, **additional_info}
        updated_missing = self._identify_missing_parameters(
            updated_parameters, logical_query.constraints
        )
        updated_logical_form = await self._generate_logical_form(
            logical_query.original_query, updated_parameters, logical_query.constraints
        )
        updated_confidence = self._calculate_confidence(
            updated_parameters, logical_query.constraints, updated_missing
        )

        refined_query = LogicalQuery(
            original_query=logical_query.original_query,
            domain=logical_query.domain,
            intent=logical_query.intent,
            parameters=updated_parameters,
            constraints=logical_query.constraints,
            missing_parameters=updated_missing,
            logical_form=updated_logical_form,
            confidence_score=updated_confidence,
            refinement_suggestions=logical_query.refinement_suggestions,
            context_summary=logical_query.context_summary,
            topic_tags=logical_query.topic_tags,
        )
        refined_query.retrieval_directive = self.generate_search_directive(refined_query)
        return refined_query

    def generate_search_directive(self, logical_query: LogicalQuery) -> SearchDirective:
        """Generate a structured search directive for retrieval orchestration."""
        definition = self.domain_registry.get(
            logical_query.domain, self.domain_registry[self.default_domain]
        )
        query_text = self._compose_search_query(logical_query, definition)

        directive = SearchDirective(
            query_text=query_text,
            mode=definition.retrieval.default_mode,
            hybrid_alpha=(
                definition.retrieval.hybrid_alpha
                if definition.retrieval.default_mode in {RetrievalMode.HYBRID, RetrievalMode.GRAPH_HYBRID}
                else None
            ),
            graph_entry_points=[
                str(logical_query.parameters[field])
                for field in definition.retrieval.graph_focus_entities
                if field in logical_query.parameters
            ],
            notes=list(definition.retrieval.notes),
            metadata={
                "domain": logical_query.domain,
                "intent": logical_query.intent,
                "context_summary": logical_query.context_summary,
                "topic_tags": logical_query.topic_tags,
            },
        )

        if definition.retrieval.enable_graph_enrichment:
            directive.graph_query = self._compose_graph_query(logical_query, definition)
            if definition.retrieval.default_mode == RetrievalMode.GRAPH_HYBRID:
                directive.mode = RetrievalMode.GRAPH_HYBRID

        return directive

    def _compose_search_query(
        self, logical_query: LogicalQuery, definition: DomainDefinition
    ) -> str:
        if not logical_query.parameters:
            return logical_query.original_query

        parts: List[str] = []
        for key in definition.required_fields + definition.optional_fields:
            value = logical_query.parameters.get(key)
            if value:
                parts.append(str(value))
        if logical_query.context_summary:
            parts.append(logical_query.context_summary)
        if logical_query.topic_tags:
            parts.extend(logical_query.topic_tags)

        combined = " ".join(dict.fromkeys(parts))
        return combined if combined.strip() else logical_query.original_query

    def _compose_graph_query(
        self, logical_query: LogicalQuery, definition: DomainDefinition
    ) -> Optional[str]:
        if not definition.retrieval.enable_graph_enrichment:
            return None

        graph_parts = []
        for field in definition.retrieval.graph_focus_entities:
            if field in logical_query.parameters:
                graph_parts.append(f"{field}:{logical_query.parameters[field]}")
        if logical_query.intent in {"compare", "analyze"}:
            graph_parts.append(f"intent:{logical_query.intent}")
        return " AND ".join(graph_parts) if graph_parts else None

    def generate_focused_search_query(self, logical_query: LogicalQuery) -> str:
        """Maintain backwards compatibility by exposing query text."""
        directive = logical_query.retrieval_directive or self.generate_search_directive(logical_query)
        return directive.query_text
