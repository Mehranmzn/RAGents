"""Query clarifier for interactive refinement of ambiguous queries."""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .domain_definitions import ClarificationTemplate
from .logical_reasoner import LogicalReasoner
from .models import LogicalConstraint, LogicalQuery
from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole


class ClarificationType(Enum):
    """Types of clarification requests."""
    MISSING_PARAMETER = "missing_parameter"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"
    DOMAIN_CLARIFICATION = "domain_clarification"


@dataclass
class ClarificationRequest:
    """Represents a request for clarification."""

    type: ClarificationType
    field_name: str
    question: str
    options: List[str] = field(default_factory=list)
    context: str = ""
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    examples: List[str] = field(default_factory=list)

    def to_user_prompt(self) -> str:
        """Convert clarification request to user-friendly prompt."""
        prompt = self.question

        if self.options:
            prompt += "\n\nOptions:"
            for i, option in enumerate(self.options, 1):
                prompt += f"\n{i}. {option}"

        if self.examples:
            prompt += "\n\nExamples:"
            for example in self.examples:
                prompt += f"\n- {example}"

        return prompt


@dataclass
class ClarificationResponse:
    """Response to a clarification request."""

    original_request: ClarificationRequest
    user_response: str
    extracted_value: Any
    confidence: float
    additional_context: Dict[str, Any] = field(default_factory=dict)


class QueryClarifier:
    """Handles interactive clarification of ambiguous queries."""

    def __init__(self, llm_client: LLMClient, logical_reasoner: LogicalReasoner):
        self.llm_client = llm_client
        self.logical_reasoner = logical_reasoner
        self.domain_registry = logical_reasoner.domain_registry
        self.clarification_patterns = self._initialize_clarification_patterns()

    def _initialize_clarification_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for generating clarification requests."""
        base_patterns: Dict[str, Dict[str, Any]] = {
            "entity": {
                "questions": [
                    "Which entity or subject should I focus on?",
                    "Can you specify the main person, product, or organization?",
                ],
                "context_hints": ["who", "what", "which"],
                "examples": ["satellite imagery dataset", "customer onboarding workflow", "Team Phoenix"],
                "priority": 1,
            },
            "subject": {
                "questions": [
                    "What is the subject of interest?",
                    "Please clarify what this request is about.",
                ],
                "context_hints": ["topic", "subject", "focus"],
                "examples": ["incident response runbook", "machine learning pipeline"],
                "priority": 1,
            },
            "time_reference": {
                "questions": [
                    "What time frame should I consider?",
                    "Do you have a specific date or range in mind?",
                ],
                "context_hints": ["when", "recent", "historical", "timeline"],
                "examples": ["past week", "Q1 2024", "January 2023", "last 6 months"],
                "priority": 1,
            },
            "metric": {
                "questions": [
                    "Which metric should we analyze?",
                    "What measurement are you looking at?",
                ],
                "examples": ["latency", "conversion rate", "customer satisfaction"],
                "priority": 1,
            },
            "location": {
                "questions": [
                    "Which location or region does this concern?",
                    "Should I focus on a particular geography?",
                ],
                "examples": ["North America", "eu-west-1", "headquarters"],
                "priority": 2,
            },
        }

        for definition in self.domain_registry.values():
            for field, template in definition.clarification_templates.items():
                existing = base_patterns.get(field, {})
                merged = self._merge_template(existing, template)
                base_patterns[field] = merged

        return base_patterns

    def _merge_template(
        self, base: Dict[str, Any], template: ClarificationTemplate
    ) -> Dict[str, Any]:
        """Merge domain clarification template into base pattern."""
        merged = {
            "questions": list(dict.fromkeys((base.get("questions") or []) + template.questions)),
            "options": list(dict.fromkeys((base.get("options") or []) + template.options)),
            "examples": list(dict.fromkeys((base.get("examples") or []) + template.examples)),
            "context_hints": list(dict.fromkeys((base.get("context_hints") or []) + template.context_hints)),
            "priority": template.priority or base.get("priority", 2),
        }
        return merged

    async def analyze_and_clarify(self, query: str) -> Tuple[LogicalQuery, List[ClarificationRequest]]:
        """Analyze query and generate clarification requests for missing information."""
        # First, perform logical analysis
        logical_query = await self.logical_reasoner.analyze_query(query)

        # Generate clarification requests for missing parameters
        clarification_requests = await self._generate_clarification_requests(logical_query)

        return logical_query, clarification_requests

    async def _generate_clarification_requests(self, logical_query: LogicalQuery) -> List[ClarificationRequest]:
        """Generate clarification requests for missing parameters."""
        requests = []

        # Generate requests for missing required parameters
        for missing_param in logical_query.missing_parameters:
            request = await self._create_clarification_request(missing_param, logical_query)
            if request:
                requests.append(request)

        # Check for ambiguous references in existing parameters
        ambiguous_requests = await self._detect_ambiguous_references(logical_query)
        requests.extend(ambiguous_requests)

        # Sort by priority (high priority first)
        requests.sort(key=lambda r: r.priority)

        return requests

    async def _create_clarification_request(self, parameter: str, logical_query: LogicalQuery) -> Optional[ClarificationRequest]:
        """Create a clarification request for a specific parameter."""
        pattern = self.clarification_patterns.get(parameter)
        if not pattern:
            # Generate generic clarification for unknown parameters
            return ClarificationRequest(
                type=ClarificationType.MISSING_PARAMETER,
                field_name=parameter,
                question=f"Please provide more information about {parameter}.",
                priority=2
            )

        # Choose appropriate question based on context
        questions = pattern.get("questions", [])
        question = questions[0] if questions else f"Please specify {parameter}."

        # Get context-appropriate options
        options = pattern.get("options", [])
        examples = pattern.get("examples", [])

        # Determine priority based on parameter importance
        priority = self._get_parameter_priority(parameter, logical_query.domain)

        # Add domain-specific context
        context = await self._generate_parameter_context(parameter, logical_query)

        return ClarificationRequest(
            type=ClarificationType.MISSING_PARAMETER,
            field_name=parameter,
            question=question,
            options=options,
            context=context,
            priority=priority,
            examples=examples
        )

    def _get_parameter_priority(self, parameter: str, domain: str) -> int:
        """Determine priority of parameter based on domain."""
        definition = self.domain_registry.get(domain, self.domain_registry.get("general"))
        if definition:
            if parameter in definition.required_fields:
                return 1
            if parameter in definition.optional_fields:
                return 2

        pattern = self.clarification_patterns.get(parameter, {})
        return pattern.get("priority", 3)

    async def _generate_parameter_context(self, parameter: str, logical_query: LogicalQuery) -> str:
        """Generate contextual information for parameter clarification."""
        context_parts = []

        # Add domain context
        definition = self.domain_registry.get(
            logical_query.domain, self.domain_registry.get("general")
        )
        if definition:
            context_parts.append(definition.description)

            template = definition.clarification_templates.get(parameter)
            if template and template.context_hints:
                context_parts.append(
                    f"Hints: {', '.join(template.context_hints[:3])}"
                )

        # Add intent context
        if logical_query.intent == "compare":
            context_parts.append(
                "Comparisons work best when the entities and criteria are explicit."
            )
        elif logical_query.intent == "analyze":
            context_parts.append("Detailed context improves the quality of the analysis.")
        elif logical_query.intent == "explain":
            context_parts.append(
                "Clarifying underlying factors helps generate a meaningful explanation."
            )

        # Add constraint context
        relevant_constraints = [c for c in logical_query.constraints if c.field_name == parameter]
        for constraint in relevant_constraints:
            context_parts.append(constraint.description)

        return " ".join(context_parts)

    async def _detect_ambiguous_references(self, logical_query: LogicalQuery) -> List[ClarificationRequest]:
        """Detect ambiguous references in existing parameters."""
        ambiguous_requests = []

        for param, value in logical_query.parameters.items():
            ambiguity_request = await self._check_parameter_ambiguity(param, value, logical_query)
            if ambiguity_request:
                ambiguous_requests.append(ambiguity_request)

        return ambiguous_requests

    async def _check_parameter_ambiguity(self, param: str, value: Any, logical_query: LogicalQuery) -> Optional[ClarificationRequest]:
        """Check if a parameter value is ambiguous and needs clarification."""
        value_str = str(value).lower()

        ambiguous_patterns = {
            "time_reference": [
                ("recent", ["last 7 days", "last 30 days", "last quarter"]),
                ("soon", ["next week", "next month", "next quarter"]),
                ("later", ["Q3", "second half of the year", "after launch"]),
            ],
            "entity": [
                ("team", ["engineering team", "support team", "marketing team"]),
                ("service", ["API service", "customer support service", "billing service"]),
                ("platform", ["web platform", "mobile platform", "data platform"]),
            ],
            "location": [
                ("office", ["headquarters", "regional office", "remote"]),
                ("region", ["North America", "EMEA", "APAC"]),
            ],
            "metric": [
                ("performance", ["latency", "throughput", "resource utilization"]),
                ("growth", ["user growth", "revenue growth", "engagement growth"]),
            ],
        }

        if param in ambiguous_patterns:
            for pattern, options in ambiguous_patterns[param]:
                if pattern in value_str:
                    return ClarificationRequest(
                        type=ClarificationType.AMBIGUOUS_REFERENCE,
                        field_name=param,
                        question=f"'{value}' could refer to multiple things. Which do you mean?",
                        options=options,
                        priority=1,
                    )

        # Use LLM to detect more subtle ambiguities
        return await self._llm_ambiguity_check(param, value, logical_query)

    async def _llm_ambiguity_check(self, param: str, value: Any, logical_query: LogicalQuery) -> Optional[ClarificationRequest]:
        """Use LLM to detect subtle ambiguities in parameter values."""
        ambiguity_prompt = f"""
        Check if this parameter value is ambiguous or unclear:

        Parameter: {param}
        Value: {value}
        Query context: {logical_query.original_query}
        Domain: {logical_query.domain}

        Could this value refer to multiple different things? Is it specific enough for accurate information retrieval?

        If ambiguous, respond with "AMBIGUOUS" and suggest 2-3 possible interpretations.
        If clear, respond with "CLEAR".

        Format for ambiguous cases:
        AMBIGUOUS
        1. [interpretation 1]
        2. [interpretation 2]
        3. [interpretation 3]
        """

        messages = [
            ChatMessage(role=MessageRole.SYSTEM,
                       content="You are an ambiguity detection expert. Identify when parameter values are unclear or could have multiple meanings."),
            ChatMessage(role=MessageRole.USER, content=ambiguity_prompt)
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            response_text = response.content.strip()

            if response_text.startswith("AMBIGUOUS"):
                lines = response_text.split('\n')[1:]  # Skip "AMBIGUOUS" line
                options = []
                for line in lines:
                    if line.strip() and (line.strip().startswith(tuple('123456789'))):
                        option = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                        options.append(option)

                if options:
                    return ClarificationRequest(
                        type=ClarificationType.AMBIGUOUS_REFERENCE,
                        field_name=param,
                        question=f"The value '{value}' for {param} could mean different things. Which do you mean?",
                        options=options,
                        priority=1
                    )
        except:
            pass

        return None

    async def process_clarification_response(self, request: ClarificationRequest, user_response: str) -> ClarificationResponse:
        """Process user's response to a clarification request."""
        # Extract structured value from user response
        extracted_value = await self._extract_value_from_response(request, user_response)

        # Calculate confidence in the extraction
        confidence = await self._calculate_extraction_confidence(request, user_response, extracted_value)

        # Extract additional context if available
        additional_context = await self._extract_additional_context(request, user_response)

        return ClarificationResponse(
            original_request=request,
            user_response=user_response,
            extracted_value=extracted_value,
            confidence=confidence,
            additional_context=additional_context
        )

    async def _extract_value_from_response(self, request: ClarificationRequest, user_response: str) -> Any:
        """Extract structured value from user's clarification response."""
        response_lower = user_response.lower().strip()

        # Handle option-based responses
        if request.options:
            # Check for numeric selection (1, 2, 3, etc.)
            for i, option in enumerate(request.options):
                if str(i + 1) in response_lower or option.lower() in response_lower:
                    return option

        # Handle direct value responses
        if request.field_name == "year":
            # Extract year from response
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', user_response)
            if year_match:
                return int(year_match.group())

        elif request.field_name == "quarter":
            # Extract quarter
            quarter_patterns = {
                'q1': 'Q1', 'first quarter': 'Q1', '1st quarter': 'Q1',
                'q2': 'Q2', 'second quarter': 'Q2', '2nd quarter': 'Q2',
                'q3': 'Q3', 'third quarter': 'Q3', '3rd quarter': 'Q3',
                'q4': 'Q4', 'fourth quarter': 'Q4', '4th quarter': 'Q4'
            }
            for pattern, quarter in quarter_patterns.items():
                if pattern in response_lower:
                    return quarter

        # Use LLM for complex extraction
        return await self._llm_value_extraction(request, user_response)

    async def _llm_value_extraction(self, request: ClarificationRequest, user_response: str) -> Any:
        """Use LLM to extract value from complex user responses."""
        extraction_prompt = f"""
        Extract the specific value from this user response:

        Question: {request.question}
        User Response: {user_response}
        Field Type: {request.field_name}

        Extract the most specific, structured value from the user's response.
        For companies: return the full official name
        For dates/periods: return in a standard format
        For metrics: return the specific metric name

        Respond with only the extracted value, nothing else.
        """

        messages = [
            ChatMessage(role=MessageRole.SYSTEM,
                       content="You are a value extraction expert. Extract specific, structured values from user responses."),
            ChatMessage(role=MessageRole.USER, content=extraction_prompt)
        ]

        try:
            response = await self.llm_client.acomplete(messages)
            return response.content.strip()
        except:
            return user_response  # Fallback to original response

    async def _calculate_extraction_confidence(self, request: ClarificationRequest, user_response: str, extracted_value: Any) -> float:
        """Calculate confidence in the value extraction."""
        confidence = 0.0

        # High confidence for option-based responses
        if request.options and str(extracted_value) in request.options:
            confidence = 0.95

        # Medium-high confidence for pattern matches
        elif request.field_name == "year" and isinstance(extracted_value, int):
            confidence = 0.9

        # Medium confidence for direct mentions
        elif str(extracted_value).lower() in user_response.lower():
            confidence = 0.8

        # Lower confidence for LLM extraction
        else:
            confidence = 0.6

        # Boost confidence for longer, more detailed responses
        if len(user_response.split()) > 3:
            confidence = min(1.0, confidence + 0.1)

        return confidence

    async def _extract_additional_context(self, request: ClarificationRequest, user_response: str) -> Dict[str, Any]:
        """Extract additional contextual information from user response."""
        context = {}

        # Look for additional temporal information
        if "annual" in user_response.lower() or "yearly" in user_response.lower():
            context["period_type"] = "annual"
        elif "quarterly" in user_response.lower():
            context["period_type"] = "quarterly"
        elif "monthly" in user_response.lower():
            context["period_type"] = "monthly"

        # Look for currency mentions
        currency_patterns = {
            "dollar": "USD", "usd": "USD", "$": "USD",
            "euro": "EUR", "eur": "EUR", "€": "EUR",
            "pound": "GBP", "gbp": "GBP", "£": "GBP"
        }
        for pattern, currency in currency_patterns.items():
            if pattern in user_response.lower():
                context["currency"] = currency
                break

        # Look for urgency or importance indicators
        urgency_words = ["urgent", "asap", "immediately", "quickly", "priority"]
        if any(word in user_response.lower() for word in urgency_words):
            context["urgency"] = "high"

        return context

    def generate_progressive_clarifications(self, logical_query: LogicalQuery, clarification_requests: List[ClarificationRequest]) -> List[ClarificationRequest]:
        """Generate progressive clarifications to minimize back-and-forth."""
        if not clarification_requests:
            return []

        # Group related clarifications
        grouped_requests = self._group_related_clarifications(clarification_requests)

        # Create combined clarifications for related parameters
        progressive_requests = []
        for group in grouped_requests:
            if len(group) == 1:
                progressive_requests.append(group[0])
            else:
                combined_request = self._combine_clarification_requests(group)
                progressive_requests.append(combined_request)

        return progressive_requests

    def _group_related_clarifications(self, requests: List[ClarificationRequest]) -> List[List[ClarificationRequest]]:
        """Group related clarification requests together."""
        groups = []
        ungrouped = requests.copy()

        # Define related parameter groups
        related_groups = [
            ["company", "segment", "division"],
            ["year", "quarter", "time_period"],
            ["metric_type", "currency", "unit"],
            ["entity", "entity_type"]
        ]

        for related_params in related_groups:
            group = []
            remaining = []

            for request in ungrouped:
                if request.field_name in related_params:
                    group.append(request)
                else:
                    remaining.append(request)

            if group:
                groups.append(group)
            ungrouped = remaining

        # Add remaining ungrouped requests as individual groups
        for request in ungrouped:
            groups.append([request])

        return groups

    def _combine_clarification_requests(self, requests: List[ClarificationRequest]) -> ClarificationRequest:
        """Combine multiple related clarification requests into one."""
        if len(requests) == 1:
            return requests[0]

        # Create combined question
        field_names = [r.field_name for r in requests]
        combined_question = f"I need a few more details to find the right information:\n"

        for i, request in enumerate(requests, 1):
            combined_question += f"{i}. {request.question}\n"

        # Combine options if any
        combined_options = []
        for request in requests:
            if request.options:
                combined_options.extend([f"{request.field_name}: {opt}" for opt in request.options])

        # Use highest priority
        min_priority = min(r.priority for r in requests)

        return ClarificationRequest(
            type=ClarificationType.MULTIPLE_INTERPRETATIONS,
            field_name=", ".join(field_names),
            question=combined_question.strip(),
            options=combined_options,
            priority=min_priority,
            context="Combined clarification for related parameters"
        )
