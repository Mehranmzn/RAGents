"""Domain configuration for logical reasoning module.

This module centralizes domain definitions so LogicalReasoner and related
components can operate without hard-coded, finance-specific assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import LogicalConstraint, LogicalOperator, RetrievalMode


@dataclass
class DomainRetrievalConfig:
    """Default retrieval preferences for a domain."""

    default_mode: RetrievalMode = RetrievalMode.SEMANTIC
    hybrid_alpha: float = 0.6
    enable_graph_enrichment: bool = False
    graph_focus_entities: List[str] = field(default_factory=list)
    preferred_vector_store: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ClarificationTemplate:
    """Template for generating clarifying questions for a field."""

    questions: List[str]
    options: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    priority: int = 2
    context_hints: List[str] = field(default_factory=list)


@dataclass
class DomainDefinition:
    """Describes a logical domain and its reasoning metadata."""

    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    constraints: List[LogicalConstraint] = field(default_factory=list)
    clarification_templates: Dict[str, ClarificationTemplate] = field(default_factory=dict)
    retrieval: DomainRetrievalConfig = field(default_factory=DomainRetrievalConfig)
    related_patterns: List[str] = field(default_factory=list)
    feature_flags: Dict[str, Any] = field(default_factory=dict)


def build_builtin_domains() -> Dict[str, DomainDefinition]:
    """Create a registry of built-in domains."""

    def constraint(field: str, operator: LogicalOperator, value: Any, description: str, required: bool = True) -> LogicalConstraint:
        return LogicalConstraint(
            field_name=field,
            operator=operator,
            value=value,
            description=description,
            required=required,
        )

    generic_time_templates = ClarificationTemplate(
        questions=[
            "What time period should we look at?",
            "Do you have a specific date or range in mind?",
        ],
        examples=["past week", "Q1 2024", "January 2023", "the last 6 months"],
        priority=1,
        context_hints=["when", "time", "recent", "historical"],
    )

    generic_entity_templates = ClarificationTemplate(
        questions=[
            "Which entity or subject should I focus on?",
            "Can you specify the main person, product, or organization?",
        ],
        examples=["customer onboarding flow", "satellite imagery dataset", "Team Phoenix"],
        priority=1,
        context_hints=["who", "what", "which"],
    )

    domains: Dict[str, DomainDefinition] = {
        "general": DomainDefinition(
            name="general",
            description="Default domain when no strong signal is detected.",
            keywords=[],
            optional_fields=["topic", "entity", "context"],
            clarification_templates={
                "entity": generic_entity_templates,
                "time_period": generic_time_templates,
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.HYBRID,
                hybrid_alpha=0.5,
                notes=["Default hybrid search balances recall and precision for broad queries."],
            ),
        ),
        "temporal": DomainDefinition(
            name="temporal",
            description="Questions about timelines, scheduling, or historical context.",
            keywords=["when", "schedule", "timeline", "deadline", "history", "date", "time"],
            required_fields=["time_reference"],
            optional_fields=["duration", "timezone"],
            constraints=[
                constraint("time_reference", LogicalOperator.EXISTS, None, "Time reference required"),
                constraint("year", LogicalOperator.GREATER_THAN, 1800, "Year must be realistic", required=False),
                constraint("year", LogicalOperator.LESS_EQUAL, 2100, "Year should not be too far in the future", required=False),
            ],
            clarification_templates={
                "time_reference": generic_time_templates,
                "timezone": ClarificationTemplate(
                    questions=["Which timezone should we assume?"],
                    options=["UTC", "PST", "EST", "CET", "IST"],
                ),
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.HYBRID,
                hybrid_alpha=0.7,
                notes=["Temporal queries benefit from keyword-aware ranking to capture explicit dates."],
            ),
        ),
        "quantitative": DomainDefinition(
            name="quantitative",
            description="Questions revolving around measurements, metrics, or statistics.",
            keywords=["measure", "metric", "statistics", "how many", "how much", "percentage", "growth", "trend"],
            required_fields=["metric", "subject"],
            optional_fields=["unit", "time_period", "location"],
            constraints=[
                constraint("metric", LogicalOperator.EXISTS, None, "Metric must be specified"),
                constraint("subject", LogicalOperator.EXISTS, None, "Subject of measurement required"),
            ],
            clarification_templates={
                "metric": ClarificationTemplate(
                    questions=["Which metric should we focus on?"],
                    examples=["conversion rate", "latency", "retention", "temperature"],
                    priority=1,
                ),
                "subject": generic_entity_templates,
                "time_period": generic_time_templates,
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.HYBRID,
                hybrid_alpha=0.65,
                notes=["Combine dense embedding similarity with keyword filters on metric names."],
            ),
        ),
        "comparative": DomainDefinition(
            name="comparative",
            description="Queries comparing entities, options, or approaches.",
            keywords=["compare", "vs", "versus", "difference", "better", "worse", "trade-off"],
            required_fields=["entities"],
            optional_fields=["comparison_dimension", "criteria", "time_period"],
            constraints=[
                constraint("entities", LogicalOperator.GREATER_EQUAL, 2, "Need at least two entities to compare"),
            ],
            clarification_templates={
                "entities": ClarificationTemplate(
                    questions=["Which entities should we compare?"],
                    examples=["Option A", "Option B", "Vendor 1", "Vendor 2"],
                    priority=1,
                ),
                "comparison_dimension": ClarificationTemplate(
                    questions=["What dimension or criteria is most important for the comparison?"],
                    examples=["cost", "performance", "security", "time to market"],
                ),
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.GRAPH_HYBRID,
                enable_graph_enrichment=True,
                graph_focus_entities=["entities", "comparison_dimension"],
                notes=["Graph adjacency can reveal relationships while hybrid search surfaces detailed evidence."],
            ),
        ),
        "causal": DomainDefinition(
            name="causal",
            description="Why/how questions focused on causality or dependencies.",
            keywords=["why", "cause", "effect", "lead to", "impact", "because", "due to"],
            required_fields=["effect"],
            optional_fields=["cause", "context", "conditions"],
            constraints=[
                constraint("effect", LogicalOperator.EXISTS, None, "Effect or outcome must be specified"),
            ],
            clarification_templates={
                "effect": ClarificationTemplate(
                    questions=["What outcome are you investigating?"],
                    examples=["drop in engagement", "system outage", "increase in churn"],
                    priority=1,
                ),
                "cause": ClarificationTemplate(
                    questions=["Any suspected causes or factors to consider?"],
                    examples=["code deployment", "marketing campaign", "user behavior change"],
                ),
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.GRAPH,
                enable_graph_enrichment=True,
                notes=["Graph traversal surfaces dependency paths; complement with supporting documents via semantic search."],
            ),
        ),
        "procedural": DomainDefinition(
            name="procedural",
            description="How-to instructions, workflows, and step-by-step processes.",
            keywords=["how to", "steps", "process", "workflow", "procedure", "guide"],
            optional_fields=["subject", "constraints", "tools"],
            clarification_templates={
                "subject": ClarificationTemplate(
                    questions=["What process or subject should the procedure cover?"],
                    examples=["database migration", "incident response", "onboarding"],
                    priority=1,
                ),
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.SEMANTIC,
                notes=["Dense embeddings capture procedural similarity even with paraphrasing."],
            ),
        ),
        "spatial": DomainDefinition(
            name="spatial",
            description="Location or geography-focused questions.",
            keywords=["where", "location", "region", "map", "distribution"],
            required_fields=["location"],
            optional_fields=["scale", "time_period"],
            constraints=[
                constraint("location", LogicalOperator.EXISTS, None, "Location must be specified"),
            ],
            clarification_templates={
                "location": ClarificationTemplate(
                    questions=["Which location or region should we focus on?"],
                    examples=["North America", "data center eu-west-1", "headquarters"],
                    priority=1,
                ),
            },
            retrieval=DomainRetrievalConfig(
                default_mode=RetrievalMode.HYBRID,
                hybrid_alpha=0.55,
                notes=["Keyword filters on place names coupled with embeddings capture synonyms and abbreviations."],
            ),
        ),
    }

    return domains
