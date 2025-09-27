"""Core data models shared across the logical LLM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LogicalOperator(str, Enum):
    """Logical operators for constraint expressions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    EQUALS = "EQUALS"
    GREATER_THAN = "GT"
    LESS_THAN = "LT"
    GREATER_EQUAL = "GE"
    LESS_EQUAL = "LE"
    IN = "IN"
    EXISTS = "EXISTS"
    FORALL = "FORALL"


@dataclass
class LogicalConstraint:
    """Represents a logical constraint on query parameters."""

    field_name: str
    operator: LogicalOperator
    value: Any
    description: str
    required: bool = True
    dependencies: List[str] = field(default_factory=list)

    def to_symbolic(self) -> str:
        """Convert constraint to symbolic representation."""
        if self.operator == LogicalOperator.EQUALS:
            return f"{self.field_name} = {self.value}"
        if self.operator == LogicalOperator.IN:
            rendered = ", ".join(map(str, self.value)) if isinstance(self.value, (list, tuple, set)) else self.value
            return f"{self.field_name} ∈ {{{rendered}}}"
        if self.operator == LogicalOperator.EXISTS:
            return f"∃ {self.field_name}"
        if self.operator == LogicalOperator.GREATER_THAN:
            return f"{self.field_name} > {self.value}"
        if self.operator == LogicalOperator.LESS_THAN:
            return f"{self.field_name} < {self.value}"
        if self.operator == LogicalOperator.GREATER_EQUAL:
            return f"{self.field_name} ≥ {self.value}"
        if self.operator == LogicalOperator.LESS_EQUAL:
            return f"{self.field_name} ≤ {self.value}"
        return f"{self.field_name} {self.operator.value} {self.value}"


class RetrievalMode(str, Enum):
    """Supported retrieval modes for downstream search."""

    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    GRAPH = "graph"
    GRAPH_HYBRID = "graph_hybrid"
    KEYWORD = "keyword"


@dataclass
class SearchDirective:
    """Instruction for how retrieval should be executed downstream."""

    query_text: str
    mode: RetrievalMode = RetrievalMode.SEMANTIC
    filters: Dict[str, Any] = field(default_factory=dict)
    hybrid_alpha: Optional[float] = None
    graph_query: Optional[str] = None
    graph_entry_points: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Return a serializable representation for logging or telemetry."""
        return {
            "query_text": self.query_text,
            "mode": self.mode.value,
            "filters": self.filters,
            "hybrid_alpha": self.hybrid_alpha,
            "graph_query": self.graph_query,
            "graph_entry_points": self.graph_entry_points,
            "notes": self.notes,
            "metadata": self.metadata,
        }


@dataclass
class LogicalQuery:
    """Structured representation of a logical query with constraints."""

    original_query: str
    domain: str
    intent: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[LogicalConstraint] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    logical_form: Optional[str] = None
    confidence_score: float = 0.0
    refinement_suggestions: List[str] = field(default_factory=list)
    context_summary: Optional[str] = None
    topic_tags: List[str] = field(default_factory=list)
    retrieval_directive: Optional[SearchDirective] = None

    def is_complete(self) -> bool:
        """Check if query has all required parameters."""
        return len(self.missing_parameters) == 0

    def get_satisfied_constraints(self) -> List[LogicalConstraint]:
        """Get constraints that are satisfied by current parameters."""
        return [c for c in self.constraints if c.field_name in self.parameters]

    def get_unsatisfied_constraints(self) -> List[LogicalConstraint]:
        """Get constraints that are not yet satisfied."""
        return [c for c in self.constraints if c.required and c.field_name not in self.parameters]
