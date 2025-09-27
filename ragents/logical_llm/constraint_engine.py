"""Constraint engine for logical reasoning and query optimization."""

import asyncio
import re
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .domain_definitions import DomainDefinition, build_builtin_domains
from .models import LogicalConstraint, LogicalOperator, LogicalQuery
from .symbolic_solver import SymbolicSolver, SolverResult, SolverStatus
from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole


class ConstraintPriority(Enum):
    """Priority levels for constraints."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


class RuleType(Enum):
    """Types of constraint rules."""
    VALIDATION = "validation"
    INFERENCE = "inference"
    OPTIMIZATION = "optimization"
    DEPENDENCY = "dependency"


@dataclass
class ConstraintRule:
    """Represents a constraint rule for logical reasoning."""

    name: str
    rule_type: RuleType
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    priority: ConstraintPriority = ConstraintPriority.MEDIUM
    description: str = ""
    domain: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

    def applies_to(self, assignments: Dict[str, Any], domain: Optional[str] = None) -> bool:
        """Check if rule applies to given assignments and domain."""
        if self.domain and domain != self.domain:
            return False

        try:
            return self.condition(assignments)
        except Exception:
            return False

    def execute(self, assignments: Dict[str, Any]) -> Any:
        """Execute the rule action."""
        try:
            return self.action(assignments)
        except Exception as e:
            return f"Rule execution failed: {str(e)}"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""

    constraint: LogicalConstraint
    message: str
    severity: ConstraintPriority
    suggested_fix: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ConstraintEngine:
    """Advanced constraint engine for logical reasoning and optimization."""

    def __init__(self, llm_client: LLMClient, domain_registry: Optional[Dict[str, DomainDefinition]] = None):
        self.llm_client = llm_client
        self.solver = SymbolicSolver()
        self.rules: List[ConstraintRule] = []
        self.violation_history: List[ConstraintViolation] = []
        self.domain_registry = domain_registry or build_builtin_domains()
        self.default_domain = "general"
        self._initialize_builtin_rules()

    def _initialize_builtin_rules(self):
        """Initialize domain-agnostic constraint rules."""

        # Temporal hygiene rules
        self.add_rule(
            ConstraintRule(
                name="temporal_year_range",
                rule_type=RuleType.VALIDATION,
                condition=lambda a: "year" in a,
                action=self._validate_year_range,
                priority=ConstraintPriority.MEDIUM,
                description="Ensure referenced years fall within a reasonable range",
                domain="temporal",
            )
        )
        self.add_rule(
            ConstraintRule(
                name="temporal_reference_normalization",
                rule_type=RuleType.OPTIMIZATION,
                condition=lambda a: any(key in a for key in ("time_reference", "time_period", "year")),
                action=self._normalize_temporal_reference,
                priority=ConstraintPriority.MEDIUM,
                description="Normalize temporal references for consistent retrieval",
            )
        )

        # Quantitative / metric-focused rules
        self.add_rule(
            ConstraintRule(
                name="quantitative_subject_required",
                rule_type=RuleType.VALIDATION,
                condition=lambda a: "metric" in a and "subject" not in a,
                action=lambda a: "Please specify the subject or entity that the metric describes",
                priority=ConstraintPriority.HIGH,
                description="Quantitative queries need a subject for the metric",
                domain="quantitative",
            )
        )
        self.add_rule(
            ConstraintRule(
                name="metric_normalization",
                rule_type=RuleType.OPTIMIZATION,
                condition=lambda a: "metric" in a,
                action=self._normalize_metric_name,
                priority=ConstraintPriority.MEDIUM,
                description="Normalize metric names to canonical forms",
            )
        )

        # Comparative reasoning rules
        self.add_rule(
            ConstraintRule(
                name="comparison_entity_count",
                rule_type=RuleType.VALIDATION,
                condition=lambda a: "entities" in a,
                action=self._validate_entities_count,
                priority=ConstraintPriority.HIGH,
                description="Comparisons require at least two entities",
                domain="comparative",
            )
        )

        # Causal reasoning rule encouraging cause context
        self.add_rule(
            ConstraintRule(
                name="causal_cause_hint",
                rule_type=RuleType.VALIDATION,
                condition=lambda a: "effect" in a and "cause" not in a,
                action=lambda a: "Mention suspected causes or factors to analyze the effect",
                priority=ConstraintPriority.MEDIUM,
                description="Causal queries benefit from explicit cause candidates",
                domain="causal",
            )
        )

        # Generic normalization + inference helpers
        self.add_rule(
            ConstraintRule(
                name="entity_name_normalization",
                rule_type=RuleType.OPTIMIZATION,
                condition=lambda a: any(key in a for key in ("entity", "subject")),
                action=self._normalize_entity_name,
                priority=ConstraintPriority.MEDIUM,
                description="Normalize entity/subject naming for retrieval",
            )
        )
        self.add_rule(
            ConstraintRule(
                name="time_reference_inference",
                rule_type=RuleType.INFERENCE,
                condition=lambda a: "time_reference" not in a and any(key in a for key in ("year", "time_period")),
                action=self._infer_time_reference,
                priority=ConstraintPriority.LOW,
                description="Infer a friendly time reference for retrieval",
            )
        )

    def add_rule(self, rule: ConstraintRule):
        """Add a constraint rule to the engine."""
        self.rules.append(rule)

    def remove_rule(self, rule_name: str):
        """Remove a constraint rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]

    async def process_logical_query(self, logical_query: LogicalQuery) -> Tuple[LogicalQuery, List[ConstraintViolation]]:
        """Process a logical query through the constraint engine."""
        violations = []
        optimized_assignments = logical_query.parameters.copy()

        # Apply validation rules
        validation_violations = await self._apply_validation_rules(
            optimized_assignments, logical_query.domain
        )
        violations.extend(validation_violations)

        # Apply inference rules to fill in missing information
        inferred_assignments = await self._apply_inference_rules(
            optimized_assignments, logical_query.domain
        )
        optimized_assignments.update(inferred_assignments)

        # Apply optimization rules
        optimization_results = await self._apply_optimization_rules(
            optimized_assignments, logical_query.domain
        )
        optimized_assignments.update(optimization_results)

        # Solve constraints with symbolic solver
        solver_result = self.solver.solve_constraints(
            logical_query.constraints, optimized_assignments, logical_query.domain
        )

        # Convert solver violations to constraint violations
        for violated_constraint in solver_result.violated_constraints:
            violation = ConstraintViolation(
                constraint=violated_constraint,
                message=violated_constraint.description,
                severity=ConstraintPriority.HIGH,
                suggested_fix=self._generate_fix_suggestion(violated_constraint, optimized_assignments)
            )
            violations.append(violation)

        # Create optimized logical query
        optimized_query = LogicalQuery(
            original_query=logical_query.original_query,
            domain=logical_query.domain,
            intent=logical_query.intent,
            parameters=optimized_assignments,
            constraints=logical_query.constraints,
            missing_parameters=logical_query.missing_parameters,
            logical_form=logical_query.logical_form,
            confidence_score=solver_result.confidence_score,
            refinement_suggestions=solver_result.suggestions or logical_query.refinement_suggestions,
            context_summary=logical_query.context_summary,
            topic_tags=logical_query.topic_tags,
        )
        optimized_query.retrieval_directive = logical_query.retrieval_directive

        return optimized_query, violations

    async def _apply_validation_rules(self, assignments: Dict[str, Any], domain: Optional[str]) -> List[ConstraintViolation]:
        """Apply validation rules and return violations."""
        violations = []

        validation_rules = [r for r in self.rules if r.rule_type == RuleType.VALIDATION]

        for rule in validation_rules:
            if rule.applies_to(assignments, domain):
                try:
                    result = rule.execute(assignments)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, str):  # Error message
                        violation = ConstraintViolation(
                            constraint=LogicalConstraint(
                                field_name="validation",
                                operator=LogicalOperator.EXISTS,
                                value=None,
                                description=result
                            ),
                            message=result,
                            severity=rule.priority,
                            context={"rule": rule.name}
                        )
                        violations.append(violation)
                except Exception as e:
                    # Log rule execution error but don't fail
                    pass

        return violations

    async def _apply_inference_rules(self, assignments: Dict[str, Any], domain: Optional[str]) -> Dict[str, Any]:
        """Apply inference rules to infer missing information."""
        inferred = {}

        inference_rules = [r for r in self.rules if r.rule_type == RuleType.INFERENCE]

        for rule in inference_rules:
            if rule.applies_to(assignments, domain):
                try:
                    result = rule.execute(assignments)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, dict):
                        inferred.update(result)
                    elif isinstance(result, tuple) and len(result) == 2:
                        key, value = result
                        inferred[key] = value
                except Exception:
                    continue

        return inferred

    async def _apply_optimization_rules(self, assignments: Dict[str, Any], domain: Optional[str]) -> Dict[str, Any]:
        """Apply optimization rules to improve query parameters."""
        optimized = {}

        optimization_rules = [r for r in self.rules if r.rule_type == RuleType.OPTIMIZATION]

        for rule in optimization_rules:
            if rule.applies_to(assignments, domain):
                try:
                    result = rule.execute(assignments)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, dict):
                        optimized.update(result)
                    elif isinstance(result, tuple) and len(result) == 2:
                        key, value = result
                        optimized[key] = value
                except Exception:
                    continue

        return optimized

    def _validate_year_range(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate that referenced years fall within a reasonable window."""
        raw_year = assignments.get("year")
        if not raw_year:
            return None

        try:
            year_int = int(str(raw_year)[:4])
        except ValueError:
            return "Year should be a numeric value"

        current_year = datetime.now().year
        if year_int < 1800 or year_int > current_year + 5:
            return f"Year {year_int} looks out of range"
        return None

    async def _normalize_entity_name(self, assignments: Dict[str, Any]) -> Tuple[str, str]:
        """Standardize entity or subject names for consistent retrieval."""
        field = "entity" if "entity" in assignments else "subject"
        value = assignments.get(field)
        if not value:
            return field, value

        prompt = f"""
        Normalize the following name so it is appropriate for search indices. Keep it concise,
        remove redundant determiners, and capitalize proper nouns.

        Original: {value}

        Respond with only the normalized name.
        """

        try:
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You clean and normalize entity names for knowledge retrieval.",
                ),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]
            response = await self.llm_client.acomplete(messages)
            normalized = response.content.strip()
            if 0 < len(normalized) <= 120:
                return field, normalized
        except Exception:
            pass
        return field, value

    def _normalize_temporal_reference(self, assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize temporal descriptors to help downstream retrieval."""
        updates: Dict[str, Any] = {}

        time_reference = assignments.get("time_reference") or assignments.get("time_period")
        year = assignments.get("year")

        def normalize(text: str) -> str:
            val = text.strip()
            patterns = [
                (r"\bq([1-4])\s*(\d{2,4})\b", lambda m: f"Q{m.group(1)} 20{m.group(2)}" if len(m.group(2)) == 2 else f"Q{m.group(1)} {m.group(2)}"),
                (r"\b([1-4])q\s*(\d{4})\b", lambda m: f"Q{m.group(1)} {m.group(2)}"),
                (r"\b(first|1st)\s+quarter\s+(\d{4})\b", lambda m: f"Q1 {m.group(2)}"),
                (r"\b(second|2nd)\s+quarter\s+(\d{4})\b", lambda m: f"Q2 {m.group(2)}"),
                (r"\b(third|3rd)\s+quarter\s+(\d{4})\b", lambda m: f"Q3 {m.group(2)}"),
                (r"\b(fourth|4th)\s+quarter\s+(\d{4})\b", lambda m: f"Q4 {m.group(2)}"),
            ]
            for pattern, repl in patterns:
                val = re.sub(pattern, repl, val, flags=re.IGNORECASE)
            return val

        if isinstance(time_reference, str) and time_reference.strip():
            updates["time_reference"] = normalize(time_reference)
        elif year:
            updates["time_reference"] = f"Year {year}"

        if year:
            updates["year"] = str(year)

        return updates

    def _normalize_metric_name(self, assignments: Dict[str, Any]) -> Tuple[str, str]:
        """Normalize metric names to a canonical lowercase form with spaces."""
        metric = assignments.get("metric")
        if not metric:
            return "metric", metric

        cleaned = " ".join(str(metric).split()).lower()
        replacements = {
            "conversion rate": "conversion rate",
            "latency": "latency",
            "response time": "response time",
            "uptime": "uptime",
            "availability": "availability",
        }
        normalized = replacements.get(cleaned, cleaned)
        return "metric", normalized

    def _validate_entities_count(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Ensure comparative queries have at least two distinct entities."""
        entities = assignments.get("entities")
        if not entities:
            return "Please provide the entities to compare"

        if isinstance(entities, str):
            split_entities = [
                e.strip()
                for e in re.split(r",|;|/| vs | versus ", entities, flags=re.IGNORECASE)
                if e.strip()
            ]
        elif isinstance(entities, (list, tuple, set)):
            split_entities = [str(e).strip() for e in entities if str(e).strip()]
        else:
            split_entities = [str(entities)]

        unique_entities = {e.lower() for e in split_entities}
        if len(unique_entities) < 2:
            return "Comparisons need at least two distinct entities"
        return None

    def _infer_time_reference(self, assignments: Dict[str, Any]) -> Tuple[str, str]:
        """Infer a human-friendly time reference from available fields."""
        if "year" in assignments and "quarter" in assignments:
            return "time_reference", f"{assignments['quarter']} {assignments['year']}"
        if "year" in assignments:
            return "time_reference", f"Year {assignments['year']}"
        if "time_period" in assignments:
            return "time_reference", str(assignments["time_period"]).strip()
        return "time_reference", "recent"

    def _generate_fix_suggestion(self, constraint: LogicalConstraint, assignments: Dict[str, Any]) -> str:
        """Generate a suggestion for fixing a constraint violation."""
        field_name = constraint.field_name
        operator = constraint.operator

        if operator == LogicalOperator.EXISTS:
            return f"Please provide a value for {field_name}"

        elif operator == LogicalOperator.IN:
            valid_values = constraint.value
            if isinstance(valid_values, (list, tuple)):
                return f"Please choose one of: {', '.join(map(str, valid_values))}"

        elif operator in [LogicalOperator.GREATER_THAN, LogicalOperator.LESS_THAN]:
            return f"Please provide a value that satisfies {field_name} {operator.value} {constraint.value}"

        else:
            return f"Please check the value for {field_name}"

    async def optimize_for_token_reduction(self, logical_query: LogicalQuery) -> LogicalQuery:
        """Produce a leaner query representation prioritizing high-signal parameters."""
        definition = self.domain_registry.get(
            logical_query.domain, self.domain_registry[self.default_domain]
        )

        priority_fields = definition.required_fields + definition.optional_fields
        specific_params = {
            field: logical_query.parameters[field]
            for field in priority_fields
            if field in logical_query.parameters
        }

        focused_query = await self._generate_focused_query(
            specific_params, logical_query, definition
        )
        optimized_constraints = self._create_focused_constraints(
            specific_params, definition
        )

        optimized = LogicalQuery(
            original_query=logical_query.original_query,
            domain=logical_query.domain,
            intent=logical_query.intent,
            parameters=specific_params,
            constraints=optimized_constraints,
            missing_parameters=[],
            logical_form=focused_query,
            confidence_score=logical_query.confidence_score,
            refinement_suggestions=logical_query.refinement_suggestions,
            context_summary=logical_query.context_summary,
            topic_tags=logical_query.topic_tags,
        )
        optimized.retrieval_directive = logical_query.retrieval_directive
        return optimized

    async def _generate_focused_query(
        self,
        parameters: Dict[str, Any],
        logical_query: LogicalQuery,
        definition: DomainDefinition,
    ) -> str:
        """Generate a focused query string from high-signal parameters."""
        if not parameters:
            return logical_query.original_query

        query_parts = [str(value) for value in parameters.values() if value]
        if logical_query.context_summary:
            query_parts.append(logical_query.context_summary)
        return " ".join(dict.fromkeys(query_parts))

    def _create_focused_constraints(
        self, parameters: Dict[str, Any], definition: DomainDefinition
    ) -> List[LogicalConstraint]:
        """Build equality constraints for retained parameters."""
        constraints = []
        for field, value in parameters.items():
            constraints.append(
                LogicalConstraint(
                    field_name=field,
                    operator=LogicalOperator.EQUALS,
                    value=value,
                    description=f"{field} fixed to {value}",
                    required=True,
                )
            )
        return constraints

    def analyze_constraint_satisfaction(self, logical_query: LogicalQuery) -> Dict[str, Any]:
        """Analyze the constraint satisfaction status of a logical query."""
        solver_result = self.solver.solve_constraints(
            logical_query.constraints,
            logical_query.parameters,
            logical_query.domain
        )

        return {
            "overall_status": solver_result.status.value,
            "satisfaction_rate": len(solver_result.satisfied_constraints) / len(logical_query.constraints)
                              if logical_query.constraints else 1.0,
            "satisfied_count": len(solver_result.satisfied_constraints),
            "violated_count": len(solver_result.violated_constraints),
            "confidence_score": solver_result.confidence_score,
            "suggestions": solver_result.suggestions,
            "optimization_potential": self._assess_optimization_potential(logical_query, solver_result)
        }

    def _assess_optimization_potential(self, logical_query: LogicalQuery, solver_result: SolverResult) -> str:
        """Assess the potential for query optimization."""
        if solver_result.status == SolverStatus.SATISFIED and solver_result.confidence_score > 0.8:
            return "high"
        elif len(logical_query.missing_parameters) <= 1:
            return "medium"
        else:
            return "low"
