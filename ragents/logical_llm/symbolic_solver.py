"""Symbolic solver for logical constraints and reasoning."""

import re
import ast
import operator
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date

from .models import LogicalConstraint, LogicalOperator


class SolverStatus(Enum):
    """Status of symbolic solver execution."""
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class SymbolicExpression:
    """Represents a symbolic expression for constraint solving."""

    expression: str
    variables: Set[str] = field(default_factory=set)
    constraints: List[LogicalConstraint] = field(default_factory=list)
    domain: Optional[str] = None

    def __post_init__(self):
        """Extract variables from expression after initialization."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> Set[str]:
        """Extract variable names from the expression."""
        # Simple variable extraction using regex
        variable_pattern = r'\b[a-zA-Z_]\w*\b'
        variables = set(re.findall(variable_pattern, self.expression))

        # Remove operators and keywords
        reserved_words = {
            'and', 'or', 'not', 'in', 'exists', 'forall',
            'true', 'false', 'gt', 'lt', 'ge', 'le', 'eq'
        }
        variables -= reserved_words

        return variables


@dataclass
class SolverResult:
    """Result from symbolic solver execution."""

    status: SolverStatus
    satisfied_constraints: List[LogicalConstraint] = field(default_factory=list)
    violated_constraints: List[LogicalConstraint] = field(default_factory=list)
    variable_assignments: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


class SymbolicSolver:
    """Symbolic solver for logical constraints and reasoning."""

    def __init__(self):
        self.operators = {
            LogicalOperator.EQUALS: operator.eq,
            LogicalOperator.GREATER_THAN: operator.gt,
            LogicalOperator.LESS_THAN: operator.lt,
            LogicalOperator.GREATER_EQUAL: operator.ge,
            LogicalOperator.LESS_EQUAL: operator.le,
            LogicalOperator.AND: operator.and_,
            LogicalOperator.OR: operator.or_,
            LogicalOperator.NOT: operator.not_,
        }

        self.type_validators = self._initialize_type_validators()
        self.constraint_rules = self._initialize_constraint_rules()

    def _initialize_type_validators(self) -> Dict[str, callable]:
        """Initialize type validation functions."""
        return {
            'year': lambda x: isinstance(x, int) and 1800 <= x <= 2100,
            'quarter': lambda x: isinstance(x, str) and bool(re.fullmatch(r'Q[1-4]', x.upper())),
            'month': lambda x: isinstance(x, int) and 1 <= x <= 12,
            'percentage': lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
            'positive_number': lambda x: isinstance(x, (int, float)) and x > 0,
            'non_negative': lambda x: isinstance(x, (int, float)) and x >= 0,
            'date_string': self._validate_date_string,
            'duration': lambda x: isinstance(x, (int, float, str)) and bool(str(x).strip()),
            'entity': lambda x: isinstance(x, str) and len(x.strip()) > 0,
            'location': lambda x: isinstance(x, str) and len(x.strip()) > 1,
            'comparison_list': lambda x: isinstance(x, (list, tuple, set)) and len(x) >= 2,
        }

    def _initialize_constraint_rules(self) -> Dict[str, List[callable]]:
        """Initialize domain-specific constraint rules."""
        return {
            'temporal': [
                self._validate_time_period_consistency,
                self._validate_temporal_consistency,
                self._validate_date_range,
            ],
            'quantitative': [
                self._validate_numeric_consistency,
                self._validate_unit_consistency,
            ],
            'comparative': [
                self._validate_comparative_scope,
            ],
            'causal': [
                self._validate_causal_relationship,
            ],
            'procedural': [
                self._validate_procedural_structure,
            ],
            'spatial': [
                self._validate_spatial_scope,
            ],
            'trend': [
                self._validate_trend_requirements,
            ],
        }

    def solve_constraints(self, constraints: List[LogicalConstraint],
                         variable_assignments: Dict[str, Any],
                         domain: Optional[str] = None) -> SolverResult:
        """Solve a set of logical constraints given variable assignments."""
        try:
            satisfied = []
            violated = []
            suggestions = []

            # Validate each constraint
            for constraint in constraints:
                is_satisfied, error_msg = self._evaluate_constraint(constraint, variable_assignments)

                if is_satisfied:
                    satisfied.append(constraint)
                else:
                    violated.append(constraint)
                    if error_msg:
                        suggestions.append(f"Fix {constraint.field_name}: {error_msg}")

            # Apply domain-specific rules
            if domain and domain in self.constraint_rules:
                domain_violations = self._apply_domain_rules(domain, variable_assignments)
                suggestions.extend(domain_violations)

            # Determine overall status
            if not violated:
                status = SolverStatus.SATISFIED
            elif violated:
                status = SolverStatus.UNSATISFIED
            else:
                status = SolverStatus.UNKNOWN

            # Calculate confidence score
            confidence = self._calculate_confidence(satisfied, violated, variable_assignments)

            return SolverResult(
                status=status,
                satisfied_constraints=satisfied,
                violated_constraints=violated,
                variable_assignments=variable_assignments,
                suggestions=suggestions,
                confidence_score=confidence
            )

        except Exception as e:
            return SolverResult(
                status=SolverStatus.ERROR,
                error_message=str(e),
                variable_assignments=variable_assignments
            )

    def _evaluate_constraint(self, constraint: LogicalConstraint,
                           assignments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Evaluate a single constraint against variable assignments."""
        field_name = constraint.field_name
        operator_type = constraint.operator
        expected_value = constraint.value

        # Check if required field exists
        if constraint.required and field_name not in assignments:
            return False, f"Required field '{field_name}' is missing"

        if field_name not in assignments:
            return True, None  # Optional field not provided

        actual_value = assignments[field_name]

        try:
            # Handle different operators
            if operator_type == LogicalOperator.EXISTS:
                return actual_value is not None, None

            elif operator_type == LogicalOperator.EQUALS:
                return actual_value == expected_value, None

            elif operator_type == LogicalOperator.IN:
                if isinstance(expected_value, (list, tuple, set)):
                    return actual_value in expected_value, None
                else:
                    return False, f"Expected value should be a collection for IN operator"

            elif operator_type == LogicalOperator.GREATER_THAN:
                if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                    return actual_value > expected_value, None
                else:
                    return False, f"Numeric values required for comparison"

            elif operator_type == LogicalOperator.LESS_THAN:
                if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                    return actual_value < expected_value, None
                else:
                    return False, f"Numeric values required for comparison"

            elif operator_type == LogicalOperator.GREATER_EQUAL:
                if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                    return actual_value >= expected_value, None
                else:
                    return False, f"Numeric values required for comparison"

            elif operator_type == LogicalOperator.LESS_EQUAL:
                if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                    return actual_value <= expected_value, None
                else:
                    return False, f"Numeric values required for comparison"

            else:
                return False, f"Unsupported operator: {operator_type}"

        except Exception as e:
            return False, f"Evaluation error: {str(e)}"

    def _apply_domain_rules(self, domain: str, assignments: Dict[str, Any]) -> List[str]:
        """Apply domain-specific validation rules."""
        violations = []
        rules = self.constraint_rules.get(domain, [])

        for rule in rules:
            try:
                violation = rule(assignments)
                if violation:
                    violations.append(violation)
            except Exception as e:
                violations.append(f"Rule validation error: {str(e)}")

        return violations

    def _validate_time_period_consistency(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate time period consistency."""
        year = assignments.get('year')
        quarter = assignments.get('quarter')
        time_period = assignments.get('time_period', '').lower()

        # Check year validity
        if year and not isinstance(year, int):
            try:
                year = int(year)
            except:
                return f"Year '{year}' is not a valid number"

        if year and (year < 1900 or year > 2030):
            return f"Year {year} seems outside reasonable range (1900-2030)"

        # Check quarter format
        if quarter and quarter not in ['Q1', 'Q2', 'Q3', 'Q4']:
            return f"Quarter '{quarter}' should be Q1, Q2, Q3, or Q4"

        # Check time period consistency
        if time_period and quarter:
            if quarter.lower() not in time_period:
                return f"Quarter '{quarter}' doesn't match time period '{time_period}'"

        return None

    def _validate_comparative_scope(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Ensure comparative queries reference multiple entities."""
        entities = assignments.get('entities')

        normalized: List[str] = []
        if isinstance(entities, (list, tuple, set)):
            normalized = [str(e).strip() for e in entities if str(e).strip()]
        elif isinstance(entities, str):
            parts = re.split(r',|;|/| vs | versus ', entities, flags=re.IGNORECASE)
            normalized = [part.strip() for part in parts if part.strip()]

        if normalized and len({item.lower() for item in normalized}) < 2:
            return "Comparative queries should include at least two distinct entities"

        return None

    def _validate_causal_relationship(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Check cause/effect consistency for causal queries."""
        effect = assignments.get('effect')
        cause = assignments.get('cause')

        if not effect:
            return "Causal queries benefit from an explicit effect or outcome"

        if cause and isinstance(cause, str) and isinstance(effect, str):
            if cause.strip().lower() == effect.strip().lower():
                return "Cause and effect should not be identical"

        return None

    def _validate_procedural_structure(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Ensure procedural queries include steps or a clear subject."""
        steps = assignments.get('steps')
        subject = assignments.get('subject')

        if steps and isinstance(steps, (list, tuple)):
            if not all(isinstance(step, str) and step.strip() for step in steps):
                return "Procedural steps should be described with non-empty text"

        if steps and not subject:
            return "Provide the subject or goal for the procedure to give steps context"

        return None

    def _validate_spatial_scope(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate spatial focus queries specify a location."""
        location = assignments.get('location')
        if not location:
            return "Spatial queries should specify a location or region"

        if isinstance(location, str) and len(location.strip()) < 2:
            return "Location descriptions should be more specific"

        return None

    def _validate_trend_requirements(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Ensure trend analysis contains time reference and subject."""
        time_reference = assignments.get('time_reference')
        metric = assignments.get('metric')

        if not time_reference:
            return "Trend analysis needs a time reference (e.g., range or period)"

        if metric and isinstance(metric, str) and not metric.strip():
            return "Metric description should not be empty"

        return None

    def _validate_temporal_consistency(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate temporal consistency."""
        time_reference = assignments.get('time_reference')
        duration = assignments.get('duration')

        if time_reference and 'ago' in str(time_reference).lower():
            # Relative time reference
            if not duration:
                return "Relative time references need duration specification"

        return None

    def _validate_date_range(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate date range consistency."""
        start_date = assignments.get('start_date')
        end_date = assignments.get('end_date')

        if start_date and end_date:
            try:
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                if start_date > end_date:
                    return "Start date cannot be after end date"
            except:
                return "Invalid date format"

        return None

    def _validate_numeric_consistency(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate numeric consistency."""
        metric = assignments.get('metric')
        unit = assignments.get('unit')

        if metric and isinstance(metric, str):
            if any(word in metric.lower() for word in ['count', 'number', 'quantity']):
                if unit and 'percentage' in unit.lower():
                    return "Count metrics shouldn't use percentage units"

        return None

    def _validate_unit_consistency(self, assignments: Dict[str, Any]) -> Optional[str]:
        """Validate unit consistency."""
        metric = assignments.get('metric', '').lower()
        unit = assignments.get('unit', '').lower()

        # Check unit-metric compatibility
        incompatible_pairs = [
            (['distance', 'length'], ['weight', 'mass']),
            (['time', 'duration'], ['currency', 'money']),
            (['temperature'], ['distance', 'weight'])
        ]

        for metric_types, unit_types in incompatible_pairs:
            if any(mt in metric for mt in metric_types) and any(ut in unit for ut in unit_types):
                return f"Unit type '{unit}' incompatible with metric type '{metric}'"

        return None

    def _validate_date_string(self, value: str) -> bool:
        """Validate date string format."""
        if not isinstance(value, str):
            return False

        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',  # M/D/YYYY
            r'Q[1-4] \d{4}',  # Q1 2023
        ]

        return any(re.match(pattern, value) for pattern in date_patterns)

    def _calculate_confidence(self, satisfied: List[LogicalConstraint],
                            violated: List[LogicalConstraint],
                            assignments: Dict[str, Any]) -> float:
        """Calculate confidence score for constraint satisfaction."""
        total_constraints = len(satisfied) + len(violated)
        if total_constraints == 0:
            return 1.0

        base_confidence = len(satisfied) / total_constraints

        # Boost confidence for having many assignments
        assignment_boost = min(0.2, len(assignments) * 0.02)

        # Reduce confidence for critical violations
        critical_penalty = 0.0
        for constraint in violated:
            if constraint.required:
                critical_penalty += 0.1

        final_confidence = max(0.0, min(1.0, base_confidence + assignment_boost - critical_penalty))
        return final_confidence

    def generate_symbolic_expression(self, constraints: List[LogicalConstraint]) -> SymbolicExpression:
        """Generate symbolic expression from logical constraints."""
        if not constraints:
            return SymbolicExpression("true", set(), [])

        # Convert constraints to symbolic form
        expressions = []
        all_variables = set()

        for constraint in constraints:
            symbolic = constraint.to_symbolic()
            expressions.append(symbolic)
            all_variables.add(constraint.field_name)

        # Combine with AND operators
        combined_expression = " ∧ ".join(expressions)

        return SymbolicExpression(
            expression=combined_expression,
            variables=all_variables,
            constraints=constraints
        )

    def optimize_query_constraints(self, constraints: List[LogicalConstraint],
                                 assignments: Dict[str, Any]) -> List[LogicalConstraint]:
        """Optimize constraints based on current assignments to reduce search space."""
        optimized = []

        for constraint in constraints:
            # Skip constraints that are already satisfied
            is_satisfied, _ = self._evaluate_constraint(constraint, assignments)
            if not is_satisfied or constraint.required:
                optimized.append(constraint)

        # Add derived constraints based on assignments
        derived_constraints = self._derive_additional_constraints(assignments)
        optimized.extend(derived_constraints)

        return optimized

    def _derive_additional_constraints(self, assignments: Dict[str, Any]) -> List[LogicalConstraint]:
        """Derive additional constraints based on current assignments."""
        derived = []

        # If company is specified, derive industry-specific constraints
        company = assignments.get('company', '').lower()
        if company:
            if any(tech in company for tech in ['apple', 'google', 'microsoft', 'meta']):
                derived.append(LogicalConstraint(
                    'industry', LogicalOperator.EQUALS, 'technology',
                    'Technology company constraint', required=False
                ))

        # If year is recent, derive quarterly reporting constraint
        year = assignments.get('year')
        if year and isinstance(year, int) and year >= 2020:
            derived.append(LogicalConstraint(
                'reporting_frequency', LogicalOperator.IN, ['quarterly', 'annual'],
                'Recent year reporting constraint', required=False
            ))

        return derived

    def suggest_constraint_relaxation(self, violated_constraints: List[LogicalConstraint]) -> List[str]:
        """Suggest ways to relax constraints for better results."""
        suggestions = []

        for constraint in violated_constraints:
            if constraint.operator == LogicalOperator.EQUALS:
                suggestions.append(f"Consider broadening the exact match requirement for {constraint.field_name}")

            elif constraint.operator == LogicalOperator.IN:
                suggestions.append(f"Consider expanding the allowed values for {constraint.field_name}")

            elif constraint.operator in [LogicalOperator.GREATER_THAN, LogicalOperator.LESS_THAN]:
                suggestions.append(f"Consider adjusting the numeric threshold for {constraint.field_name}")

            elif constraint.operator == LogicalOperator.EXISTS:
                suggestions.append(f"Consider making {constraint.field_name} optional if possible")

        return suggestions
