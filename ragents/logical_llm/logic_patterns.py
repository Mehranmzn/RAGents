"""Logic patterns for intelligent query classification and reasoning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern, Tuple

from .models import LogicalConstraint, LogicalOperator


class PatternType(Enum):
    """Types of logic patterns aligned with domain definitions."""

    GENERAL = "general_query"
    TEMPORAL = "temporal_query"
    QUANTITATIVE = "quantitative_query"
    COMPARATIVE = "comparative_query"
    CAUSAL = "causal_query"
    PROCEDURAL = "procedural_query"
    SPATIAL = "spatial_query"
    TREND = "trend_query"


@dataclass
class LogicPattern:
    """Represents a logic pattern for query classification."""

    name: str
    pattern_type: PatternType
    keywords: List[str]
    regex_patterns: List[Pattern] = field(default_factory=list)
    required_entities: List[str] = field(default_factory=list)
    optional_entities: List[str] = field(default_factory=list)
    constraints: List[LogicalConstraint] = field(default_factory=list)
    confidence_threshold: float = 0.7
    token_reduction_potential: float = 0.3
    refinement_questions: List[str] = field(default_factory=list)

    def matches(self, query: str) -> Tuple[bool, float]:
        """Check if pattern matches the query and return confidence score."""
        query_lower = query.lower()
        score = 0.0
        total_checks = 0.0

        keyword_matches = sum(1 for keyword in self.keywords if keyword in query_lower)
        if self.keywords:
            score += (keyword_matches / len(self.keywords)) * 0.4
            total_checks += 0.4

        regex_matches = sum(1 for pattern in self.regex_patterns if pattern.search(query))
        if self.regex_patterns:
            score += (regex_matches / len(self.regex_patterns)) * 0.3
            total_checks += 0.3

        entity_score = self._check_entity_presence(query_lower)
        score += entity_score * 0.3
        total_checks += 0.3

        final_score = score / total_checks if total_checks > 0 else 0.0
        return final_score >= self.confidence_threshold, final_score

    def _check_entity_presence(self, query_lower: str) -> float:
        total_required = len(self.required_entities)
        total_optional = len(self.optional_entities)

        if total_required == 0 and total_optional == 0:
            return 1.0

        required_found = sum(1 for entity in self.required_entities if entity in query_lower)
        optional_found = sum(1 for entity in self.optional_entities if entity in query_lower)

        required_score = required_found / total_required if total_required > 0 else 1.0
        optional_score = optional_found / total_optional if total_optional > 0 else 0.0
        return required_score * 0.8 + optional_score * 0.2


class PatternMatcher:
    """Matches queries against logical patterns for intelligent processing."""

    def __init__(self) -> None:
        self.patterns: List[LogicPattern] = []
        self.pattern_cache: Dict[str, Tuple[Optional[LogicPattern], float]] = {}

    def add_pattern(self, pattern: LogicPattern) -> None:
        self.patterns.append(pattern)

    def match_query(self, query: str) -> Tuple[Optional[LogicPattern], float]:
        if query in self.pattern_cache:
            return self.pattern_cache[query]

        best_pattern: Optional[LogicPattern] = None
        best_score = 0.0

        for pattern in self.patterns:
            matches, score = pattern.matches(query)
            if matches and score > best_score:
                best_pattern = pattern
                best_score = score

        self.pattern_cache[query] = (best_pattern, best_score)
        return best_pattern, best_score

    def get_all_matches(self, query: str, min_confidence: float = 0.5) -> List[Tuple[LogicPattern, float]]:
        matches: List[Tuple[LogicPattern, float]] = []
        for pattern in self.patterns:
            pattern_matches, score = pattern.matches(query)
            if pattern_matches and score >= min_confidence:
                matches.append((pattern, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches

    def estimate_token_reduction(self, query: str) -> float:
        pattern, confidence = self.match_query(query)
        if pattern:
            return pattern.token_reduction_potential * confidence
        return 0.0


class BuiltinPatterns:
    """Factory for creating built-in logic patterns."""

    @staticmethod
    def create_all_patterns() -> List[LogicPattern]:
        patterns: List[LogicPattern] = []
        patterns.extend(BuiltinPatterns._create_general_patterns())
        patterns.extend(BuiltinPatterns._create_temporal_patterns())
        patterns.extend(BuiltinPatterns._create_quantitative_patterns())
        patterns.extend(BuiltinPatterns._create_comparative_patterns())
        patterns.extend(BuiltinPatterns._create_causal_patterns())
        patterns.extend(BuiltinPatterns._create_procedural_patterns())
        patterns.extend(BuiltinPatterns._create_spatial_patterns())
        patterns.extend(BuiltinPatterns._create_trend_patterns())
        return patterns

    @staticmethod
    def create_pattern_matcher() -> PatternMatcher:
        matcher = PatternMatcher()
        for pattern in BuiltinPatterns.create_all_patterns():
            matcher.add_pattern(pattern)
        return matcher

    @staticmethod
    def _create_general_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="insight_overview",
                pattern_type=PatternType.GENERAL,
                keywords=["overview", "context", "insight", "explain", "summary", "understand"],
                optional_entities=["topic", "entity", "context"],
                constraints=[
                    LogicalConstraint(
                        "topic",
                        LogicalOperator.EXISTS,
                        None,
                        "Clarify the topic or focus area",
                        required=False,
                    )
                ],
                token_reduction_potential=0.2,
                refinement_questions=[
                    "Which aspect should we focus on?",
                    "Is there a particular audience or stakeholder?",
                ],
            )
        ]

    @staticmethod
    def _create_temporal_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="temporal_specific",
                pattern_type=PatternType.TEMPORAL,
                keywords=["when", "date", "time", "timeline", "schedule", "deadline"],
                regex_patterns=[
                    re.compile(r"\b(\d{4}|today|tomorrow|yesterday)\b", re.IGNORECASE),
                    re.compile(r"\b(q[1-4]|quarter|month|week)\b", re.IGNORECASE),
                ],
                required_entities=["time_reference"],
                optional_entities=["duration", "timezone"],
                constraints=[
                    LogicalConstraint(
                        "time_reference",
                        LogicalOperator.EXISTS,
                        None,
                        "Time reference is required for temporal queries",
                    )
                ],
                refinement_questions=[
                    "What time period or range should we focus on?",
                    "Is there a relevant timezone or calendar system?",
                ],
            )
        ]

    @staticmethod
    def _create_quantitative_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="quantitative_metric_lookup",
                pattern_type=PatternType.QUANTITATIVE,
                keywords=[
                    "how many",
                    "how much",
                    "metric",
                    "rate",
                    "percentage",
                    "statistics",
                    "average",
                    "count",
                    "trend",
                ],
                regex_patterns=[
                    re.compile(r"\b\d+%\b"),
                    re.compile(r"\b(increase|decrease|growth)\b", re.IGNORECASE),
                ],
                required_entities=["metric", "subject"],
                optional_entities=["unit", "time_period", "location"],
                constraints=[
                    LogicalConstraint("metric", LogicalOperator.EXISTS, None, "Metric or KPI must be identified"),
                    LogicalConstraint("subject", LogicalOperator.EXISTS, None, "Subject or entity of measurement required"),
                ],
                token_reduction_potential=0.35,
                refinement_questions=[
                    "Which metric or KPI matters most?",
                    "Which entity or cohort should we analyse?",
                    "Is there a relevant time frame or location?",
                ],
            )
        ]

    @staticmethod
    def _create_comparative_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="comparative_evaluation",
                pattern_type=PatternType.COMPARATIVE,
                keywords=["compare", "versus", "vs", "difference", "better", "worse", "trade-off"],
                regex_patterns=[re.compile(r"\b(vs|versus)\b", re.IGNORECASE)],
                required_entities=["entities"],
                optional_entities=["comparison_dimension", "criteria", "time_period"],
                constraints=[
                    LogicalConstraint(
                        "entities",
                        LogicalOperator.GREATER_EQUAL,
                        2,
                        "Comparisons need at least two entities",
                    )
                ],
                refinement_questions=[
                    "Which options or entities should we evaluate?",
                    "What criteria or dimensions matter most?",
                ],
            )
        ]

    @staticmethod
    def _create_causal_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="causal_analysis",
                pattern_type=PatternType.CAUSAL,
                keywords=["why", "cause", "effect", "impact", "due to", "lead to", "influence"],
                regex_patterns=[re.compile(r"\b(because|resulted in|led to)\b", re.IGNORECASE)],
                required_entities=["effect"],
                optional_entities=["cause", "context", "conditions"],
                constraints=[
                    LogicalConstraint("effect", LogicalOperator.EXISTS, None, "Specify the observed effect or outcome"),
                ],
                refinement_questions=[
                    "What outcome are you analysing?",
                    "Are there suspected causes or contributing factors?",
                ],
            )
        ]

    @staticmethod
    def _create_procedural_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="procedural_steps",
                pattern_type=PatternType.PROCEDURAL,
                keywords=["how to", "steps", "process", "workflow", "procedure", "guide"],
                regex_patterns=[re.compile(r"\b(step\s*\d+|checklist)\b", re.IGNORECASE)],
                optional_entities=["subject", "tools", "constraints"],
                constraints=[
                    LogicalConstraint(
                        "subject",
                        LogicalOperator.EXISTS,
                        None,
                        "Clarify the subject or goal of the procedure",
                        required=False,
                    )
                ],
                token_reduction_potential=0.25,
                refinement_questions=[
                    "What process or subject should the steps cover?",
                    "Are there specific tools or constraints to account for?",
                ],
            )
        ]

    @staticmethod
    def _create_spatial_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="spatial_focus",
                pattern_type=PatternType.SPATIAL,
                keywords=["where", "location", "region", "map", "geography", "distribution"],
                regex_patterns=[re.compile(r"\b(north|south|east|west|global|regional)\b", re.IGNORECASE)],
                required_entities=["location"],
                optional_entities=["scale", "time_period"],
                constraints=[
                    LogicalConstraint("location", LogicalOperator.EXISTS, None, "Location or geography must be specified"),
                ],
                refinement_questions=[
                    "Which location, region, or environment matters here?",
                    "Do we need a particular level of granularity?",
                ],
            )
        ]

    @staticmethod
    def _create_trend_patterns() -> List[LogicPattern]:
        return [
            LogicPattern(
                name="trend_analysis",
                pattern_type=PatternType.TREND,
                keywords=["trend", "over time", "evolution", "history", "progression"],
                regex_patterns=[re.compile(r"\b(since|over the past|historical)\b", re.IGNORECASE)],
                required_entities=["time_reference"],
                optional_entities=["metric", "subject"],
                constraints=[
                    LogicalConstraint(
                        "time_reference",
                        LogicalOperator.EXISTS,
                        None,
                        "Indicate the time horizon for the trend",
                    )
                ],
                refinement_questions=[
                    "What time range should we analyse trends across?",
                    "Which metric or subject defines the trend?",
                ],
            )
        ]
