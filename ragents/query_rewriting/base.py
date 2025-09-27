"""Base classes for query rewriting and prompt optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field


class RewriteStrategy(Enum):
    """Available query rewriting strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    CHAIN_OF_DENSITY = "chain_of_density"
    HYPOTHESIS_REFINEMENT = "hypothesis_refinement"
    CONTEXTUAL = "contextual"
    DECOMPOSITION = "decomposition"
    PERSONA_BASED = "persona_based"


@dataclass
class RewriteResult:
    """Result of a query rewriting operation."""
    original_query: str
    rewritten_query: str
    strategy: RewriteStrategy
    confidence_score: float
    metadata: Dict[str, Any]
    timestamp: datetime
    reasoning: Optional[str] = None
    intermediate_steps: Optional[List[str]] = None


class QueryRewriter(ABC):
    """Abstract base class for query rewriters."""

    def __init__(self, strategy: RewriteStrategy):
        self.strategy = strategy
        self.optimization_history: List[RewriteResult] = []

    @abstractmethod
    async def rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> RewriteResult:
        """Rewrite a query using the specific strategy."""
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the base prompt template for this rewriter."""
        pass

    def add_to_history(self, result: RewriteResult):
        """Add a rewrite result to the optimization history."""
        self.optimization_history.append(result)

    def get_recent_performance(self, limit: int = 10) -> List[RewriteResult]:
        """Get recent rewriting performance for optimization."""
        return self.optimization_history[-limit:]


class PromptTemplate(BaseModel):
    """Structured prompt template with variables and constraints."""

    template: str = Field(description="The prompt template with placeholders")
    variables: List[str] = Field(description="List of variable names in template")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints for variables")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="Example inputs/outputs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)

    def validate_variables(self, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return True


class OptimizationObjective(BaseModel):
    """Defines optimization objectives for prompt improvement."""

    primary_metric: str = Field(description="Primary metric to optimize")
    secondary_metrics: List[str] = Field(default_factory=list, description="Secondary metrics")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optimization constraints")
    target_performance: Optional[float] = Field(None, description="Target performance threshold")

    class Config:
        extra = "allow"


class RewriteConfig(BaseModel):
    """Configuration for query rewriting operations."""

    strategy: RewriteStrategy
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    use_examples: bool = True
    preserve_intent: bool = True
    optimize_for_rag: bool = True
    custom_instructions: Optional[str] = None
    context_window_size: int = 2048

    class Config:
        use_enum_values = True