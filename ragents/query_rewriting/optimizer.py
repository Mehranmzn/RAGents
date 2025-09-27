"""Automatic prompt optimization inspired by DSPy's approach."""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

from pydantic import BaseModel

from ..llm.client import LLMClient
from .base import QueryRewriter, RewriteResult, RewriteStrategy, OptimizationObjective


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization."""
    max_iterations: int = 10
    population_size: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    convergence_threshold: float = 0.05
    evaluation_samples: int = 3
    use_genetic_algorithm: bool = True
    use_gradient_free: bool = True


class PromptCandidate(BaseModel):
    """A candidate prompt in the optimization process."""
    template: str
    variables: Dict[str, Any]
    performance_score: float
    generation: int
    parent_ids: List[str] = []
    mutation_history: List[str] = []

    class Config:
        arbitrary_types_allowed = True


class PromptOptimizer:
    """DSPy-inspired automatic prompt optimizer."""

    def __init__(
        self,
        llm_client: LLMClient,
        evaluation_function: Optional[Callable] = None
    ):
        self.llm_client = llm_client
        self.evaluation_function = evaluation_function or self._default_evaluation
        self.optimization_history: List[Dict[str, Any]] = []

    async def optimize_prompt(
        self,
        base_rewriter: QueryRewriter,
        test_queries: List[str],
        objective: OptimizationObjective,
        config: OptimizationConfig = OptimizationConfig()
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimize a prompt template using evolutionary or gradient-free methods."""

        if config.use_genetic_algorithm:
            return await self._genetic_optimization(
                base_rewriter, test_queries, objective, config
            )
        else:
            return await self._gradient_free_optimization(
                base_rewriter, test_queries, objective, config
            )

    async def _genetic_optimization(
        self,
        base_rewriter: QueryRewriter,
        test_queries: List[str],
        objective: OptimizationObjective,
        config: OptimizationConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """Genetic algorithm-based prompt optimization."""

        # Initialize population
        population = await self._initialize_population(
            base_rewriter, config.population_size
        )

        best_candidate = None
        best_score = float('-inf')
        generation = 0

        optimization_log = {
            "method": "genetic_algorithm",
            "generations": [],
            "convergence_data": []
        }

        while generation < config.max_iterations:
            # Evaluate population
            evaluated_population = await self._evaluate_population(
                population, test_queries, objective, config.evaluation_samples
            )

            # Track best candidate
            current_best = max(evaluated_population, key=lambda x: x.performance_score)
            if current_best.performance_score > best_score:
                best_candidate = current_best
                best_score = current_best.performance_score

            # Log generation data
            generation_data = {
                "generation": generation,
                "best_score": best_score,
                "avg_score": sum(c.performance_score for c in evaluated_population) / len(evaluated_population),
                "population_size": len(evaluated_population)
            }
            optimization_log["generations"].append(generation_data)

            # Check convergence
            if self._check_convergence(evaluated_population, config.convergence_threshold):
                break

            # Create next generation
            population = await self._create_next_generation(
                evaluated_population, config
            )

            generation += 1

        optimization_log["final_generation"] = generation
        optimization_log["best_score"] = best_score

        return best_candidate.template, optimization_log

    async def _gradient_free_optimization(
        self,
        base_rewriter: QueryRewriter,
        test_queries: List[str],
        objective: OptimizationObjective,
        config: OptimizationConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """Gradient-free optimization using random search and local improvements."""

        base_template = base_rewriter.get_prompt_template()
        best_template = base_template
        best_score = await self._evaluate_template(
            base_template, base_rewriter, test_queries, objective
        )

        optimization_log = {
            "method": "gradient_free",
            "iterations": [],
            "improvements": []
        }

        for iteration in range(config.max_iterations):
            # Generate random variations
            candidates = await self._generate_template_variations(
                best_template, config.population_size
            )

            # Evaluate candidates
            for candidate_template in candidates:
                score = await self._evaluate_template(
                    candidate_template, base_rewriter, test_queries, objective
                )

                iteration_data = {
                    "iteration": iteration,
                    "candidate_score": score,
                    "best_score": best_score
                }
                optimization_log["iterations"].append(iteration_data)

                if score > best_score:
                    best_template = candidate_template
                    best_score = score
                    optimization_log["improvements"].append({
                        "iteration": iteration,
                        "new_score": score,
                        "improvement": score - best_score
                    })

        optimization_log["final_score"] = best_score
        return best_template, optimization_log

    async def _initialize_population(
        self,
        base_rewriter: QueryRewriter,
        population_size: int
    ) -> List[PromptCandidate]:
        """Initialize the population with variations of the base template."""

        base_template = base_rewriter.get_prompt_template()
        population = []

        # Add base template
        population.append(PromptCandidate(
            template=base_template,
            variables={},
            performance_score=0.0,
            generation=0
        ))

        # Generate variations
        for i in range(population_size - 1):
            variant = await self._mutate_template(base_template)
            population.append(PromptCandidate(
                template=variant,
                variables={},
                performance_score=0.0,
                generation=0,
                mutation_history=["initial_mutation"]
            ))

        return population

    async def _evaluate_population(
        self,
        population: List[PromptCandidate],
        test_queries: List[str],
        objective: OptimizationObjective,
        evaluation_samples: int
    ) -> List[PromptCandidate]:
        """Evaluate the performance of each candidate in the population."""

        for candidate in population:
            if candidate.performance_score == 0.0:  # Not yet evaluated
                score = await self._evaluate_candidate(
                    candidate, test_queries, objective, evaluation_samples
                )
                candidate.performance_score = score

        return population

    async def _evaluate_candidate(
        self,
        candidate: PromptCandidate,
        test_queries: List[str],
        objective: OptimizationObjective,
        evaluation_samples: int
    ) -> float:
        """Evaluate a single candidate's performance."""

        scores = []
        sample_queries = random.sample(test_queries, min(evaluation_samples, len(test_queries)))

        for query in sample_queries:
            try:
                # Use the template to generate a response
                prompt = candidate.template.format(query=query, context="")
                response = await self.llm_client.complete(prompt)

                # Evaluate the response
                score = await self.evaluation_function(query, response, objective)
                scores.append(score)

            except Exception:
                scores.append(0.0)  # Penalty for invalid templates

        return sum(scores) / len(scores) if scores else 0.0

    async def _create_next_generation(
        self,
        population: List[PromptCandidate],
        config: OptimizationConfig
    ) -> List[PromptCandidate]:
        """Create the next generation using selection, crossover, and mutation."""

        # Selection: choose parents based on fitness
        sorted_population = sorted(population, key=lambda x: x.performance_score, reverse=True)
        elite_size = max(1, len(population) // 4)
        parents = sorted_population[:elite_size]

        next_generation = []

        # Keep elite
        for parent in parents:
            next_generation.append(PromptCandidate(
                template=parent.template,
                variables=parent.variables,
                performance_score=0.0,  # Re-evaluate in new generation
                generation=parent.generation + 1,
                parent_ids=[id(parent)]
            ))

        # Generate offspring
        while len(next_generation) < len(population):
            if random.random() < config.crossover_rate and len(parents) >= 2:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                child_template = await self._crossover_templates(parent1.template, parent2.template)
            else:
                # Mutation only
                parent = random.choice(parents)
                child_template = parent.template

            # Mutation
            if random.random() < config.mutation_rate:
                child_template = await self._mutate_template(child_template)

            next_generation.append(PromptCandidate(
                template=child_template,
                variables={},
                performance_score=0.0,
                generation=parents[0].generation + 1,
                parent_ids=[id(p) for p in parents[:2]] if len(parents) >= 2 else [id(parents[0])]
            ))

        return next_generation

    async def _mutate_template(self, template: str) -> str:
        """Mutate a template by making small variations."""

        mutation_prompt = f"""
        Improve this prompt template by making small, beneficial changes:

        Original template: {template}

        Make one of these types of improvements:
        1. Add clarifying instructions
        2. Improve wording for clarity
        3. Add structure or formatting
        4. Include helpful constraints
        5. Enhance specificity

        Improved template:
        """

        try:
            response = await self.llm_client.complete(mutation_prompt)
            return response.strip()
        except Exception:
            return template  # Return original if mutation fails

    async def _crossover_templates(self, template1: str, template2: str) -> str:
        """Combine two templates to create a new one."""

        crossover_prompt = f"""
        Combine the best elements of these two prompt templates:

        Template 1: {template1}

        Template 2: {template2}

        Create a new template that incorporates the strengths of both:
        """

        try:
            response = await self.llm_client.complete(crossover_prompt)
            return response.strip()
        except Exception:
            return template1  # Return first template if crossover fails

    async def _generate_template_variations(self, template: str, count: int) -> List[str]:
        """Generate multiple variations of a template."""

        variations = []
        for _ in range(count):
            variation = await self._mutate_template(template)
            variations.append(variation)
        return variations

    async def _evaluate_template(
        self,
        template: str,
        base_rewriter: QueryRewriter,
        test_queries: List[str],
        objective: OptimizationObjective
    ) -> float:
        """Evaluate a single template's performance."""

        scores = []
        for query in test_queries:
            try:
                prompt = template.format(query=query, context="")
                response = await self.llm_client.complete(prompt)
                score = await self.evaluation_function(query, response, objective)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _check_convergence(self, population: List[PromptCandidate], threshold: float) -> bool:
        """Check if the population has converged."""

        scores = [c.performance_score for c in population]
        if len(scores) < 2:
            return False

        score_variance = sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
        return score_variance < threshold

    async def _default_evaluation(
        self,
        query: str,
        response: str,
        objective: OptimizationObjective
    ) -> float:
        """Default evaluation function based on response quality metrics."""

        # Simple heuristics for response quality
        length_score = min(len(response) / 100, 1.0)  # Prefer longer responses up to a point

        # Check for relevant keywords
        relevance_keywords = ["answer", "explain", "because", "therefore", "analysis"]
        relevance_score = sum(1 for word in relevance_keywords if word.lower() in response.lower()) / len(relevance_keywords)

        # Check for structure
        structure_score = 0.0
        if any(marker in response for marker in ["1.", "2.", "•", "-"]):
            structure_score += 0.3
        if any(marker in response for marker in ["First", "Second", "Additionally", "Furthermore"]):
            structure_score += 0.2

        # Combine scores
        total_score = (length_score * 0.3 + relevance_score * 0.4 + structure_score * 0.3)
        return min(total_score, 1.0)