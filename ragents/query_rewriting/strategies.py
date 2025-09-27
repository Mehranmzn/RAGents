"""Concrete implementations of query rewriting strategies."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from .base import QueryRewriter, RewriteResult, RewriteStrategy, PromptTemplate


class CoTRewriter(QueryRewriter):
    """Chain-of-Thought query rewriter for step-by-step reasoning."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(RewriteStrategy.CHAIN_OF_THOUGHT)
        self.llm_client = llm_client
        self.template = PromptTemplate(
            template="""Rewrite the following query to encourage step-by-step reasoning:

Original query: {query}

Guidelines:
1. Break down complex questions into logical steps
2. Ask for explicit reasoning at each step
3. Encourage showing work and intermediate conclusions
4. Maintain the original intent and scope

Context: {context}

Rewritten query with Chain-of-Thought prompting:""",
            variables=["query", "context"]
        )

    async def rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> RewriteResult:
        """Rewrite query to include chain-of-thought prompting."""

        context_str = self._format_context(context or {})

        prompt = self.template.format(
            query=query,
            context=context_str
        )

        response = await self.llm_client.complete(prompt)
        rewritten_query = response.strip()

        # Add explicit CoT instructions if not present
        if "step by step" not in rewritten_query.lower():
            rewritten_query += " Please think through this step by step and show your reasoning."

        confidence = self._calculate_confidence(query, rewritten_query)

        result = RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            strategy=self.strategy,
            confidence_score=confidence,
            metadata={
                "method": "cot_prompting",
                "added_reasoning_prompt": True,
                "context_used": bool(context)
            },
            timestamp=datetime.now(),
            reasoning="Enhanced query with chain-of-thought prompting for step-by-step reasoning"
        )

        self.add_to_history(result)
        return result

    def get_prompt_template(self) -> str:
        return self.template.template

    def _format_context(self, context: Dict[str, Any]) -> str:
        if not context:
            return "No additional context provided."
        return "\n".join([f"{k}: {v}" for k, v in context.items()])

    def _calculate_confidence(self, original: str, rewritten: str) -> float:
        # Simple heuristic: confidence based on length increase and reasoning keywords
        length_ratio = len(rewritten) / len(original)
        reasoning_keywords = ["step", "first", "then", "because", "therefore", "reasoning"]
        keyword_count = sum(1 for word in reasoning_keywords if word in rewritten.lower())

        confidence = min(0.5 + (length_ratio - 1) * 0.2 + keyword_count * 0.1, 1.0)
        return max(confidence, 0.1)


class FewShotRewriter(QueryRewriter):
    """Few-shot learning query rewriter with example-based prompting."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(RewriteStrategy.FEW_SHOT)
        self.llm_client = llm_client
        self.template = PromptTemplate(
            template="""Rewrite the query using few-shot prompting with relevant examples:

{examples}

Original query: {query}
Context: {context}

Following the pattern above, rewrite this query to be more specific and effective:""",
            variables=["examples", "query", "context"]
        )

    async def rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> RewriteResult:
        """Rewrite query using few-shot examples."""

        examples_str = self._format_examples(examples or self._get_default_examples())
        context_str = self._format_context(context or {})

        prompt = self.template.format(
            examples=examples_str,
            query=query,
            context=context_str
        )

        response = await self.llm_client.complete(prompt)
        rewritten_query = response.strip()

        confidence = self._calculate_confidence(query, rewritten_query, examples)

        result = RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            strategy=self.strategy,
            confidence_score=confidence,
            metadata={
                "method": "few_shot_prompting",
                "examples_count": len(examples or []),
                "context_used": bool(context)
            },
            timestamp=datetime.now(),
            reasoning="Enhanced query using few-shot examples for better specificity"
        )

        self.add_to_history(result)
        return result

    def get_prompt_template(self) -> str:
        return self.template.template

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        if not examples:
            return "No examples provided."

        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"  Input: {example.get('input', '')}")
            formatted.append(f"  Output: {example.get('output', '')}")
            formatted.append("")

        return "\n".join(formatted)

    def _format_context(self, context: Dict[str, Any]) -> str:
        if not context:
            return "No additional context provided."
        return "\n".join([f"{k}: {v}" for k, v in context.items()])

    def _get_default_examples(self) -> List[Dict[str, str]]:
        """Default examples for common query patterns."""
        return [
            {
                "input": "What is machine learning?",
                "output": "Explain machine learning, including its definition, main types (supervised, unsupervised, reinforcement learning), common algorithms, and practical applications in industry."
            },
            {
                "input": "How to optimize database performance?",
                "output": "Provide a comprehensive guide to database performance optimization, covering indexing strategies, query optimization, hardware considerations, and monitoring tools."
            }
        ]

    def _calculate_confidence(self, original: str, rewritten: str, examples: Optional[List[Dict[str, str]]]) -> float:
        # Confidence based on specificity increase and example alignment
        length_ratio = len(rewritten) / len(original)
        specificity_keywords = ["specific", "include", "explain", "describe", "analyze", "compare"]
        keyword_count = sum(1 for word in specificity_keywords if word in rewritten.lower())

        example_bonus = 0.1 if examples and len(examples) > 0 else 0
        confidence = min(0.4 + (length_ratio - 1) * 0.3 + keyword_count * 0.05 + example_bonus, 1.0)
        return max(confidence, 0.1)


class ChainOfDensityRewriter(QueryRewriter):
    """Chain-of-Density rewriter for progressive information gathering."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(RewriteStrategy.CHAIN_OF_DENSITY)
        self.llm_client = llm_client
        self.template = PromptTemplate(
            template="""Rewrite the query to use Chain-of-Density approach for comprehensive coverage:

Original query: {query}
Context: {context}

Create a series of progressively more detailed queries that:
1. Start with the basic question
2. Add layers of specificity and detail
3. Include related aspects and implications
4. Ensure comprehensive coverage

Progressive query chain:""",
            variables=["query", "context"]
        )

    async def rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> RewriteResult:
        """Rewrite query using chain-of-density approach."""

        context_str = self._format_context(context or {})

        prompt = self.template.format(
            query=query,
            context=context_str
        )

        response = await self.llm_client.complete(prompt)

        # Parse the progressive queries
        queries = self._parse_progressive_queries(response)

        # Combine into a comprehensive rewritten query
        rewritten_query = self._combine_progressive_queries(query, queries)

        confidence = self._calculate_confidence(query, rewritten_query, queries)

        result = RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            strategy=self.strategy,
            confidence_score=confidence,
            metadata={
                "method": "chain_of_density",
                "progressive_layers": len(queries),
                "context_used": bool(context)
            },
            timestamp=datetime.now(),
            reasoning="Enhanced query with progressive density layers for comprehensive coverage",
            intermediate_steps=queries
        )

        self.add_to_history(result)
        return result

    def get_prompt_template(self) -> str:
        return self.template.template

    def _format_context(self, context: Dict[str, Any]) -> str:
        if not context:
            return "No additional context provided."
        return "\n".join([f"{k}: {v}" for k, v in context.items()])

    def _parse_progressive_queries(self, response: str) -> List[str]:
        """Parse progressive queries from the LLM response."""
        lines = response.strip().split('\n')
        queries = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # Remove numbering and formatting
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                clean_line = re.sub(r'^-\s*', '', clean_line)
                if clean_line:
                    queries.append(clean_line)

        return queries[:5]  # Limit to 5 progressive layers

    def _combine_progressive_queries(self, original: str, queries: List[str]) -> str:
        """Combine progressive queries into a comprehensive rewritten query."""
        if not queries:
            return original + " Please provide a comprehensive and detailed response."

        base_query = queries[0] if queries else original

        if len(queries) > 1:
            additional_aspects = queries[1:]
            aspects_text = " Additionally, please address: " + "; ".join(additional_aspects)
            return base_query + aspects_text

        return base_query

    def _calculate_confidence(self, original: str, rewritten: str, queries: List[str]) -> float:
        # Confidence based on comprehensiveness and layer quality
        length_ratio = len(rewritten) / len(original)
        layer_count = len(queries)

        confidence = min(0.3 + (length_ratio - 1) * 0.2 + layer_count * 0.1, 1.0)
        return max(confidence, 0.1)


class HypothesisRefinementRewriter(QueryRewriter):
    """Hypothesis-based query rewriter for scientific reasoning."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(RewriteStrategy.HYPOTHESIS_REFINEMENT)
        self.llm_client = llm_client
        self.template = PromptTemplate(
            template="""Rewrite the query to include hypothesis formation and testing:

Original query: {query}
Context: {context}

Transform this into a hypothesis-driven inquiry that:
1. States potential hypotheses or assumptions
2. Asks for evidence evaluation
3. Encourages critical analysis
4. Seeks alternative explanations

Hypothesis-driven query:""",
            variables=["query", "context"]
        )

    async def rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> RewriteResult:
        """Rewrite query to include hypothesis formation and testing."""

        context_str = self._format_context(context or {})

        prompt = self.template.format(
            query=query,
            context=context_str
        )

        response = await self.llm_client.complete(prompt)
        rewritten_query = response.strip()

        # Enhance with hypothesis testing instructions
        if "hypothesis" not in rewritten_query.lower():
            rewritten_query += " Please form testable hypotheses and evaluate the supporting evidence."

        confidence = self._calculate_confidence(query, rewritten_query)

        result = RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            strategy=self.strategy,
            confidence_score=confidence,
            metadata={
                "method": "hypothesis_refinement",
                "scientific_approach": True,
                "context_used": bool(context)
            },
            timestamp=datetime.now(),
            reasoning="Enhanced query with hypothesis-driven scientific reasoning approach"
        )

        self.add_to_history(result)
        return result

    def get_prompt_template(self) -> str:
        return self.template.template

    def _format_context(self, context: Dict[str, Any]) -> str:
        if not context:
            return "No additional context provided."
        return "\n".join([f"{k}: {v}" for k, v in context.items()])

    def _calculate_confidence(self, original: str, rewritten: str) -> float:
        # Confidence based on scientific reasoning elements
        scientific_keywords = ["hypothesis", "evidence", "test", "analyze", "evaluate", "alternative"]
        keyword_count = sum(1 for word in scientific_keywords if word in rewritten.lower())
        length_ratio = len(rewritten) / len(original)

        confidence = min(0.4 + keyword_count * 0.1 + (length_ratio - 1) * 0.2, 1.0)
        return max(confidence, 0.1)


class ContextualRewriter(QueryRewriter):
    """Context-aware query rewriter that adapts based on domain and situation."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(RewriteStrategy.CONTEXTUAL)
        self.llm_client = llm_client
        self.template = PromptTemplate(
            template="""Rewrite the query to be contextually appropriate and domain-specific:

Original query: {query}
Domain context: {domain}
Situational context: {situation}
User context: {user_context}

Guidelines:
1. Adapt language and terminology for the specific domain
2. Consider the user's background and expertise level
3. Include relevant contextual constraints
4. Optimize for the specific use case

Contextually optimized query:""",
            variables=["query", "domain", "situation", "user_context"]
        )

    async def rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> RewriteResult:
        """Rewrite query with contextual adaptation."""

        ctx = context or {}
        domain = ctx.get("domain", "general")
        situation = ctx.get("situation", "information seeking")
        user_context = ctx.get("user_context", "general audience")

        prompt = self.template.format(
            query=query,
            domain=domain,
            situation=situation,
            user_context=user_context
        )

        response = await self.llm_client.complete(prompt)
        rewritten_query = response.strip()

        confidence = self._calculate_confidence(query, rewritten_query, ctx)

        result = RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            strategy=self.strategy,
            confidence_score=confidence,
            metadata={
                "method": "contextual_adaptation",
                "domain": domain,
                "situation": situation,
                "user_context": user_context
            },
            timestamp=datetime.now(),
            reasoning=f"Adapted query for {domain} domain and {situation} situation"
        )

        self.add_to_history(result)
        return result

    def get_prompt_template(self) -> str:
        return self.template.template

    def _calculate_confidence(self, original: str, rewritten: str, context: Dict[str, Any]) -> float:
        # Confidence based on context richness and adaptation
        context_richness = len([v for v in context.values() if v and v != "general"])
        length_ratio = len(rewritten) / len(original)

        confidence = min(0.5 + context_richness * 0.1 + (length_ratio - 1) * 0.2, 1.0)
        return max(confidence, 0.2)