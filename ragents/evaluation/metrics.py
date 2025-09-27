"""RAGAS-style evaluation metrics for RAG systems."""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole
from .types import EvaluationDataPoint


class RAGMetric(ABC):
    """Base class for RAG evaluation metrics."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    @abstractmethod
    async def evaluate(self, data_point: EvaluationDataPoint) -> float:
        """Evaluate the metric for a data point."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @property
    def description(self) -> str:
        """Description of the metric."""
        return f"{self.name} metric for RAG evaluation"


class Faithfulness(RAGMetric):
    """Measures how faithful the answer is to the retrieved contexts."""

    @property
    def name(self) -> str:
        return "faithfulness"

    async def evaluate(self, data_point: EvaluationDataPoint) -> float:
        """Evaluate faithfulness by checking if answer statements are supported by context."""
        if not data_point.answer or not data_point.contexts:
            return 0.0

        # Extract statements from the answer
        statements = await self._extract_statements(data_point.answer)
        if not statements:
            return 1.0  # No statements to verify

        # Check each statement against the contexts
        supported_count = 0
        for statement in statements:
            if await self._is_statement_supported(statement, data_point.contexts):
                supported_count += 1

        return supported_count / len(statements)

    async def _extract_statements(self, answer: str) -> List[str]:
        """Extract factual statements from the answer."""
        prompt = f"""
Extract all factual statements from the following answer.
Each statement should be a complete, verifiable claim.
Return them as a numbered list.

Answer: {answer}

Statements:
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at extracting factual statements from text."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)

        # Parse the numbered list
        statements = []
        for line in response.content.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                statement = re.sub(r'^\d+\.\s*', '', line.strip())
                if statement:
                    statements.append(statement)

        return statements

    async def _is_statement_supported(self, statement: str, contexts: List[str]) -> bool:
        """Check if a statement is supported by the given contexts."""
        context_text = "\n\n".join(contexts)

        prompt = f"""
Context:
{context_text}

Statement: {statement}

Is this statement supported by the context above? Answer only "YES" or "NO".
A statement is supported if it can be directly inferred from the context or if the context provides evidence for it.
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert fact-checker. Be strict about what constitutes support."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content.strip().upper() == "YES"


class AnswerRelevance(RAGMetric):
    """Measures how relevant the answer is to the question."""

    @property
    def name(self) -> str:
        return "answer_relevance"

    async def evaluate(self, data_point: EvaluationDataPoint) -> float:
        """Evaluate answer relevance by generating questions from the answer and comparing with original."""
        if not data_point.answer:
            return 0.0

        # Generate questions that this answer could answer
        generated_questions = await self._generate_questions_from_answer(data_point.answer)
        if not generated_questions:
            return 0.0

        # Calculate similarity between original question and generated questions
        relevance_scores = []
        for gen_question in generated_questions:
            score = await self._calculate_question_similarity(data_point.question, gen_question)
            relevance_scores.append(score)

        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    async def _generate_questions_from_answer(self, answer: str) -> List[str]:
        """Generate questions that this answer could address."""
        prompt = f"""
Based on the following answer, generate 3-5 questions that this answer could appropriately address.
Make the questions specific and relevant to the content of the answer.

Answer: {answer}

Questions:
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at generating relevant questions from answers."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)

        # Parse the questions
        questions = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line.endswith('?') or re.match(r'^\d+\.', line)):
                question = re.sub(r'^\d+\.\s*', '', line)
                if question.endswith('?'):
                    questions.append(question)

        return questions

    async def _calculate_question_similarity(self, original: str, generated: str) -> float:
        """Calculate semantic similarity between two questions."""
        prompt = f"""
Question 1: {original}
Question 2: {generated}

How semantically similar are these two questions?
Rate the similarity on a scale of 0.0 to 1.0 where:
- 1.0 = Identical meaning
- 0.8 = Very similar meaning
- 0.6 = Somewhat similar
- 0.4 = Slightly related
- 0.2 = Minimally related
- 0.0 = Not related

Provide only the numerical score (e.g., 0.7).
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at measuring semantic similarity."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)

        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            return 0.0


class ContextPrecision(RAGMetric):
    """Measures precision of retrieved contexts - how many are relevant."""

    @property
    def name(self) -> str:
        return "context_precision"

    async def evaluate(self, data_point: EvaluationDataPoint) -> float:
        """Evaluate context precision by checking relevance of each context."""
        if not data_point.contexts:
            return 0.0

        relevant_count = 0
        for context in data_point.contexts:
            if await self._is_context_relevant(data_point.question, context):
                relevant_count += 1

        return relevant_count / len(data_point.contexts)

    async def _is_context_relevant(self, question: str, context: str) -> bool:
        """Check if a context is relevant to the question."""
        prompt = f"""
Question: {question}

Context: {context}

Is this context relevant for answering the question? Answer only "YES" or "NO".
A context is relevant if it contains information that could help answer the question.
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at determining context relevance."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content.strip().upper() == "YES"


class ContextRecall(RAGMetric):
    """Measures recall of retrieved contexts - how much of the ground truth is covered."""

    @property
    def name(self) -> str:
        return "context_recall"

    async def evaluate(self, data_point: EvaluationDataPoint) -> float:
        """Evaluate context recall by checking how much ground truth is covered by contexts."""
        if not data_point.ground_truth or not data_point.contexts:
            return 0.0

        # Extract key facts from ground truth
        ground_truth_facts = await self._extract_facts(data_point.ground_truth)
        if not ground_truth_facts:
            return 1.0  # No facts to cover

        # Check which facts are covered by the contexts
        covered_count = 0
        context_text = "\n\n".join(data_point.contexts)

        for fact in ground_truth_facts:
            if await self._is_fact_covered(fact, context_text):
                covered_count += 1

        return covered_count / len(ground_truth_facts)

    async def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from the ground truth."""
        prompt = f"""
Extract the key factual statements from the following text.
Focus on specific, verifiable facts that would be needed to answer related questions.
Return them as a numbered list.

Text: {text}

Key facts:
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at extracting key facts from text."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)

        # Parse the numbered list
        facts = []
        for line in response.content.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                fact = re.sub(r'^\d+\.\s*', '', line.strip())
                if fact:
                    facts.append(fact)

        return facts

    async def _is_fact_covered(self, fact: str, context_text: str) -> bool:
        """Check if a fact is covered by the context."""
        prompt = f"""
Context: {context_text}

Fact: {fact}

Is this fact covered or supported by the context above? Answer only "YES" or "NO".
A fact is covered if the context contains the same information or provides evidence for it.
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at checking fact coverage."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content.strip().upper() == "YES"


# Convenience function to create all metrics
def create_all_metrics(llm_client: LLMClient) -> List[RAGMetric]:
    """Create instances of all available metrics."""
    return [
        Faithfulness(llm_client),
        AnswerRelevance(llm_client),
        ContextPrecision(llm_client),
        ContextRecall(llm_client),
    ]