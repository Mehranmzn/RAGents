# Custom Agents

Learn how to build your own specialized agent types by extending RAGents' base classes and implementing custom behavior patterns.

## Overview

RAGents provides a flexible foundation for creating custom agents. You can:
- Extend the base `Agent` class for completely custom behavior
- Inherit from existing agent types and customize specific aspects
- Create mixins for reusable agent capabilities
- Implement domain-specific reasoning patterns

## Base Agent Architecture

### Agent Base Class

All agents inherit from the `Agent` base class:

```python
from ragents.agents.base import Agent
from ragents import AgentConfig
from ragents.llm.client import LLMClient

class CustomAgent(Agent):
    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        **kwargs
    ):
        super().__init__(config, llm_client, **kwargs)

        # Custom initialization
        self.custom_state = {}
        self.specialized_tools = []

    async def process_message(self, message: str, **kwargs) -> str:
        """Main message processing method to override."""
        # Your custom processing logic
        return await self._custom_processing_pipeline(message, **kwargs)

    async def _custom_processing_pipeline(self, message: str, **kwargs) -> str:
        """Implement your custom processing pipeline."""
        # 1. Preprocess message
        processed_message = await self._preprocess_message(message)

        # 2. Custom reasoning
        reasoning_result = await self._custom_reasoning(processed_message)

        # 3. Generate response
        response = await self._generate_response(reasoning_result)

        return response
```

## Creating Specialized Agents

### Domain-Specific Agent

Create an agent specialized for a specific domain:

```python
from ragents.agents.base import Agent
from ragents.tools import tool

class MedicalAssistantAgent(Agent):
    """Specialized agent for medical information assistance."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)

        # Medical-specific configuration
        self.medical_databases = ["pubmed", "mayo_clinic", "medline"]
        self.safety_filters = ["medication_checker", "contraindication_detector"]

    async def process_message(self, message: str, **kwargs) -> str:
        # Add medical safety checks
        if await self._requires_medical_disclaimer(message):
            disclaimer = "⚠️ This is for informational purposes only. Consult a healthcare professional."

        # Check for drug interactions
        if await self._mentions_medications(message):
            interaction_check = await self._check_drug_interactions(message)

        # Process with medical context
        return await self._medical_response_pipeline(message, **kwargs)

    async def _medical_response_pipeline(self, message: str, **kwargs) -> str:
        """Medical-specific processing pipeline."""

        # 1. Extract medical entities
        entities = await self._extract_medical_entities(message)

        # 2. Search medical knowledge bases
        medical_context = await self._search_medical_databases(entities)

        # 3. Apply safety filters
        filtered_context = await self._apply_safety_filters(medical_context)

        # 4. Generate response with medical guidelines
        response = await self._generate_medical_response(
            message, filtered_context, entities
        )

        return response

    @tool(name="medical_entity_extractor")
    async def _extract_medical_entities(self, text: str) -> dict:
        """Extract medical entities from text."""
        # Your medical NLP logic
        return {
            "symptoms": ["headache", "fever"],
            "medications": ["aspirin"],
            "conditions": ["migraine"]
        }
```

### Workflow-Specialized Agent

Create an agent for specific workflows:

```python
class CodeReviewAgent(Agent):
    """Agent specialized for code review tasks."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)

        # Code review specific setup
        self.supported_languages = ["python", "javascript", "java", "go"]
        self.review_criteria = [
            "code_quality", "security", "performance",
            "maintainability", "documentation"
        ]

    async def process_message(self, message: str, **kwargs) -> str:
        # Check if this is a code review request
        if "review" in message.lower() and "code" in message.lower():
            return await self._conduct_code_review(message, **kwargs)
        else:
            return await super().process_message(message, **kwargs)

    async def _conduct_code_review(self, message: str, **kwargs) -> str:
        """Conduct comprehensive code review."""

        # 1. Extract code from message or attachments
        code_content = await self._extract_code(message, kwargs.get("attachments"))

        # 2. Analyze code structure
        analysis = await self._analyze_code_structure(code_content)

        # 3. Run automated checks
        automated_issues = await self._run_automated_checks(code_content)

        # 4. Perform manual review
        manual_review = await self._manual_code_review(code_content, analysis)

        # 5. Generate comprehensive report
        review_report = await self._generate_review_report(
            analysis, automated_issues, manual_review
        )

        return review_report

    async def _analyze_code_structure(self, code: str) -> dict:
        """Analyze code structure and patterns."""
        return {
            "complexity": "medium",
            "patterns": ["singleton", "factory"],
            "architecture": "mvc",
            "test_coverage": 85
        }
```

## Agent Mixins

Create reusable capabilities with mixins:

### Memory Mixin

```python
class AdvancedMemoryMixin:
    """Mixin for advanced memory capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodic_memory = {}
        self.semantic_memory = {}
        self.working_memory = {}

    async def store_episodic_memory(self, event: dict):
        """Store episodic memory (specific events)."""
        timestamp = event.get("timestamp", time.time())
        self.episodic_memory[timestamp] = event

    async def store_semantic_memory(self, concept: str, knowledge: dict):
        """Store semantic memory (general knowledge)."""
        self.semantic_memory[concept] = knowledge

    async def recall_similar_episodes(self, current_situation: dict) -> list:
        """Recall similar past episodes."""
        # Your similarity matching logic
        similar_episodes = []
        for timestamp, episode in self.episodic_memory.items():
            similarity = self._calculate_similarity(current_situation, episode)
            if similarity > 0.7:
                similar_episodes.append(episode)
        return similar_episodes

class LearningMixin:
    """Mixin for learning capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experience_buffer = []
        self.performance_metrics = {}

    async def learn_from_interaction(self, interaction: dict):
        """Learn from user interactions."""
        self.experience_buffer.append(interaction)

        # Update performance metrics
        success = interaction.get("success", False)
        task_type = interaction.get("task_type", "general")

        if task_type not in self.performance_metrics:
            self.performance_metrics[task_type] = {"successes": 0, "total": 0}

        self.performance_metrics[task_type]["total"] += 1
        if success:
            self.performance_metrics[task_type]["successes"] += 1

    async def adapt_behavior(self):
        """Adapt behavior based on learning."""
        for task_type, metrics in self.performance_metrics.items():
            success_rate = metrics["successes"] / metrics["total"]

            if success_rate < 0.5:
                # Poor performance, adjust strategy
                await self._adjust_strategy_for_task(task_type)

# Combine mixins in custom agent
class LearningMedicalAgent(AdvancedMemoryMixin, LearningMixin, MedicalAssistantAgent):
    """Medical agent with advanced memory and learning."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)

    async def process_message(self, message: str, **kwargs) -> str:
        # Process message with parent classes
        response = await super().process_message(message, **kwargs)

        # Learn from this interaction
        interaction = {
            "message": message,
            "response": response,
            "task_type": "medical_query",
            "success": True  # Determine success criteria
        }
        await self.learn_from_interaction(interaction)

        return response
```

## Custom Reasoning Patterns

### Chain of Thought Reasoning

```python
class ChainOfThoughtAgent(Agent):
    """Agent with explicit chain of thought reasoning."""

    async def process_message(self, message: str, **kwargs) -> str:
        # Generate step-by-step reasoning
        thought_chain = await self._generate_thought_chain(message)

        # Execute each step
        results = []
        for step in thought_chain.steps:
            step_result = await self._execute_reasoning_step(step)
            results.append(step_result)

        # Synthesize final answer
        final_answer = await self._synthesize_answer(thought_chain, results)

        # Return with reasoning trace
        return {
            "answer": final_answer,
            "reasoning": thought_chain,
            "step_results": results
        }

    async def _generate_thought_chain(self, message: str) -> dict:
        """Generate explicit chain of thought."""
        prompt = f"""
        Break down this problem into clear reasoning steps:
        Problem: {message}

        Think step by step:
        1. What information do I need?
        2. What steps should I take?
        3. How will I verify my answer?
        """

        thought_response = await self.llm_client.complete(prompt)
        return self._parse_thought_chain(thought_response)
```

### Multi-Modal Reasoning

```python
class MultiModalAgent(Agent):
    """Agent capable of processing multiple modalities."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()

    async def process_message(self, message: str, **kwargs) -> str:
        attachments = kwargs.get("attachments", [])

        # Process different modalities
        processed_inputs = {
            "text": message,
            "images": [],
            "audio": [],
            "documents": []
        }

        for attachment in attachments:
            modality = self._detect_modality(attachment)
            processed_data = await self._process_modality(attachment, modality)
            processed_inputs[modality].append(processed_data)

        # Multi-modal reasoning
        unified_representation = await self._create_unified_representation(processed_inputs)
        response = await self._multi_modal_reasoning(unified_representation)

        return response

    async def _create_unified_representation(self, inputs: dict) -> dict:
        """Create unified representation from multi-modal inputs."""
        # Your multi-modal fusion logic
        return {
            "text_features": inputs["text"],
            "visual_features": self._extract_visual_features(inputs["images"]),
            "audio_features": self._extract_audio_features(inputs["audio"]),
            "semantic_alignment": self._align_modalities(inputs)
        }
```

## Agent Coordination

### Multi-Agent Systems

```python
class CoordinatorAgent(Agent):
    """Agent that coordinates multiple specialized agents."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)
        self.specialist_agents = {}

    def register_specialist(self, domain: str, agent: Agent):
        """Register a specialist agent for a domain."""
        self.specialist_agents[domain] = agent

    async def process_message(self, message: str, **kwargs) -> str:
        # Analyze which specialists are needed
        required_specialists = await self._analyze_required_expertise(message)

        if len(required_specialists) == 1:
            # Single specialist needed
            specialist = self.specialist_agents[required_specialists[0]]
            return await specialist.process_message(message, **kwargs)

        elif len(required_specialists) > 1:
            # Multiple specialists needed - coordinate
            return await self._coordinate_specialists(message, required_specialists, **kwargs)

        else:
            # Handle with general capabilities
            return await super().process_message(message, **kwargs)

    async def _coordinate_specialists(self, message: str, specialists: list, **kwargs) -> str:
        """Coordinate multiple specialists."""
        specialist_responses = {}

        # Get responses from each specialist
        for specialist_name in specialists:
            specialist = self.specialist_agents[specialist_name]
            response = await specialist.process_message(message, **kwargs)
            specialist_responses[specialist_name] = response

        # Synthesize responses
        final_response = await self._synthesize_specialist_responses(
            message, specialist_responses
        )

        return final_response

# Usage example
coordinator = CoordinatorAgent(config, llm_client)
coordinator.register_specialist("medical", MedicalAssistantAgent(config, llm_client))
coordinator.register_specialist("code_review", CodeReviewAgent(config, llm_client))
coordinator.register_specialist("legal", LegalAssistantAgent(config, llm_client))
```

## Testing Custom Agents

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock

class TestCustomAgent:
    @pytest.fixture
    async def agent(self):
        config = AgentConfig(name="Test Agent")
        llm_client = AsyncMock()
        return CustomAgent(config, llm_client)

    async def test_custom_processing(self, agent):
        """Test custom processing pipeline."""
        message = "Test message"
        response = await agent.process_message(message)

        assert response is not None
        assert isinstance(response, str)

    async def test_specialized_behavior(self, agent):
        """Test domain-specific behavior."""
        medical_query = "What are the symptoms of flu?"
        response = await agent.process_message(medical_query)

        # Verify medical-specific processing
        assert "medical" in response.lower() or "symptom" in response.lower()

    async def test_error_handling(self, agent):
        """Test error handling in custom agent."""
        # Test with malformed input
        with pytest.raises(ValueError):
            await agent.process_message("")
```

### Integration Testing

```python
async def test_agent_integration():
    """Test custom agent with real LLM."""
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    config = AgentConfig(
        name="Integration Test Agent",
        enable_tools=True,
        enable_memory=True
    )

    agent = CustomAgent(config, llm_client)

    # Test actual interaction
    response = await agent.process_message("Hello, how are you?")
    assert len(response) > 0

    # Test memory
    response2 = await agent.process_message("What did I just say?")
    assert "hello" in response2.lower()
```

## Best Practices

### Design Principles

1. **Single Responsibility** - Each agent should have a clear, focused purpose
2. **Composition over Inheritance** - Use mixins for reusable capabilities
3. **Async First** - Design all methods to be async-compatible
4. **Error Handling** - Implement robust error handling and recovery
5. **Testing** - Write comprehensive tests for custom behavior

### Performance Considerations

1. **Lazy Loading** - Load expensive resources only when needed
2. **Caching** - Cache expensive operations and computations
3. **Resource Management** - Properly manage memory and connections
4. **Batching** - Batch operations when possible for efficiency

### Maintenance

1. **Documentation** - Document custom behavior and configuration
2. **Versioning** - Version your custom agents for reproducibility
3. **Monitoring** - Add observability to custom components
4. **Backwards Compatibility** - Maintain compatibility when updating

## Example: Scientific Research Agent

```python
class ScientificResearchAgent(AdvancedMemoryMixin, Agent):
    """Agent specialized for scientific research tasks."""

    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config, llm_client)

        # Scientific tools and databases
        self.scientific_databases = ["arxiv", "pubmed", "google_scholar"]
        self.analysis_tools = ["statistical_analyzer", "citation_tracker"]

    async def process_message(self, message: str, **kwargs) -> str:
        # Detect research intent
        research_type = await self._classify_research_intent(message)

        if research_type == "literature_review":
            return await self._conduct_literature_review(message)
        elif research_type == "data_analysis":
            return await self._analyze_research_data(message, **kwargs)
        elif research_type == "hypothesis_generation":
            return await self._generate_hypotheses(message)
        else:
            return await super().process_message(message, **kwargs)

    async def _conduct_literature_review(self, query: str) -> str:
        """Conduct systematic literature review."""

        # 1. Search scientific databases
        papers = await self._search_scientific_literature(query)

        # 2. Analyze and categorize papers
        categorized_papers = await self._categorize_papers(papers)

        # 3. Extract key findings
        key_findings = await self._extract_key_findings(categorized_papers)

        # 4. Identify research gaps
        research_gaps = await self._identify_research_gaps(key_findings)

        # 5. Generate review report
        review_report = await self._generate_literature_review(
            key_findings, research_gaps, categorized_papers
        )

        return review_report
```

## Next Steps

- **[RAG Engine](../rag/overview.md)** - Enhance agents with knowledge retrieval
- **[Tools Development](../api/tools.md)** - Create custom tools for your agents
- **[Advanced Features](../advanced/logical-reasoning.md)** - Add advanced reasoning capabilities
- **[Deployment](../deployment/production.md)** - Deploy custom agents to production