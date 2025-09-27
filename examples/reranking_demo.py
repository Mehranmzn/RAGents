"""Demonstration of RAGents reranking and Autocut capabilities."""

import asyncio
import os
from typing import Dict, List, Optional

from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env
from ragents.reranking import (
    SemanticReranker,
    CrossEncoderReranker,
    HybridReranker,
    LLMReranker,
    AutocutFilter,
    RerankingEvaluator,
    RerankingConfig,
)
from ragents.reranking.base import RetrievedDocument
from ragents.reranking.autocut import CutoffStrategy


async def create_sample_documents() -> List[RetrievedDocument]:
    """Create sample retrieved documents for demonstration."""

    documents = [
        RetrievedDocument(
            content="Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns from data without explicit programming.",
            metadata={"source": "ai_basics.pdf", "page": 1},
            similarity_score=0.95,
            document_id="doc_1",
            source="ai_basics.pdf"
        ),
        RetrievedDocument(
            content="Deep learning networks use multiple layers to progressively extract higher-level features from raw input.",
            metadata={"source": "deep_learning.pdf", "page": 3},
            similarity_score=0.87,
            document_id="doc_2",
            source="deep_learning.pdf"
        ),
        RetrievedDocument(
            content="Neural networks are inspired by biological neurons and consist of interconnected nodes that process information.",
            metadata={"source": "neural_networks.pdf", "page": 2},
            similarity_score=0.82,
            document_id="doc_3",
            source="neural_networks.pdf"
        ),
        RetrievedDocument(
            content="Supervised learning requires labeled training data to learn the relationship between inputs and outputs.",
            metadata={"source": "ml_types.pdf", "page": 1},
            similarity_score=0.75,
            document_id="doc_4",
            source="ml_types.pdf"
        ),
        RetrievedDocument(
            content="Natural language processing enables computers to understand and generate human language.",
            metadata={"source": "nlp_intro.pdf", "page": 1},
            similarity_score=0.68,
            document_id="doc_5",
            source="nlp_intro.pdf"
        ),
        RetrievedDocument(
            content="Computer vision algorithms can analyze and interpret visual information from images and videos.",
            metadata={"source": "computer_vision.pdf", "page": 1},
            similarity_score=0.52,
            document_id="doc_6",
            source="computer_vision.pdf"
        ),
        RetrievedDocument(
            content="Reinforcement learning agents learn through trial and error by receiving rewards or penalties.",
            metadata={"source": "rl_basics.pdf", "page": 2},
            similarity_score=0.45,
            document_id="doc_7",
            source="rl_basics.pdf"
        ),
        RetrievedDocument(
            content="The weather forecast predicts sunny skies with temperatures reaching 75 degrees.",
            metadata={"source": "weather_report.txt", "page": 1},
            similarity_score=0.15,
            document_id="doc_8",
            source="weather_report.txt"
        ),
        RetrievedDocument(
            content="Recipe for chocolate chip cookies: Mix flour, sugar, eggs, and chocolate chips.",
            metadata={"source": "recipes.txt", "page": 5},
            similarity_score=0.08,
            document_id="doc_9",
            source="recipes.txt"
        ),
        RetrievedDocument(
            content="The stock market closed higher today with technology shares leading the gains.",
            metadata={"source": "financial_news.txt", "page": 1},
            similarity_score=0.05,
            document_id="doc_10",
            source="financial_news.txt"
        )
    ]

    return documents


async def demo_basic_reranking():
    """Demonstrate basic reranking strategies."""
    print("🔧 RAGents Reranking Demo")
    print("=" * 60)

    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create sample documents
    documents = await create_sample_documents()
    query = "What is machine learning and how does it work?"

    print(f"📝 Query: {query}")
    print(f"📄 Original Documents: {len(documents)} documents retrieved")

    # Show original ranking
    print("\n📊 Original Ranking:")
    for i, doc in enumerate(documents[:5], 1):
        print(f"  {i}. Score: {doc.similarity_score:.2f} | {doc.content[:80]}...")

    # Initialize rerankers
    rerankers = {
        "Semantic": SemanticReranker(),
        "Cross-Encoder": CrossEncoderReranker(),
        "Hybrid": HybridReranker(),
        "LLM-Based": LLMReranker(llm_client),
    }

    for strategy_name, reranker in rerankers.items():
        print(f"\n🔧 {strategy_name} Reranking:")
        print("-" * 40)

        try:
            result = await reranker.rerank(query, documents, top_k=5)

            print(f"   Processing Time: {result.processing_time:.3f}s")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Reranked Results:")

            for i, doc in enumerate(result.reranked_documents, 1):
                print(f"     {i}. Score: {doc.similarity_score:.2f} | {doc.content[:60]}...")

        except Exception as e:
            print(f"❌ Error with {strategy_name}: {e}")

    print("\n" + "=" * 60)


async def demo_autocut_filtering():
    """Demonstrate Autocut filtering with different strategies."""
    print("\n🔪 Autocut Filtering Demo")
    print("-" * 40)

    documents = await create_sample_documents()

    # Test different cutoff strategies
    cutoff_strategies = [
        CutoffStrategy.PERCENTILE,
        CutoffStrategy.GRADIENT_CHANGE,
        CutoffStrategy.ELBOW_METHOD,
        CutoffStrategy.ADAPTIVE_THRESHOLD,
    ]

    print(f"📄 Original Documents: {len(documents)} documents")
    print("📊 Original Similarity Scores:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc.similarity_score:.2f}")

    for strategy in cutoff_strategies:
        print(f"\n🔪 {strategy.value.title()} Strategy:")

        autocut_filter = AutocutFilter(strategy)
        try:
            filtered_docs, cutoff_result = autocut_filter.filter_documents(documents, strategy)

            print(f"   Cutoff Index: {cutoff_result.cutoff_index}")
            print(f"   Cutoff Score: {cutoff_result.cutoff_score:.2f}")
            print(f"   Documents Kept: {cutoff_result.kept_count}")
            print(f"   Documents Removed: {cutoff_result.removed_count}")
            print(f"   Confidence: {cutoff_result.confidence:.2f}")
            print(f"   Score Gap: {cutoff_result.score_gap:.2f}")

            print("   Kept Documents:")
            for i, doc in enumerate(filtered_docs[:3], 1):
                print(f"     {i}. Score: {doc.similarity_score:.2f} | {doc.content[:50]}...")

        except Exception as e:
            print(f"❌ Error with {strategy.value}: {e}")


async def demo_reranking_with_autocut():
    """Demonstrate combined reranking and Autocut filtering."""
    print("\n🚀 Combined Reranking + Autocut Demo")
    print("-" * 40)

    # Initialize components
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    documents = await create_sample_documents()
    query = "Explain deep learning and neural networks"

    # Use hybrid reranker
    reranker = HybridReranker()
    autocut_filter = AutocutFilter(CutoffStrategy.ADAPTIVE_THRESHOLD)

    print(f"📝 Query: {query}")
    print(f"📄 Starting with {len(documents)} documents")

    # Step 1: Reranking
    print("\n🔧 Step 1: Reranking")
    reranking_result = await reranker.rerank(query, documents)

    print(f"   Reranking Confidence: {reranking_result.confidence_score:.2f}")
    print(f"   Top 5 After Reranking:")
    for i, doc in enumerate(reranking_result.reranked_documents[:5], 1):
        print(f"     {i}. Score: {doc.similarity_score:.2f} | {doc.content[:60]}...")

    # Step 2: Autocut Filtering
    print("\n🔪 Step 2: Autocut Filtering")
    filtered_docs, cutoff_result = autocut_filter.filter_documents(
        reranking_result.reranked_documents
    )

    print(f"   Cutoff Applied: {cutoff_result.strategy.value}")
    print(f"   Final Document Count: {len(filtered_docs)}")
    print(f"   Cutoff Confidence: {cutoff_result.confidence:.2f}")
    print(f"   Removed {cutoff_result.removed_count} irrelevant documents")

    print("\n✅ Final Filtered Results:")
    for i, doc in enumerate(filtered_docs, 1):
        print(f"   {i}. Score: {doc.similarity_score:.2f} | {doc.content[:70]}...")


async def demo_reranking_evaluation():
    """Demonstrate reranking evaluation metrics."""
    print("\n📊 Reranking Evaluation Demo")
    print("-" * 40)

    # Initialize components
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)
    evaluator = RerankingEvaluator()

    documents = await create_sample_documents()
    query = "What is machine learning?"

    # Create ground truth relevance scores
    ground_truth = {
        "doc_1": 1.0,  # Highly relevant
        "doc_2": 0.8,  # Relevant
        "doc_3": 0.7,  # Somewhat relevant
        "doc_4": 0.6,  # Somewhat relevant
        "doc_5": 0.3,  # Low relevance
        "doc_6": 0.2,  # Low relevance
        "doc_7": 0.4,  # Low-medium relevance
        "doc_8": 0.0,  # Irrelevant
        "doc_9": 0.0,  # Irrelevant
        "doc_10": 0.0, # Irrelevant
    }

    print(f"📝 Query: {query}")
    print(f"📊 Ground Truth Provided: {len(ground_truth)} documents")

    # Test different rerankers
    rerankers = {
        "Semantic": SemanticReranker(),
        "Hybrid": HybridReranker(),
    }

    for strategy_name, reranker in rerankers.items():
        print(f"\n🔧 Evaluating {strategy_name} Reranker:")

        try:
            # Perform reranking
            result = await reranker.rerank(query, documents, top_k=10)

            # Evaluate results
            metrics = await evaluator.evaluate_reranking(result, ground_truth)

            print(f"   📈 Evaluation Metrics:")
            print(f"      Precision@K: {metrics.precision_at_k:.3f}")
            print(f"      Recall@K: {metrics.recall_at_k:.3f}")
            print(f"      NDCG@K: {metrics.ndcg_at_k:.3f}")
            print(f"      MAP Score: {metrics.map_score:.3f}")
            print(f"      MRR Score: {metrics.mrr_score:.3f}")
            print(f"      Reranking Effectiveness: {metrics.reranking_effectiveness:.3f}")
            print(f"      Processing Efficiency: {metrics.processing_efficiency:.3f}")
            print(f"      Confidence Accuracy: {metrics.confidence_accuracy:.3f}")

        except Exception as e:
            print(f"❌ Error evaluating {strategy_name}: {e}")


async def demo_configuration_profiles():
    """Demonstrate different reranking configuration profiles."""
    print("\n⚙️  Configuration Profiles Demo")
    print("-" * 40)

    from ragents.reranking.config import get_profile_config, RerankingProfile

    profiles = [
        RerankingProfile.PRECISION_FOCUSED,
        RerankingProfile.RECALL_FOCUSED,
        RerankingProfile.BALANCED,
        RerankingProfile.SPEED_OPTIMIZED,
    ]

    for profile in profiles:
        config = get_profile_config(profile)
        print(f"\n📋 {profile.value.title()} Profile:")
        print(f"   Strategy: {config.strategy.value}")
        print(f"   Top K: {config.top_k}")
        print(f"   Min Similarity: {config.min_similarity_threshold}")
        print(f"   Autocut Enabled: {config.enable_autocut}")
        print(f"   Cutoff Strategy: {config.cutoff_strategy.value}")
        if hasattr(config, 'fusion_weights'):
            print(f"   Fusion Weights: {config.fusion_weights}")


async def main():
    """Run the complete reranking demonstration."""
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    print("🌟 Welcome to RAGents Reranking and Autocut Demo!")
    print("This demo showcases advanced document reranking and relevance filtering.")

    try:
        await demo_basic_reranking()
        await demo_autocut_filtering()
        await demo_reranking_with_autocut()
        await demo_reranking_evaluation()
        await demo_configuration_profiles()

        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Benefits:")
        print("   • Reranking improves relevance of retrieved documents")
        print("   • Autocut removes irrelevant information to prevent hallucinations")
        print("   • Multiple strategies available for different use cases")
        print("   • Comprehensive evaluation metrics for optimization")
        print("   • Easy integration with existing RAG systems")

    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())