"""Environment configuration utilities."""

import os
from typing import Any, Dict, Optional

from ..llm.types import ModelConfig, ModelProvider


def get_env_config() -> Dict[str, Any]:
    """Get comprehensive environment configuration."""
    return {
        "llm_config": get_llm_config_from_env(),
        "rag_config": get_rag_env_vars(),
        "api_keys": get_api_keys_from_env(),
        "debug": os.environ.get("RAGENTS_DEBUG", "false").lower() == "true",
    }


def get_llm_config_from_env(provider: Optional[str] = None) -> ModelConfig:
    """Get LLM configuration from environment variables."""
    provider = provider or os.environ.get("RAGENTS_LLM_PROVIDER", "openai")

    provider_enum = ModelProvider(provider.lower())

    if provider_enum == ModelProvider.OPENAI:
        return ModelConfig(
            provider=provider_enum,
            model_name=os.environ.get("RAGENTS_OPENAI_MODEL", "gpt-4"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            temperature=float(os.environ.get("RAGENTS_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("RAGENTS_MAX_TOKENS", "4000"))
            if os.environ.get("RAGENTS_MAX_TOKENS")
            else None,
        )
    elif provider_enum == ModelProvider.ANTHROPIC:
        return ModelConfig(
            provider=provider_enum,
            model_name=os.environ.get("RAGENTS_ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            temperature=float(os.environ.get("RAGENTS_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("RAGENTS_MAX_TOKENS", "4000"))
            if os.environ.get("RAGENTS_MAX_TOKENS")
            else None,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_rag_env_vars() -> Dict[str, Any]:
    """Get RAG-specific environment variables."""
    return {
        "chunk_size": int(os.environ.get("RAGENTS_CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.environ.get("RAGENTS_CHUNK_OVERLAP", "200")),
        "top_k": int(os.environ.get("RAGENTS_TOP_K", "5")),
        "similarity_threshold": float(
            os.environ.get("RAGENTS_SIMILARITY_THRESHOLD", "0.7")
        ),
        "enable_vision": os.environ.get("RAGENTS_ENABLE_VISION", "false").lower()
        == "true",
        "enable_caching": os.environ.get("RAGENTS_ENABLE_CACHING", "true").lower()
        == "true",
    }


def get_api_keys_from_env() -> Dict[str, Optional[str]]:
    """Get API keys from environment variables."""
    return {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "weaviate": os.environ.get("WEAVIATE_API_KEY"),
        "huggingface": os.environ.get("HUGGINGFACE_API_KEY"),
    }


def validate_required_env_vars() -> bool:
    """Validate that required environment variables are set."""
    provider = os.environ.get("RAGENTS_LLM_PROVIDER", "openai").lower()

    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    return True