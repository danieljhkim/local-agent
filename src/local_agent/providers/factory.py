"""Provider factory for creating LLM provider instances."""

import os
from typing import Dict, Type

from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .ollama import OllamaProvider
from ..config.schema import ProviderConfig


# Registry mapping provider names to their classes
PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}

# Environment variable names for API keys
API_KEY_ENV_VARS: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": "",  # Ollama doesn't require an API key
}


def create_provider(provider_config: ProviderConfig) -> LLMProvider:
    """Create a provider instance from configuration.
    
    Args:
        provider_config: Provider configuration with name, model, etc.
        
    Returns:
        Configured LLMProvider instance
        
    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    provider_name = provider_config.name
    
    # Get provider class from registry
    provider_class = PROVIDER_REGISTRY.get(provider_name)
    if not provider_class:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available providers: {available}"
        )
    
    # Get API key from config or environment
    api_key = provider_config.api_key
    if not api_key:
        env_var = API_KEY_ENV_VARS.get(provider_name, "")
        if env_var:
            api_key = os.environ.get(env_var)
        elif provider_name == "ollama":
            api_key = "default"  # Ollama doesn't require an API key
    
    if not api_key:
        env_var = API_KEY_ENV_VARS.get(provider_name, f"{provider_name.upper()}_API_KEY")
        raise ValueError(
            f"API key not found for provider '{provider_name}'. "
            f"Set in config or via {env_var} environment variable."
        )
    
    # Build provider-specific kwargs
    kwargs = {
        "api_key": api_key,
        "model": provider_config.model,
    }
    
    # Add base_url for providers that support it
    if provider_config.base_url:
        kwargs["base_url"] = provider_config.base_url
    
    return provider_class(**kwargs)


def register_provider(name: str, provider_class: Type[LLMProvider]) -> None:
    """Register a new provider type.
    
    Args:
        name: Provider name (used in config)
        provider_class: Provider class to instantiate
    """
    PROVIDER_REGISTRY[name] = provider_class


def list_providers() -> list[str]:
    """List available provider names.
    
    Returns:
        List of registered provider names
    """
    return list(PROVIDER_REGISTRY.keys())
