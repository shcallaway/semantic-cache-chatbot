"""
Base provider interface for LLM implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize the LLM provider.

        Args:
            api_key: The API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The generated response text
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the text.

        Args:
            text: The input text

        Returns:
            Number of tokens in the text
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the provider.

        Returns:
            Provider name string
        """
        pass

    @property
    @abstractmethod
    def default_system_prompt(self) -> str:
        """Get the default system prompt for this provider.

        Returns:
            Default system prompt string
        """
        pass

    @property
    def default_config(self) -> Dict:
        """Get the default configuration for this provider.

        Returns:
            Dictionary of default configuration values
        """
        return {
            "temperature": 0.7,
            "max_tokens": None,
        }
