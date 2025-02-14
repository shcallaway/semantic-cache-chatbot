"""
Anthropic provider implementation using Claude.
"""
from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from chatbot.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic implementation of the LLM provider interface."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration options
        """
        super().__init__(api_key, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
        self._model = kwargs.get("model", "claude-3-opus-20240229")  # or claude-3-sonnet-20240229 for a smaller model
        self._system_prompt = kwargs.get(
            "system_prompt",
            "You are Claude, a helpful AI assistant. Provide clear, accurate, and concise responses.",
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_system_prompt(self) -> str:
        return self._system_prompt

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response using Anthropic's Claude.

        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The generated response text
        """
        response = await self.client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else self.default_system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.content[0].text

    def get_token_count(self, text: str) -> int:
        """Get the approximate number of tokens in the text.

        This is a simple approximation. For more accurate counts,
        you would need to use Claude's tokenizer.

        Args:
            text: Input text

        Returns:
            Approximate number of tokens
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    @property
    def default_config(self) -> dict:
        """Get the default configuration for Anthropic.

        Returns:
            Dictionary of default configuration values
        """
        config = super().default_config
        config.update({
            "model": "claude-3-opus-20240229",  # or claude-3-sonnet-20240229 for a smaller model
            "max_tokens": 1024,
        })
        return config
