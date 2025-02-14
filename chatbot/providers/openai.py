"""
OpenAI provider implementation.
"""

from typing import Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from chatbot.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM provider interface."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional configuration options
        """
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
        self._model = kwargs.get("model", "gpt-3.5-turbo")
        self._embedding_model = kwargs.get("embedding_model", "text-embedding-ada-002")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_system_prompt(self) -> str:
        return "You are a helpful AI assistant. Provide clear, accurate, and concise responses."

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response using OpenAI's chat completion.

        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The generated response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": self.default_system_prompt})

        messages.append({"role": "user", "content": prompt})

        response: ChatCompletion = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embeddings for the given text.

        Args:
            text: Input text to generate embeddings for

        Returns:
            List of embedding values
        """
        response = await self.client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def get_token_count(self, text: str) -> int:
        """Get the approximate number of tokens in the text.

        This is a simple approximation. For more accurate counts,
        you would need to use a proper tokenizer.

        Args:
            text: Input text

        Returns:
            Approximate number of tokens
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4
