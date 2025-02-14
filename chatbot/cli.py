"""
Command-line interface for the semantic chatbot.
"""
import asyncio
import sys
from typing import Optional

import click
from openai import AsyncOpenAI

from chatbot.cache.manager import CacheManager
from chatbot.config import Config
from chatbot.providers.anthropic import AnthropicProvider
from chatbot.providers.openai import OpenAIProvider


def get_provider(config: Config, provider_override: Optional[str] = None):
    """Get the configured LLM provider.

    Args:
        config: Application configuration
        provider_override: Optional provider to use instead of default

    Returns:
        Configured LLM provider instance
    """
    provider = provider_override or config.provider.default_provider
    if provider == "openai":
        return OpenAIProvider(
            api_key=config.provider.openai_api_key,
            temperature=config.provider.temperature,
            max_tokens=config.provider.max_tokens,
        )
    else:
        return AnthropicProvider(
            api_key=config.provider.anthropic_api_key,
            temperature=config.provider.temperature,
            max_tokens=config.provider.max_tokens,
        )


async def chat_loop(
    cache_manager: CacheManager,
    system_prompt: Optional[str] = None,
):
    """Run the interactive chat loop.

    Args:
        cache_manager: Cache manager instance
        system_prompt: Optional system prompt for the LLM
    """
    click.echo("Welcome to Semantic Chatbot! Type 'exit' to quit.")
    click.echo("Using provider: " + cache_manager.provider.provider_name)
    click.echo()

    while True:
        # Get user input
        question = click.prompt("You", prompt_suffix="> ", type=str)
        
        if question.lower() in ["exit", "quit"]:
            break

        try:
            # Get response (from cache or LLM)
            response, from_cache, cache_info = await cache_manager.get_response(
                question=question,
                system_prompt=system_prompt,
            )

            # Display response
            click.echo()
            if from_cache and cache_info:
                matched_q, similarity = cache_info
                click.secho("Bot (cached)", fg="green", nl=False)
                click.echo("> " + response)
                click.secho(f"Cache hit: {similarity:.2%} similar to: {matched_q}", fg="green")
            else:
                click.secho("Bot", fg="blue", nl=False)
                click.echo("> " + response)
                click.secho("Cache miss: No similar questions found", fg="yellow")
            click.echo()

        except Exception as e:
            click.secho(f"Error: {str(e)}", fg="red")
            click.echo()


@click.group()
def cli():
    """Semantic Chatbot - A terminal chatbot with semantic caching."""
    pass


@cli.command()
@click.option(
    "--system-prompt",
    "-s",
    help="System prompt to guide the model's behavior",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic"]),
    help="LLM provider to use (overrides DEFAULT_PROVIDER setting)",
)
@click.option(
    "--vector-store",
    "-v",
    type=click.Choice(["pinecone", "qdrant"]),
    help="Vector store to use (overrides VECTOR_STORE setting)",
)
def chat(system_prompt: Optional[str], provider: Optional[str], vector_store: Optional[str]):
    """Start an interactive chat session."""
    try:
        # Load configuration
        config = Config(vector_store_override=vector_store)

        # Initialize OpenAI client for embeddings
        openai_client = AsyncOpenAI(api_key=config.provider.openai_api_key)

        # Initialize provider and cache manager
        llm_provider = get_provider(config, provider)
        cache_manager = CacheManager(config, openai_client, llm_provider)

        # Run chat loop
        asyncio.run(chat_loop(cache_manager, system_prompt))

    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--vector-store",
    "-v",
    type=click.Choice(["pinecone", "qdrant"]),
    help="Vector store to use (overrides VECTOR_STORE setting)",
)
def cleanup(vector_store: Optional[str]):
    """Clean up old cache entries."""
    try:
        # Load configuration
        config = Config(vector_store_override=vector_store)

        # Initialize OpenAI client for embeddings
        openai_client = AsyncOpenAI(api_key=config.provider.openai_api_key)

        # Initialize provider and cache manager
        llm_provider = get_provider(config, None)
        cache_manager = CacheManager(config, openai_client, llm_provider)

        # Run cleanup
        removed = asyncio.run(cache_manager.cleanup())
        click.echo(f"Removed {removed} old cache entries")

    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
