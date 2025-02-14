# Semantic Cache Chatbot

A Python-based terminal chatbot that supports multiple LLM providers (OpenAI and Anthropic) with semantic caching for efficient response retrieval.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic)
- Semantic caching using vector embeddings and Pinecone
- Efficient response retrieval for similar questions
- Interactive terminal interface
- Conversation history support

## Setup Instructions

### 1. Prerequisites

Before starting, ensure you have:

- Python 3.7 or higher installed
- A Pinecone account (free tier available at [pinecone.io](https://www.pinecone.io))
- OpenAI API key and/or Anthropic API key

### 2. Virtual Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
# Install dependencies
pip install -r requirements.txt

# For development (includes testing tools):
pip install -e ".[dev]"
```

### 4. Pinecone Setup

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io)
2. Create a new project in Pinecone
3. Create an index with the following settings:
   - Name: `chatbot`
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: Cosine
   - Pod Type: Starter (free tier)
4. Copy your API key and environment from the Pinecone console

### 5. Environment Configuration

Create a `.env` file in the project root directory with the following settings:

```env
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=chatbot
PINECONE_NAMESPACE=default

# Provider Settings
DEFAULT_PROVIDER=openai  # or anthropic
TEMPERATURE=0.7
MAX_TOKENS=1024  # optional

# Cache Settings
SIMILARITY_THRESHOLD=0.85
CACHE_TTL_DAYS=30
```

## Running the Chatbot

1. Ensure your virtual environment is activated:

```bash
source venv/bin/activate
```

2. Start the chatbot:

```bash
python -m chatbot.cli
```

## Development

Run tests:

```bash
pytest tests/
```

## Exiting

1. To exit the chatbot, type `exit` or press `Ctrl+C` in the terminal
2. To deactivate the virtual environment when you're done:

```bash
deactivate
```

## Troubleshooting

1. If you see "command not found: python", try using `python3` instead
2. Ensure your Pinecone index is created with the correct dimensions (1536)
3. Verify all API keys in your `.env` file are correct
4. Check that your Pinecone index is in the "Ready" state in the Pinecone console

## License

GNU General Public License v3.0 (GPLv3)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
