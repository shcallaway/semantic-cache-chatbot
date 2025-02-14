# Semantic Cache Chatbot

A Python-based terminal chatbot that supports multiple LLM providers (OpenAI and Anthropic) with semantic caching for efficient response retrieval.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic)
- Semantic caching using vector embeddings (Pinecone or Qdrant)
- Efficient response retrieval for similar questions
- Interactive terminal interface
- Conversation history support

## Setup Instructions

### 1. Prerequisites

Before starting, ensure you have:

- Python 3.9 or higher installed (tested with Python 3.9.6)
- A Pinecone account (free tier available at [pinecone.io](https://www.pinecone.io)) or Qdrant instance
- OpenAI API key and/or Anthropic API key

### 2. Development Setup

The easiest way to set up the development environment is using the provided setup script:

```bash
# Make the setup script executable
chmod +x setup_dev.sh

# Run the setup script
./setup_dev.sh
```

This will:

- Create a Python virtual environment
- Install all dependencies including development tools (pytest, black, flake8)
- Create a .env file from the example template

Alternatively, you can set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies and development tools
python -m pip install -r requirements.txt pytest-asyncio
python -m pip install -e .
```

### 3. Running Tests

The project uses pytest with async support for testing. To run the tests:

```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Run tests with verbose output
python -m pytest tests/ -v

# Run tests with more detailed output
python -m pytest tests/ -vv

# Run a specific test file
python -m pytest tests/test_cache_manager.py -v
```

Note: The tests use mocking extensively, so no API keys or vector store setup is required to run them.

### 4. Vector Store Setup

#### Option 1: Pinecone

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io)
2. Create a new project in Pinecone
3. Create an index with the following settings:
   - Name: `chatbot`
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: Cosine
   - Pod Type: Starter (free tier)
4. Copy your API key from the Pinecone console

#### Option 2: Qdrant

1. You can either:
   - Run Qdrant locally using Docker:
     ```bash
     docker run -p 6333:6333 qdrant/qdrant
     ```
   - Use Qdrant Cloud at [cloud.qdrant.io](https://cloud.qdrant.io)
2. If using Qdrant Cloud, copy your API key from the dashboard

### 5. Environment Configuration

Create a `.env` file in the project root directory with the following settings:

```env
# LLM Provider API Keys
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here

# Vector Store Configuration
VECTOR_STORE=pinecone # Or "qdrant"

# Pinecone Configuration (when using Pinecone)
PINECONE_API_KEY=your_api_key_here

# Qdrant Configuration (when using Qdrant)
QDRANT_URL=http://localhost:6333 # Or your Qdrant Cloud URL
QDRANT_API_KEY=your_api_key_here # Optional, required for Qdrant Cloud

# Vector Store Settings
VECTOR_INDEX_NAME=chatbot
VECTOR_NAMESPACE=default

# Provider Settings
DEFAULT_PROVIDER=openai # Or "anthropic"
TEMPERATURE=0.7
MAX_TOKENS=1024 # Optional

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

### Code Linting

The project uses black for code formatting and flake8 for style checking:

```bash
# Format code with black
black .

# Check code style with flake8
flake8
```

Configuration files:

- `.flake8`: Flake8 configuration with 88 character line length to match black
- `pyproject.toml`: Black configuration and build settings

### Running Tests

The project uses pytest with async support for testing:
The project uses pytest with async support for testing:

```bash
# Run tests with verbose output
python -m pytest tests/ -v

# Run tests with more detailed output
python -m pytest tests/ -vv

# Run a specific test file
python -m pytest tests/test_cache_manager.py -v
```

Note: The tests use mocking extensively, so no API keys or vector store setup is required to run them.

## Exiting

1. To exit the chatbot, type `exit` or press `Ctrl+C` in the terminal
2. To deactivate the virtual environment when you're done:

```bash
deactivate
```

## Troubleshooting

### Environment Setup Issues

1. If you see "command not found: python", try using `python3` instead
2. For virtual environment issues:
   ```bash
   # If venv exists but seems corrupted
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   ```
3. If you get pip-related errors:
   ```bash
   # Upgrade pip to the latest version
   python -m pip install --upgrade pip
   ```

### Testing Issues

1. If tests fail with missing pytest-asyncio:
   ```bash
   python -m pip install pytest-asyncio
   ```
2. If you get SSL-related warnings with urllib3:
   - This is a known issue with LibreSSL on some systems
   - The warnings can be safely ignored for development
3. For test failures:
   - Run tests with -vv flag for detailed output: `python -m pytest tests/ -vv`
   - Ensure you're using Python 3.9 or higher
   - Make sure all test dependencies are installed

### Vector Store Issues

#### Pinecone Issues

1. Ensure your Pinecone index is created with the correct dimensions (1536)
2. Verify your API key in the `.env` file is correct
3. Check that your Pinecone index is in the "Ready" state in the console
4. If index creation fails:
   - Verify you have selected the Starter (free) tier
   - Check that you haven't exceeded the free tier limits

#### Qdrant Issues

1. If using local Qdrant:
   - Ensure Docker is running
   - Check that port 6333 is available
   - Verify the container is running with `docker ps`
2. If using Qdrant Cloud:
   - Verify your API key in the `.env` file
   - Check that your cluster is in the "Active" state
   - Ensure your firewall isn't blocking the connection

## License

GNU General Public License v3.0 (GPLv3)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
