# Semantic Cache Chatbot

A Python-based terminal chatbot that supports multiple LLM providers (OpenAI and Anthropic) with semantic caching for efficient response retrieval.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic)
- Semantic caching using vector embeddings
- Vector store options: Pinecone, Qdrant, or pgvector
- Interactive terminal interface with conversation history
- Efficient response retrieval for similar questions

## Prerequisites

- Python 3.9 or higher
- Vector store: Pinecone account (free tier), Qdrant instance, or PostgreSQL with pgvector
- OpenAI API key and/or Anthropic API key

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install Python packages
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Vector Store Setup

### Pinecone

1. Create an account at [pinecone.io](https://www.pinecone.io)
2. Create a new index with the following settings:
   - Name: `chatbot`
   - Dimensions: 1536
   - Metric: Cosine
   - Pod Type: Starter (free tier)
3. Copy your API key from the console

### Qdrant

Qdrant can be used locally or in the cloud.

Local:

```bash
# Install Docker (if not already installed)
brew install docker

# Pull the latest Qdrant image
docker pull qdrant/qdrant

# Run a new Qdrant container and expose port 6333 to the host
docker run -p 6333:6333 qdrant/qdrant
```

Cloud:

1. Create an account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Copy your API key from the console

### pgvector

1. Install PostgreSQL and pgvector:

```bash
# Install PostgreSQL
brew install postgresql@15

# Install pgvector
brew install pgvector
```

2. Create a new PSQL database:

```bash
createdb chatbot
psql chatbot
```

3. Create the pgvector extension and a new user:

```bash
# Start a psql shell
psql -d chatbot

# Create the pgvector extension
CREATE EXTENSION vector;

# Create a new user
CREATE USER chatbot WITH PASSWORD 'password';

# Grant all privileges to the new user
GRANT ALL PRIVILEGES ON DATABASE chatbot TO chatbot;

# Exit the shell
\q
```

## Environment Variables

Create a `.env` file by copying `.env.example` and filling in the missing values.

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the chatbot CLI
python -m chatbot.cli

# Deactivate virtual environment
deactivate
```

## Development

### Code Formatting

The project uses black for code formatting:

```bash
black .
```

Configuration for black can be found in `pyproject.toml` and `.black.toml`.

### Code Linting

The project uses flake8 for linting:

```bash
flake8
```

Configuration for flake8 can be found in `.flake8`.

### Running Tests

The project uses pytest for testing:

```bash
# Run tests with verbose output
python -m pytest tests/ -v

# Run tests with more detailed output
python -m pytest tests/ -vv

# Run a specific test file
python -m pytest tests/test_cache_manager.py -v
```

## Troubleshooting

### Common Issues

#### Python/venv Issues

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

#### Testing Issues

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

#### Vector Store Issues

##### Pinecone Issues

1. Ensure your Pinecone index is created with the correct dimensions (1536)
2. Verify your API key in the `.env` file is correct
3. Check that your Pinecone index is in the "Ready" state in the console
4. If index creation fails:
   - Verify you have selected the Starter (free) tier
   - Check that you haven't exceeded the free tier limits

##### Qdrant Issues

1. If using local Qdrant:
   - Ensure Docker is running
   - Check that port 6333 is available
   - Verify the container is running with `docker ps`
2. If using Qdrant Cloud:
   - Verify your API key in the `.env` file
   - Check that your cluster is in the "Active" state
   - Ensure your firewall isn't blocking the connection

##### pgvector Issues

1. If the pgvector extension is not available:

- Verify PostgreSQL version is 11 or higher
- Check that pgvector is installed correctly for your OS
- Try reinstalling the extension: `DROP EXTENSION vector; CREATE EXTENSION vector;`

2. For connection issues:

- Verify PostgreSQL is running: `pg_isready`
- Check connection settings in `.env` file
- Ensure the database user has proper permissions
- Try connecting manually: `psql -U chatbot -d chatbot`

3. For performance issues:

- Verify the vector index is created properly
- Check PostgreSQL logs for any errors
- Consider adjusting PostgreSQL configuration for better vector search performance

## License

GNU General Public License v3.0 (GPLv3)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. See <https://www.gnu.org/licenses/>.
