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

1. Create account at [pinecone.io](https://www.pinecone.io)
2. Create index with:
   - Name: `chatbot`
   - Dimensions: 1536
   - Metric: Cosine
   - Pod Type: Starter (free tier)
3. Copy API key from console

### Qdrant

Choose one:

- Local: Run `docker run -p 6333:6333 qdrant/qdrant`
- Cloud: Setup at [cloud.qdrant.io](https://cloud.qdrant.io) and copy API key

### pgvector

1. Install PostgreSQL and pgvector:

```bash
brew install postgresql@15 pgvector
```

2. Create database:

```bash
createdb chatbot
psql chatbot
```

3. Start PostgreSQL shell:

```bash
psql
```

4. Create pgvector extension:

```bash
CREATE EXTENSION vector;
```

5. Create user:

```bash
CREATE USER chatbot WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE chatbot TO chatbot;
```

6. Exit PostgreSQL shell:

```bash
\q
```

## Configuration

Create a `.env` file with by copying `.env.example` and filling in the missing values.

## Usage

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run the chatbot CLI
python -m chatbot.cli

# Deactivate virtual environment
deactivate
```

## Development

### Code Linting

The project uses black for code formatting and flake8 for style checking:

```bash
# Format code with black
black .

# Run linting with flake8
flake8
```

Configuration files:

- `.flake8`: Flake8 configuration with 88 character line length to match black
- `pyproject.toml`: Black configuration and build settings

### Running Tests

The project uses pytest with async support for testing:

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
