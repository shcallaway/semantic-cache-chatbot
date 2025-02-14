#!/bin/bash

# Exit on error
set -e

echo "Setting up Chatbot development environment..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (($(echo "$python_version 3.9" | awk '{print ($1 < $2)}'))); then
    echo "Error: Python 3.9 or higher is required (found $python_version)"
    exit 1
fi

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
python -m pip install -r requirements.txt pytest-asyncio
python -m pip install -e .

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env with your API keys"
fi

echo "Running test suite to verify setup..."
python -m pytest tests/ -v

echo "Setup complete! Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Create a Pinecone index named 'chatbot' (see README for details)"
echo ""
echo "Development commands:"
echo "1. Run tests: python -m pytest tests/ -v"
echo "2. Start the chatbot: python -m chatbot.cli"
echo ""
echo "Note: Some urllib3 SSL warnings during tests can be safely ignored"
