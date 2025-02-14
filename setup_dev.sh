#!/bin/bash

# Exit on error
set -e

echo "Setting up Chatbot development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env with your API keys"
fi

echo "Setup complete! Don't forget to:"
echo "1. Edit .env with your API keys"
echo "2. Create a Pinecone index named 'chatbot'"
echo ""
echo "To start the chatbot:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the chatbot: chat"
