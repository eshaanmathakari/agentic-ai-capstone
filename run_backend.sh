#!/bin/bash

# Script to run the backend development server

echo "🚀 Starting AI Portfolio Rebalancing Agent Backend..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Please update .env with your API keys and configuration"
fi

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the server
echo "🔥 Starting FastAPI server..."
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

