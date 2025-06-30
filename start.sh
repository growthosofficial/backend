#!/bin/bash

# Development startup script

set -e

echo "🔧 Starting Second Brain Backend in development mode..."

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your actual configuration values"
fi

# Create necessary directories
mkdir -p logs uploads tmp/prompts

# Install dependencies if requirements.txt is newer than last install
if [ "requirements.txt" -nt ".last_install" ] || [ ! -f ".last_install" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    touch .last_install
fi

# Pre-compute embeddings if needed
if [ ! -f "main_category_embeddings.pkl" ]; then
    echo "🧠 Pre-computing category embeddings..."
    python setup_embeddings.py
fi

# Start the development server
echo "🚀 Starting development server..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""

python main.py