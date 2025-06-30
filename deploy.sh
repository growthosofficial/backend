#!/bin/bash

# Production deployment script for Second Brain Knowledge Management Backend

set -e

echo "🚀 Starting deployment of Second Brain Backend..."

# Check if required environment files exist
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please create it from .env.example"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs uploads ssl tmp/prompts

# Set proper permissions
chmod 755 logs uploads tmp/prompts

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build --no-cache

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo "🔍 Checking service health..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    fi
    echo "⏳ Waiting for backend to be ready... (attempt $i/30)"
    sleep 2
done

# Final health check
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend health check failed after 60 seconds"
    echo "📋 Checking logs..."
    docker-compose logs backend
    exit 1
fi

echo "✅ Deployment completed successfully!"
echo ""
echo "🌐 Services are now running:"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Update: ./deploy.sh"