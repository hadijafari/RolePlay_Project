#!/bin/bash
# Render.com startup script for AI Interview Platform

echo "🚀 Starting AI Interview Platform API on Render..."
echo "📅 Starting at: $(date)"

# Install system dependencies if needed
echo "🔧 Installing system dependencies..."

# For audio processing (pydub) - install ffmpeg
apt-get update && apt-get install -y ffmpeg

# Start the FastAPI application
echo "🌐 Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port $PORT
