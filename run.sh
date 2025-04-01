#!/bin/bash

# Project 42 - Launch Script
echo "🚀 Starting Project 42 Speech Assistant..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "🔧 Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Check if Ollama is running
echo "🔍 Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️ Ollama seems to be not running. Starting Ollama..."
    # Try to start Ollama (may require appropriate permissions)
    ollama serve &
    
    # Wait a bit for Ollama to start
    echo "⏳ Waiting for Ollama to start..."
    sleep 5
fi

# Start the application
echo "✨ Launching Project 42..."
python main.py

# Deactivate virtual environment when done
deactivate
