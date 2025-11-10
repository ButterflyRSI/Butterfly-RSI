#!/usr/bin/env bash
# Quick Start Script for Butterfly-Ollama Integration

echo "════════════════════════════════════════════════════════════"
echo "                BUTTERFLY-OLLAMA QUICK START                "
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if Ollama is running
echo "Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running!"
else
    echo "❌ Ollama is not running"
    echo "   Start it with: ollama serve"
    echo "   (Run in another terminal)"
    exit 1
fi

# Check if llama3 model is available
echo ""
echo "Checking for llama3 model..."
if ollama list | grep -q "llama3"; then
    echo "✅ llama3 model found!"
else
    echo "⚠️  llama3 model not found"
    echo "   Pull it with: ollama pull llama3"
    echo ""
    read -p "   Pull llama3 now? (y/n): " pull_model
    if [ "$pull_model" = "y" ]; then
        ollama pull llama3
    else
        echo "   Please pull a model before continuing"
        exit 1
    fi
fi

# Check if requests is installed
echo ""
echo "Checking Python dependencies..."
if python3 -c "import requests" 2>/dev/null; then
    echo "✅ requests library installed!"
else
    echo "⚠️  requests library not found"
    echo "   Installing..."
    pip install requests --break-system-packages
fi

# Check if butterfly files exist
echo ""
echo "Checking for Butterfly files..."
if [ ! -f "ButterflyRSI.py" ]; then
    echo "❌ ButterflyRSI.py not found!"
    echo "   Please copy your v4.5 file to this directory"
    exit 1
else
    echo "✅ butterfly_rsi_v4_5.py found!"
fi

if [ ! -f "butterfly_ollama.py" ]; then
    echo "❌ butterfly_ollama.py not found!"
    echo "   This file should be in the same directory"
    exit 1
else
    echo "✅ butterfly_ollama.py found!"
fi

# All checks passed
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ ALL SYSTEMS GO!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Starting Butterfly-Ollama integration..."
echo ""

# Run the integration
python3 butterfly_ollama.py
