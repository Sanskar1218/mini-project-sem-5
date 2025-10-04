#!/bin/bash

echo "================================"
echo "Stock Forecasting API Starter"
echo "================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo "✓ .env created. Please edit it with your credentials:"
    echo "  nano .env"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if credentials are set
if [ -z "$ANGEL_API_KEY" ] || [ "$ANGEL_API_KEY" = "your_api_key_here" ]; then
    echo "❌ ERROR: Please configure your credentials in .env file!"
    echo "Edit .env and add your Angel One API credentials"
    exit 1
fi

echo "✓ Environment configured"
echo "✓ Starting API server..."
echo ""
echo "API will be available at: http://localhost:5000"
echo "Health check: http://localhost:5000/health"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

# Run the API
python stock_forecast_api.py