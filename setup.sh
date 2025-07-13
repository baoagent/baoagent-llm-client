#!/bin/bash
# Setup script for BaoAgent LLM Client

echo "Setting up BaoAgent LLM Client..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the environment: source venv/bin/activate"
echo "To test the client: python test_client.py"
