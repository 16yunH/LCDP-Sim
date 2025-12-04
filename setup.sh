#!/bin/bash

# LCDP-Sim Quick Start Script
# This script helps you get started with LCDP-Sim quickly

echo "=================================="
echo "LCDP-Sim Quick Start"
echo "=================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash)
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # Linux/Mac
    source venv/bin/activate
else
    echo "Error: Could not find activation script"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo ""
echo "Installing LCDP-Sim..."
pip install -e .

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p videos
mkdir -p visualizations

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Collect demonstration data:"
echo "   python scripts/collect_data.py --env PickCube-v0 --num-episodes 100"
echo ""
echo "2. Train the model:"
echo "   python scripts/train.py --config configs/train_config.yaml"
echo ""
echo "3. Evaluate the policy:"
echo "   python scripts/eval.py --checkpoint checkpoints/best.pth --num-episodes 50"
echo ""
echo "For more information, see docs/USAGE.md"
echo ""
