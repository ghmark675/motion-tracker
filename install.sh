#!/bin/bash
# Installation script for Motion Tracker

set -e  # Exit on error

echo "========================================"
echo "Motion Tracker Installation"
echo "========================================"

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if version is 3.10+
required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10 or higher is required"
    echo "Please install a newer version of Python"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed successfully"

# Install in development mode
echo ""
echo "Installing Motion Tracker in development mode..."
pip install -e .

# Run tests
echo ""
read -p "Run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running tests..."
    pytest tests/ -v
fi

# Success message
echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run demos:"
echo "  python demos/webcam_demo.py"
echo "  python demos/posture_correction_demo.py"
echo "  python demos/fitness_trainer_demo.py"
echo ""
echo "For more information, see:"
echo "  README.md - Full documentation"
echo "  QUICKSTART.md - Quick start guide"
echo ""
echo "Happy tracking! 🏃‍♂️💪🤸‍♀️"
echo "========================================"
