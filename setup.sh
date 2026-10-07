#!/bin/bash

# 📦 Churn Prediction Project Setup Script (Dev Environment)
# This script sets up a Python virtual environment and installs all dev dependencies.

echo "Starting setup..."

# 1. Create virtual environment
echo "Creating virtual environment in ./venv..."
python3 -m venv venv || { echo "❌ Failed to create virtual environment"; exit 1; }

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install pinned dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Success message
echo "✅ Setup complete!"
echo "🔁 To activate this environment later, run: source venv/bin/activate"
