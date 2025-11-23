#!/bin/bash
# Startup script for Product Matching System
# Automatically detects GPU and uses correct Python version

echo "========================================"
echo "Product Matching System - Starting..."
echo "========================================"
echo ""

# Check if we're on macOS with Apple Silicon
if [[ $(uname) == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
    echo "Apple Silicon detected - using default Python"
    cd backend
    python3 app.py
    exit 0
fi

# Check if Python 3.12 is available (required for AMD ROCm on Windows/Linux)
if command -v python3.12 &> /dev/null; then
    echo "Python 3.12 detected - checking for AMD GPU..."
    
    # Check if AMD GPU is present
    if python3.12 -c "import torch; exit(0 if torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0).upper() else 1)" 2>/dev/null; then
        echo "AMD GPU detected! Using Python 3.12 for ROCm support"
        echo ""
        cd backend
        python3.12 app.py
        exit 0
    fi
fi

# Fall back to default Python
echo "Using default Python"
echo ""
cd backend
python3 app.py
