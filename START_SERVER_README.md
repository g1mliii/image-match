# Starting the Product Matching System

## Quick Start

### Windows
Double-click `start_server.bat` or run:
```bash
start_server.bat
```

### macOS/Linux
```bash
./start_server.sh
```

## What the Startup Script Does

The startup script automatically:
1. **Detects your GPU** (AMD, NVIDIA, or Apple Silicon)
2. **Selects the correct Python version**:
   - **AMD GPU**: Uses Python 3.12 (required for ROCm)
   - **NVIDIA/Apple Silicon**: Uses any available Python 3.8+
3. **Starts the Flask server** on http://127.0.0.1:5000

## Manual Start (Advanced)

If you prefer to start manually:

### AMD GPU (Requires Python 3.12)
```bash
cd backend
py -3.12 app.py
```

### NVIDIA GPU or Apple Silicon (Any Python 3.8+)
```bash
cd backend
python app.py
```

### CPU Mode (Any Python 3.8+)
```bash
cd backend
python app.py
```

## Why Python 3.12 for AMD?

AMD ROCm on Windows requires:
- **Python 3.12 exactly** (ROCm wheels only available for 3.12)
- **sentence-transformers < 3.0.0** (newer versions incompatible with ROCm)

If you use Python 3.13+ with AMD GPU, the system will fall back to CPU mode.

## Troubleshooting

### "Python 3.12 not found" (AMD GPU users)
Install Python 3.12 from https://www.python.org/downloads/

### "AMD GPU not detected"
1. Verify ROCm HIP SDK 6.4 is installed
2. Run: `py -3.12 gpu/check_gpu.py`
3. See `gpu/GPU_SETUP_GUIDE.md` for detailed setup

### Server won't start
1. Check if port 5000 is already in use
2. Try running manually: `cd backend && python app.py`
3. Check for error messages in the console

## First Run

On first run, the CLIP model (~350MB) will download automatically. This takes 1-2 minutes depending on your internet speed. Subsequent runs will be instant as the model is cached locally.

## Accessing the Application

Once started, open your browser to:
```
http://127.0.0.1:5000
```

You should see the GPU status indicator in the top right:
- ⚡ **AMD GPU Active** (black) - ROCm working
- ⚡ **NVIDIA GPU Active** (black) - CUDA working  
- ⚡ **Apple Silicon Active** (black) - MPS working
- 💻 **CPU Mode** (white) - No GPU detected

## Stopping the Server

Press `Ctrl+C` in the terminal window to stop the server.
