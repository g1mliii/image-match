# GPU Acceleration

This folder contains all GPU-related setup and testing tools.

## Quick Start

```bash
cd gpu
python setup_gpu.py
```

## Files

- **`setup_gpu.py`** - Automated GPU setup (AMD/NVIDIA/Apple Silicon)
- **`check_gpu.py`** - Quick GPU detection check
- **`benchmark_gpu.py`** - Performance benchmark
- **`verify_setup.py`** - Complete verification suite
- **`GPU_SETUP_GUIDE.md`** - Comprehensive setup guide
- **`requirements_gpu.txt`** - GPU-specific dependencies

## Documentation

See **[GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md)** for complete instructions.

## Quick Commands

```bash
# Setup GPU
python setup_gpu.py

# Check GPU
python check_gpu.py

# Benchmark
python benchmark_gpu.py

# Verify everything
python verify_setup.py
```

## Supported GPUs

- **AMD Radeon RX 6000/7000/9000** (ROCm 6.4, Python 3.12)
- **NVIDIA GeForce/RTX** (CUDA 12.4, Python 3.8+)
- **Apple Silicon M1/M2/M3/M4/M5** (MPS, Python 3.8+)
- **CPU Fallback** (No GPU required)

## Performance

| GPU | Throughput | Rating |
|-----|-----------|--------|
| AMD RX 9070 XT | 193 img/s | Excellent |
| NVIDIA RTX 4090 | 250-300 img/s | Excellent |
| Apple M3 Max | 100-150 img/s | Very Good |
