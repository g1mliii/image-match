# Intel GPU Acceleration Setup

This guide helps you enable Intel GPU acceleration for faster image processing on Intel integrated and discrete GPUs.

## Supported Hardware

✅ **Intel Arc GPUs** (A-Series discrete GPUs)
✅ **Intel Iris Xe Graphics** (11th gen+ integrated)
✅ **Intel UHD Graphics** (11th gen+ integrated)
❌ Older Intel HD Graphics (pre-11th gen) - use CPU mode

## Performance Comparison

| Device | Speed | Notes |
|--------|-------|-------|
| **AMD/NVIDIA GPU** | 150-300 img/s | Best performance |
| **Intel GPU (with extension)** | 30-80 img/s | 3-5x faster than CPU |
| **CPU only** | 5-20 img/s | Slowest but works everywhere |

## Installation

### Step 1: Install Intel Extension for PyTorch

```bash
pip install intel-extension-for-pytorch
```

### Step 2: Restart the Server

```bash
python backend/app.py
```

### Step 3: Verify Intel GPU is Detected

Check the server logs for:
```
INFO:image_processing_clip:Intel GPU detected (1 device(s))
INFO:image_processing_clip:Intel GPU: Intel(R) Iris(R) Xe Graphics
INFO:image_processing_clip:CLIP model optimized for Intel GPU on xpu:0
```

## Troubleshooting

### Intel GPU Not Detected

**Check if extension is installed:**
```bash
python -c "import intel_extension_for_pytorch as ipex; print('Intel Extension:', ipex.__version__)"
```

**If not installed:**
```bash
pip install intel-extension-for-pytorch
```

### Still Using CPU

**Possible reasons:**
1. Intel GPU is too old (pre-11th gen)
2. Intel drivers not installed
3. Extension not compatible with your GPU

**Solution:** The system will automatically fall back to CPU. Everything still works, just slower.

## Automatic Fallback

The system automatically handles Intel GPU detection:

1. **Intel Extension installed + Compatible GPU** → Uses Intel GPU (fast)
2. **Intel Extension not installed** → Uses CPU (slow but works)
3. **Intel GPU fails** → Falls back to CPU (reliable)

No configuration needed - it just works! 🚀

## Uninstall

To remove Intel GPU support:

```bash
pip uninstall intel-extension-for-pytorch
```

The system will automatically fall back to CPU mode.
