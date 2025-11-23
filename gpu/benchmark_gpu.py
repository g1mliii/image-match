"""
GPU Benchmark Script
Compares CPU vs GPU performance for CLIP embeddings
Tests AMD ROCm, NVIDIA CUDA, Apple MPS, and CPU fallback
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from image_processing_clip import (
    detect_device,
    extract_clip_embedding,
    batch_extract_clip_embeddings,
    get_clip_model
)
from tests.test_clip import create_test_image


def benchmark_device(device_name, num_images=100):
    """Benchmark CLIP performance on a specific device"""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {device_name}")
    print(f"{'='*80}")
    
    # Create test images
    print(f"\nCreating {num_images} test images...")
    images = [create_test_image(width=512, height=512) for _ in range(num_images)]
    
    try:
        # Single image benchmark
        print("\n[1/3] Single Image Processing")
        start = time.time()
        for i in range(10):
            extract_clip_embedding(images[i])
        single_time = (time.time() - start) / 10
        print(f"  Average time per image: {single_time:.3f}s")
        print(f"  Throughput: {1/single_time:.1f} images/sec")
        
        # Batch processing benchmark
        print("\n[2/3] Batch Processing (batch_size=32)")
        start = time.time()
        batch_extract_clip_embeddings(images, batch_size=32, skip_errors=True)
        batch_time = time.time() - start
        print(f"  Total time for {num_images} images: {batch_time:.2f}s")
        print(f"  Average time per image: {batch_time/num_images:.3f}s")
        print(f"  Throughput: {num_images/batch_time:.1f} images/sec")
        
        # Memory usage
        print("\n[3/3] Memory Usage")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory Allocated: {allocated:.2f} GB")
            print(f"  GPU Memory Reserved: {reserved:.2f} GB")
        else:
            print(f"  Running on CPU (no GPU memory tracking)")
        
        return {
            'device': device_name,
            'single_time': single_time,
            'single_throughput': 1/single_time,
            'batch_time': batch_time,
            'batch_throughput': num_images/batch_time,
            'speedup': single_time * num_images / batch_time
        }
        
    finally:
        # Cleanup
        import os
        for img in images:
            if os.path.exists(img):
                os.remove(img)


def main():
    """Run comprehensive GPU benchmark"""
    print("="*80)
    print("GPU Benchmark - CLIP Embeddings")
    print("="*80)
    
    # Detect current device
    device = detect_device()
    print(f"\nCurrent Device: {device}")
    
    # Get GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"ROCm Version: {torch.version.hip}")
        elif hasattr(torch.version, 'cuda'):
            print(f"CUDA Version: {torch.version.cuda}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"GPU: Apple Silicon (MPS)")
    else:
        print(f"GPU: None (CPU mode)")
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Load model once
    print("\nLoading CLIP model...")
    model, model_device = get_clip_model()
    print(f"Model loaded on: {model_device}")
    
    # Run benchmark
    results = benchmark_device(device, num_images=100)
    
    # Summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"\nDevice: {results['device']}")
    print(f"\nSingle Image:")
    print(f"  Time: {results['single_time']:.3f}s")
    print(f"  Throughput: {results['single_throughput']:.1f} images/sec")
    print(f"\nBatch Processing:")
    print(f"  Time per image: {results['batch_time']/100:.3f}s")
    print(f"  Throughput: {results['batch_throughput']:.1f} images/sec")
    print(f"  Speedup vs single: {results['speedup']:.1f}x")
    
    # Performance rating
    print(f"\n{'='*80}")
    print("PERFORMANCE RATING")
    print(f"{'='*80}")
    
    throughput = results['batch_throughput']
    if throughput > 100:
        rating = "🚀 EXCELLENT (High-end GPU)"
    elif throughput > 50:
        rating = "✅ VERY GOOD (Mid-range GPU)"
    elif throughput > 20:
        rating = "👍 GOOD (Entry-level GPU or fast CPU)"
    elif throughput > 5:
        rating = "⚠️  ACCEPTABLE (CPU mode)"
    else:
        rating = "❌ SLOW (Upgrade recommended)"
    
    print(f"\n{rating}")
    print(f"Throughput: {throughput:.1f} images/sec")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if device == 'cpu':
        print("\n⚠️  Running in CPU mode")
        print("   To enable GPU acceleration:")
        print("   - AMD GPU: Install Python 3.12 and run: python setup_gpu.py")
        print("   - NVIDIA GPU: Run: python setup_gpu.py")
        print("   - Apple Silicon: Already optimized (MPS)")
    elif device == 'cuda' and throughput < 50:
        print("\n⚠️  GPU performance lower than expected")
        print("   Possible causes:")
        print("   - ROCm optimization issues (AMD GPUs on Windows)")
        print("   - Outdated drivers")
        print("   - Thermal throttling")
        print("   - Background processes using GPU")
    else:
        print("\n✅ GPU acceleration is working well!")
        print(f"   Your {device.upper()} GPU is providing good performance.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
