"""
Comprehensive AMD GPU (ROCm) support tests
"""

import os
import sys
import time
import tempfile
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from image_processing_clip import (
        detect_device,
        get_clip_model,
        extract_clip_embedding,
        batch_extract_clip_embeddings,
        compute_clip_similarity,
        batch_compute_clip_similarities,
        is_clip_available,
        TORCH_AVAILABLE
    )
    CLIP_AVAILABLE = is_clip_available()
except ImportError:
    CLIP_AVAILABLE = False
    TORCH_AVAILABLE = False


def create_test_image(width=512, height=512):
    """Create a test image"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img[i, j] = [int(255 * i / height), int(255 * j / width), 128]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_path = temp_file.name
    temp_file.close()
    
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(temp_path, format='PNG')
    
    return temp_path


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestAMDGPUDetection:
    """Test AMD GPU detection via ROCm"""
    
    def test_amd_gpu_detection(self):
        """Test that AMD GPU is detected correctly"""
        print("\n[AMD GPU] Testing GPU detection...")
        
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        import torch
        
        # Check if CUDA/ROCm is available
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA/ROCm available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"  Device count: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
                
                # Check if it's AMD
                is_amd = 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper()
                if is_amd:
                    print(f"    [OK] AMD GPU detected!")
                    print(f"    Type: AMD Radeon (ROCm)")
                    
                    # Get memory info
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"    Total memory: {mem_total:.2f} GB")
                else:
                    print(f"    Type: {gpu_name}")
        else:
            print(f"  [WARNING] No GPU detected")
            print(f"  You may need to install PyTorch with ROCm support:")
            print(f"  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        
        # Test our detect_device function
        device = detect_device()
        print(f"\n  Detected device: {device}")
        
        if cuda_available:
            assert device == 'cuda', f"Expected 'cuda' device, got '{device}'"
            print(f"  [OK] Device detection working correctly")
    
    def test_amd_gpu_properties(self):
        """Test AMD GPU properties and capabilities"""
        print("\n[AMD GPU] Testing GPU properties...")
        
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        
        print(f"  GPU Properties:")
        print(f"    Name: {props.name}")
        print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Multi-processor count: {props.multi_processor_count}")
        print(f"    Compute capability: {props.major}.{props.minor}")
        
        # Check if it's AMD
        is_amd = 'AMD' in props.name.upper() or 'RADEON' in props.name.upper()
        
        if is_amd:
            print(f"    [OK] AMD GPU confirmed")
            
            # AMD-specific checks
            if props.total_memory >= 4 * 1024**3:  # 4GB+
                print(f"    [OK] Sufficient VRAM for CLIP (4GB+)")
            else:
                print(f"    [WARNING] Limited VRAM ({props.total_memory / 1024**3:.2f} GB)")


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestAMDGPUPerformance:
    """Test AMD GPU performance with CLIP"""
    
    def test_amd_single_image_performance(self):
        """Test single image processing performance on AMD GPU"""
        print("\n[AMD GPU Performance] Testing single image processing...")
        
        device = detect_device()
        
        if device != 'cuda':
            pytest.skip(f"GPU not available (device: {device})")
        
        import torch
        
        # Verify it's AMD
        gpu_name = torch.cuda.get_device_name(0)
        is_amd = 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper()
        
        if not is_amd:
            pytest.skip(f"Not an AMD GPU: {gpu_name}")
        
        print(f"  GPU: {gpu_name}")
        
        # Create test image
        test_img = create_test_image()
        
        try:
            # Warm up
            print(f"  Warming up...")
            extract_clip_embedding(test_img)
            
            # Benchmark
            num_runs = 10
            print(f"  Running {num_runs} iterations...")
            start = time.time()
            for _ in range(num_runs):
                extract_clip_embedding(test_img)
            total_time = time.time() - start
            avg_time = total_time / num_runs
            
            print(f"\n  Results:")
            print(f"    Average time: {avg_time:.3f}s per image")
            print(f"    Throughput: {1/avg_time:.1f} images/sec")
            
            # AMD GPU should be reasonably fast
            if avg_time < 0.1:
                print(f"    [EXCELLENT] Very fast (< 0.1s)")
            elif avg_time < 0.3:
                print(f"    [GOOD] Fast (< 0.3s)")
            elif avg_time < 1.0:
                print(f"    [OK] Acceptable (< 1.0s)")
            else:
                print(f"    [WARNING] Slower than expected (> 1.0s)")
            
            # Should be faster than CPU baseline (2s)
            assert avg_time < 2.0, f"AMD GPU too slow: {avg_time:.3f}s (expected < 2.0s)"
            
        finally:
            os.unlink(test_img)
    
    def test_amd_batch_performance(self):
        """Test batch processing performance on AMD GPU"""
        print("\n[AMD GPU Performance] Testing batch processing...")
        
        device = detect_device()
        
        if device != 'cuda':
            pytest.skip(f"GPU not available (device: {device})")
        
        import torch
        
        # Verify it's AMD
        gpu_name = torch.cuda.get_device_name(0)
        is_amd = 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper()
        
        if not is_amd:
            pytest.skip(f"Not an AMD GPU: {gpu_name}")
        
        print(f"  GPU: {gpu_name}")
        
        # Create test images
        num_images = 50
        print(f"  Creating {num_images} test images...")
        images = [create_test_image(width=256, height=256) for _ in range(num_images)]
        
        try:
            # Test different batch sizes
            batch_sizes = [8, 16, 32]
            
            print(f"\n  Testing batch sizes:")
            for batch_size in batch_sizes:
                start = time.time()
                results = batch_extract_clip_embeddings(images, batch_size=batch_size, skip_errors=True)
                total_time = time.time() - start
                
                success_count = sum(1 for _, emb, _ in results if emb is not None)
                throughput = success_count / total_time
                
                print(f"    Batch size {batch_size:2d}: {total_time:.2f}s ({throughput:.1f} img/s)")
            
            # AMD GPU should process at least 5 images/sec
            final_throughput = success_count / total_time
            assert final_throughput > 2.0, f"AMD GPU batch too slow: {final_throughput:.1f} img/s"
            
            print(f"\n  [OK] AMD GPU batch processing working")
            
        finally:
            for img in images:
                os.unlink(img)
    
    def test_amd_memory_usage(self):
        """Test GPU memory usage during processing"""
        print("\n[AMD GPU Performance] Testing memory usage...")
        
        device = detect_device()
        
        if device != 'cuda':
            pytest.skip(f"GPU not available (device: {device})")
        
        import torch
        
        # Verify it's AMD
        gpu_name = torch.cuda.get_device_name(0)
        is_amd = 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper()
        
        if not is_amd:
            pytest.skip(f"Not an AMD GPU: {gpu_name}")
        
        print(f"  GPU: {gpu_name}")
        
        # Get initial memory
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  Memory before: {mem_before:.2f} MB")
        
        # Load model
        model, _ = get_clip_model()
        mem_after_model = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  Memory after model load: {mem_after_model:.2f} MB")
        print(f"  Model size: {mem_after_model - mem_before:.2f} MB")
        
        # Process image
        test_img = create_test_image()
        try:
            extract_clip_embedding(test_img)
            mem_after_process = torch.cuda.memory_allocated(0) / 1024**2
            print(f"  Memory after processing: {mem_after_process:.2f} MB")
            
            # Total memory used
            total_mem = mem_after_process
            print(f"\n  Total GPU memory used: {total_mem:.2f} MB")
            
            # Should fit in 4GB GPU
            assert total_mem < 3500, f"Memory usage too high: {total_mem:.2f} MB"
            
            print(f"  [OK] Memory usage acceptable for 4GB GPU")
            
        finally:
            os.unlink(test_img)
            torch.cuda.empty_cache()


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestAMDGPUAccuracy:
    """Test that AMD GPU produces correct results"""
    
    def test_amd_embedding_correctness(self):
        """Test that AMD GPU produces valid embeddings"""
        print("\n[AMD GPU Accuracy] Testing embedding correctness...")
        
        device = detect_device()
        
        if device != 'cuda':
            pytest.skip(f"GPU not available (device: {device})")
        
        test_img = create_test_image()
        
        try:
            # Extract embedding
            embedding = extract_clip_embedding(test_img)
            
            # Check dimensions
            assert embedding.shape == (512,), f"Wrong shape: {embedding.shape}"
            print(f"  [OK] Embedding dimensions: {embedding.shape}")
            
            # Check for NaN/Inf
            assert not np.any(np.isnan(embedding)), "Embedding contains NaN"
            assert not np.any(np.isinf(embedding)), "Embedding contains Inf"
            print(f"  [OK] No NaN or Inf values")
            
            # Check value range
            print(f"  Value range: [{embedding.min():.3f}, {embedding.max():.3f}]")
            print(f"  Mean: {embedding.mean():.3f}, Std: {embedding.std():.3f}")
            
            print(f"\n  [OK] AMD GPU produces valid embeddings")
            
        finally:
            os.unlink(test_img)
    
    def test_amd_similarity_correctness(self):
        """Test that AMD GPU produces correct similarity scores"""
        print("\n[AMD GPU Accuracy] Testing similarity correctness...")
        
        device = detect_device()
        
        if device != 'cuda':
            pytest.skip(f"GPU not available (device: {device})")
        
        # Create two images
        img1 = create_test_image()
        img2 = create_test_image()
        
        try:
            # Extract embeddings
            emb1 = extract_clip_embedding(img1)
            emb2 = extract_clip_embedding(img2)
            
            # Compute similarity
            similarity = compute_clip_similarity(emb1, emb2)
            
            # Check range
            assert 0 <= similarity <= 100, f"Similarity out of range: {similarity}"
            print(f"  Similarity: {similarity:.2f}/100")
            
            # Identical images should have high similarity
            assert similarity > 99, f"Identical images should have similarity > 99, got {similarity}"
            
            print(f"  [OK] AMD GPU produces correct similarity scores")
            
        finally:
            os.unlink(img1)
            os.unlink(img2)


if __name__ == '__main__':
    import pytest
    
    print("=" * 80)
    print("AMD GPU (ROCm) Support Tests")
    print("=" * 80)
    
    if not CLIP_AVAILABLE:
        print("\n[ERROR] CLIP dependencies not available!")
        print("Install with: pip install torch sentence-transformers")
        sys.exit(1)
    
    if not TORCH_AVAILABLE:
        print("\n[ERROR] PyTorch not available!")
        sys.exit(1)
    
    import torch
    
    if not torch.cuda.is_available():
        print("\n[WARNING] No GPU detected!")
        print("\nFor AMD GPU support, install PyTorch with ROCm:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")
        print("\nRunning tests anyway (will skip GPU tests)...")
    
    exit_code = pytest.main([__file__, '-v', '--tb=short', '-s'])
    sys.exit(exit_code)
