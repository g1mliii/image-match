"""
GPU Memory Management Tests
Tests that GPU memory is properly managed and cleaned up
"""

import pytest
import torch
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_processing_clip import (
    extract_clip_embedding,
    batch_extract_clip_embeddings,
    detect_device,
    TORCH_AVAILABLE
)


def create_test_image(width=512, height=512):
    """Create a test image"""
    import tempfile
    import numpy as np
    from PIL import Image
    
    # Create random image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to temp file
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    img.save(path)
    
    return path


@pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                    reason="GPU not available")
class TestGPUMemoryManagement:
    """Test GPU memory management"""
    
    def test_memory_cleanup_after_batch(self):
        """Test that GPU memory is cleaned up after batch processing"""
        print("\n[GPU Memory] Testing memory cleanup after batch processing...")
        
        device = detect_device()
        if device != 'cuda':
            pytest.skip("GPU not available")
        
        # Pre-load model to get baseline with model in memory
        dummy_img = create_test_image()
        try:
            extract_clip_embedding(dummy_img)
        finally:
            if os.path.exists(dummy_img):
                os.remove(dummy_img)
        
        # Get baseline memory (with model loaded)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        baseline_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        
        print(f"  Baseline memory (model loaded): {baseline_allocated:.1f} MB allocated, {baseline_reserved:.1f} MB reserved")
        
        # Process batch
        images = [create_test_image() for _ in range(50)]
        
        try:
            results = batch_extract_clip_embeddings(images, batch_size=32, skip_errors=True)
            
            # Force cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Check memory after cleanup
            after_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            after_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            
            print(f"  After processing: {after_allocated:.1f} MB allocated, {after_reserved:.1f} MB reserved")
            
            # Memory should not grow significantly beyond baseline
            memory_growth = after_allocated - baseline_allocated
            print(f"  Memory growth: {memory_growth:.1f} MB")
            
            # Allow some growth for temporary buffers, but not excessive
            # Model is ~600MB, so growth should be minimal (< 50MB)
            assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.1f} MB"
            
            print(f"✓ Memory properly managed (growth: {memory_growth:.1f} MB)")
            
        finally:
            # Cleanup test images
            for img in images:
                if os.path.exists(img):
                    os.remove(img)
    
    def test_memory_cleanup_after_single_images(self):
        """Test that memory doesn't leak when processing many single images"""
        print("\n[GPU Memory] Testing memory cleanup for sequential single images...")
        
        device = detect_device()
        if device != 'cuda':
            pytest.skip("GPU not available")
        
        # Get initial memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"  Initial memory: {initial_allocated:.1f} MB")
        
        # Process many single images
        num_images = 100
        images = []
        
        try:
            for i in range(num_images):
                img = create_test_image()
                images.append(img)
                extract_clip_embedding(img)
                
                # Check memory every 25 images
                if (i + 1) % 25 == 0:
                    current_allocated = torch.cuda.memory_allocated() / 1024**2
                    print(f"  After {i+1} images: {current_allocated:.1f} MB")
            
            # Force cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Check final memory
            final_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_growth = final_allocated - initial_allocated
            
            print(f"  Final memory: {final_allocated:.1f} MB")
            print(f"  Total growth: {memory_growth:.1f} MB")
            
            # Memory should not grow excessively
            assert memory_growth < 100, f"Memory leak detected: {memory_growth:.1f} MB growth"
            
            print(f"✓ No memory leak detected (growth: {memory_growth:.1f} MB)")
            
        finally:
            # Cleanup test images
            for img in images:
                if os.path.exists(img):
                    os.remove(img)
    
    def test_memory_peak_usage(self):
        """Test peak memory usage during batch processing"""
        print("\n[GPU Memory] Testing peak memory usage...")
        
        device = detect_device()
        if device != 'cuda':
            pytest.skip("GPU not available")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Process large batch
        images = [create_test_image() for _ in range(100)]
        
        try:
            results = batch_extract_clip_embeddings(images, batch_size=32, skip_errors=True)
            
            # Get peak memory
            peak_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
            
            print(f"  Peak allocated: {peak_allocated:.1f} MB")
            print(f"  Peak reserved: {peak_reserved:.1f} MB")
            
            # Peak memory should be reasonable (< 2GB for this workload)
            assert peak_allocated < 2048, f"Peak memory too high: {peak_allocated:.1f} MB"
            
            print(f"✓ Peak memory within limits")
            
        finally:
            # Cleanup test images
            for img in images:
                if os.path.exists(img):
                    os.remove(img)
    
    def test_memory_cleanup_on_error(self):
        """Test that memory is cleaned up even when errors occur"""
        print("\n[GPU Memory] Testing memory cleanup on error...")
        
        device = detect_device()
        if device != 'cuda':
            pytest.skip("GPU not available")
        
        # Get initial memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        
        print(f"  Initial memory: {initial_allocated:.1f} MB")
        
        # Create mix of valid and invalid images
        images = []
        for i in range(20):
            if i % 5 == 0:
                # Create invalid image path
                images.append("/nonexistent/image.jpg")
            else:
                images.append(create_test_image())
        
        try:
            # Process with skip_errors=True
            results = batch_extract_clip_embeddings(images, batch_size=10, skip_errors=True)
            
            # Force cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Check memory
            final_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_growth = final_allocated - initial_allocated
            
            print(f"  Final memory: {final_allocated:.1f} MB")
            print(f"  Memory growth: {memory_growth:.1f} MB")
            
            # Memory should not leak even with errors
            assert memory_growth < 100, f"Memory leak on error: {memory_growth:.1f} MB"
            
            print(f"✓ Memory properly cleaned up despite errors")
            
        finally:
            # Cleanup test images
            for img in images:
                if os.path.exists(img) and img != "/nonexistent/image.jpg":
                    os.remove(img)


@pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                    reason="GPU not available")
class TestGPUMemoryInfo:
    """Test GPU memory information reporting"""
    
    def test_get_gpu_memory_info(self):
        """Test getting GPU memory information"""
        print("\n[GPU Memory] Getting GPU memory info...")
        
        device = detect_device()
        if device != 'cuda':
            pytest.skip("GPU not available")
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        free = total_memory - allocated
        
        print(f"  Total VRAM: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Free: {free:.2f} GB ({free/total_memory*100:.1f}%)")
        
        # Verify we have reasonable memory available
        assert free > 0.5, f"Very low GPU memory: {free:.2f} GB free"
        
        print(f"✓ GPU memory info retrieved successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
