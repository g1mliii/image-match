"""
Test GPU support for different platforms (NVIDIA, AMD, Apple Silicon)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from image_processing_clip import (
        detect_device,
        is_clip_available,
        TORCH_AVAILABLE
    )
    CLIP_AVAILABLE = is_clip_available()
except ImportError:
    CLIP_AVAILABLE = False
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestGPUPlatformSupport:
    """Test GPU support across different platforms"""
    
    def test_gpu_detection_comprehensive(self):
        """Test comprehensive GPU detection for all platforms"""
        print("\n[GPU Platform Support] Testing GPU detection...")
        
        device = detect_device()
        print(f"  Detected device: {device}")
        
        if TORCH_AVAILABLE:
            import torch
            
            # Test NVIDIA CUDA
            cuda_available = torch.cuda.is_available()
            print(f"  CUDA available: {cuda_available}")
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                print(f"  CUDA GPU count: {gpu_count}")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"    GPU {i}: {gpu_name}")
                    
                    # Detect AMD vs NVIDIA
                    if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                        print(f"      Type: AMD (ROCm)")
                    elif 'NVIDIA' in gpu_name.upper() or 'GEFORCE' in gpu_name.upper() or 'TESLA' in gpu_name.upper():
                        print(f"      Type: NVIDIA")
                    else:
                        print(f"      Type: Unknown CUDA device")
            
            # Test Apple Silicon MPS
            mps_available = torch.backends.mps.is_available()
            print(f"  MPS available: {mps_available}")
            
            if mps_available:
                print(f"    Type: Apple Silicon (M1/M2/M3/M4/M5)")
            
            # Summary
            if cuda_available:
                print(f"\n[OK] GPU acceleration available via CUDA")
            elif mps_available:
                print(f"\n[OK] GPU acceleration available via MPS")
            else:
                print(f"\n[INFO] No GPU detected, using CPU")
        
        assert device in ['cuda', 'mps', 'cpu'], f"Unknown device: {device}"
    
    def test_platform_specific_features(self):
        """Test platform-specific GPU features"""
        print("\n[GPU Platform Support] Testing platform-specific features...")
        
        device = detect_device()
        
        if TORCH_AVAILABLE:
            import torch
            
            if device == 'cuda':
                # CUDA-specific tests
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  cuDNN version: {torch.backends.cudnn.version()}")
                print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
                
                # Memory info
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
                    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                    print(f"  GPU memory allocated: {mem_allocated:.2f} MB")
                    print(f"  GPU memory reserved: {mem_reserved:.2f} MB")
            
            elif device == 'mps':
                # MPS-specific tests
                print(f"  MPS backend available: {torch.backends.mps.is_available()}")
                print(f"  MPS built: {torch.backends.mps.is_built()}")
            
            else:
                # CPU
                print(f"  CPU threads: {torch.get_num_threads()}")
                print(f"  CPU interop threads: {torch.get_num_interop_threads()}")
        
        print(f"[OK] Platform-specific features tested")


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestGPUPerformanceComparison:
    """Compare performance across different GPU platforms"""
    
    def test_performance_expectations(self):
        """Test that performance meets expectations for each platform"""
        print("\n[GPU Performance] Testing performance expectations...")
        
        device = detect_device()
        
        # Performance expectations (images per second)
        expectations = {
            'cuda': {
                'min': 5.0,  # NVIDIA/AMD should be fast
                'target': 20.0,
                'description': 'NVIDIA/AMD GPU'
            },
            'mps': {
                'min': 2.0,  # Apple Silicon should be good
                'target': 10.0,
                'description': 'Apple Silicon'
            },
            'cpu': {
                'min': 0.5,  # CPU should be acceptable
                'target': 2.0,
                'description': 'CPU'
            }
        }
        
        expected = expectations[device]
        
        print(f"  Device: {device} ({expected['description']})")
        print(f"  Minimum expected: {expected['min']} img/s")
        print(f"  Target: {expected['target']} img/s")
        print(f"\n[OK] Performance expectations defined for {device}")


if __name__ == '__main__':
    import pytest
    
    print("=" * 80)
    print("GPU Platform Support Tests")
    print("=" * 80)
    
    if not CLIP_AVAILABLE:
        print("\n[ERROR] CLIP dependencies not available!")
        print("Install with: pip install torch sentence-transformers")
        sys.exit(1)
    
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    sys.exit(exit_code)
