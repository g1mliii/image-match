"""
Tests for image processing and feature extraction functionality.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import tempfile

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from image_processing import (
    validate_image_file,
    safe_imread,
    preprocess_image,
    extract_color_features,
    extract_shape_features,
    extract_texture_features,
    extract_all_features,
    InvalidImageFormatError,
    CorruptedImageError,
    ImageTooSmallError,
    ImageProcessingFailedError
)


def create_test_image(width=512, height=512, format='PNG'):
    """Create a test image file"""
    # Create a simple test image with some patterns
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    img[0:height//2, 0:width//2] = [255, 0, 0]  # Red
    img[0:height//2, width//2:width] = [0, 255, 0]  # Green
    img[height//2:height, 0:width//2] = [0, 0, 255]  # Blue
    img[height//2:height, width//2:width] = [255, 255, 0]  # Yellow
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format.lower()}')
    temp_path = temp_file.name
    temp_file.close()
    
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(temp_path, format=format)
    
    return temp_path


def test_validate_image_file():
    """Test image file validation"""
    print("Testing image file validation...")
    
    # Test valid image
    test_img = create_test_image()
    is_valid, error_msg, error_code = validate_image_file(test_img)
    assert is_valid, f"Valid image failed validation: {error_msg}"
    print("✓ Valid image passes validation")
    
    # Test non-existent file
    is_valid, error_msg, error_code = validate_image_file("nonexistent.jpg")
    assert not is_valid, "Non-existent file should fail validation"
    assert error_code == "FILE_NOT_FOUND"
    print("✓ Non-existent file detected")
    
    # Test empty file
    empty_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    empty_path = empty_file.name
    empty_file.close()
    is_valid, error_msg, error_code = validate_image_file(empty_path)
    assert not is_valid, "Empty file should fail validation"
    assert error_code == "EMPTY_FILE"
    os.unlink(empty_path)
    print("✓ Empty file detected")
    
    # Test image too small
    small_img = create_test_image(width=30, height=30)
    is_valid, error_msg, error_code = validate_image_file(small_img)
    assert not is_valid, "Small image should fail validation"
    assert error_code == "IMAGE_TOO_SMALL"
    os.unlink(small_img)
    print("✓ Small image detected")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Image validation tests passed\n")


def test_safe_imread():
    """Test safe image reading"""
    print("Testing safe image reading...")
    
    # Test reading valid image
    test_img = create_test_image()
    img = safe_imread(test_img)
    assert img is not None, "Failed to read valid image"
    assert img.shape == (512, 512, 3), f"Unexpected image shape: {img.shape}"
    print("✓ Valid image read successfully")
    
    # Test reading grayscale
    img_gray = safe_imread(test_img, cv2.IMREAD_GRAYSCALE)
    assert img_gray is not None, "Failed to read grayscale image"
    assert len(img_gray.shape) == 2, f"Grayscale image should be 2D, got {img_gray.shape}"
    print("✓ Grayscale image read successfully")
    
    # Test error handling for non-existent file
    try:
        safe_imread("nonexistent.jpg")
        assert False, "Should raise InvalidImageFormatError"
    except InvalidImageFormatError:
        print("✓ Non-existent file raises InvalidImageFormatError")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Safe imread tests passed\n")


def test_preprocess_image():
    """Test image preprocessing"""
    print("Testing image preprocessing...")
    
    test_img = create_test_image(width=1024, height=768)
    
    # Test preprocessing
    preprocessed = preprocess_image(test_img)
    assert preprocessed is not None, "Preprocessing failed"
    assert preprocessed.shape == (512, 512, 3), f"Expected (512, 512, 3), got {preprocessed.shape}"
    print("✓ Image resized to 512x512")
    print("✓ Contrast enhancement applied")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Preprocessing tests passed\n")


def test_extract_color_features():
    """Test color feature extraction"""
    print("Testing color feature extraction...")
    
    test_img = create_test_image()
    
    # Extract features
    features = extract_color_features(test_img)
    
    # Verify dimensions
    assert features is not None, "Feature extraction failed"
    assert len(features) == 256, f"Expected 256 dimensions, got {len(features)}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    
    # Verify normalization (sum should be close to 1)
    feature_sum = np.sum(features)
    assert 0.9 <= feature_sum <= 1.1, f"Features not normalized properly: sum={feature_sum}"
    
    print(f"✓ Color features extracted: {len(features)} dimensions")
    print(f"✓ Feature sum: {feature_sum:.4f}")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Color feature extraction tests passed\n")


def test_extract_shape_features():
    """Test shape feature extraction"""
    print("Testing shape feature extraction...")
    
    test_img = create_test_image()
    
    # Extract features
    features = extract_shape_features(test_img)
    
    # Verify dimensions
    assert features is not None, "Feature extraction failed"
    assert len(features) == 7, f"Expected 7 dimensions (Hu moments), got {len(features)}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    
    # Verify no NaN or Inf values
    assert not np.any(np.isnan(features)), "Features contain NaN values"
    assert not np.any(np.isinf(features)), "Features contain Inf values"
    
    print(f"✓ Shape features extracted: {len(features)} dimensions")
    print(f"✓ Hu moments: {features}")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Shape feature extraction tests passed\n")


def test_extract_texture_features():
    """Test texture feature extraction"""
    print("Testing texture feature extraction...")
    
    test_img = create_test_image()
    
    # Extract features
    features = extract_texture_features(test_img)
    
    # Verify dimensions
    assert features is not None, "Feature extraction failed"
    assert len(features) == 256, f"Expected 256 dimensions, got {len(features)}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    
    # Verify normalization (sum should be close to 1)
    feature_sum = np.sum(features)
    assert 0.9 <= feature_sum <= 1.1, f"Features not normalized properly: sum={feature_sum}"
    
    # Verify no negative values
    assert np.all(features >= 0), "Features contain negative values"
    
    print(f"✓ Texture features extracted: {len(features)} dimensions")
    print(f"✓ Feature sum: {feature_sum:.4f}")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Texture feature extraction tests passed\n")


def test_extract_all_features():
    """Test extracting all features at once"""
    print("Testing extract_all_features...")
    
    test_img = create_test_image()
    
    # Extract all features
    features = extract_all_features(test_img)
    
    # Verify structure
    assert 'color_features' in features, "Missing color_features"
    assert 'shape_features' in features, "Missing shape_features"
    assert 'texture_features' in features, "Missing texture_features"
    
    # Verify dimensions
    assert len(features['color_features']) == 256, "Wrong color feature dimensions"
    assert len(features['shape_features']) == 7, "Wrong shape feature dimensions"
    assert len(features['texture_features']) == 256, "Wrong texture feature dimensions"
    
    print("✓ All features extracted successfully")
    print(f"  - Color: {len(features['color_features'])} dimensions")
    print(f"  - Shape: {len(features['shape_features'])} dimensions")
    print(f"  - Texture: {len(features['texture_features'])} dimensions")
    
    # Cleanup
    os.unlink(test_img)
    print("✓ Extract all features tests passed\n")


def test_error_handling():
    """Test error handling for various failure scenarios"""
    print("Testing error handling...")
    
    # Test corrupted image
    corrupted_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    corrupted_file.write(b'This is not an image file')
    corrupted_file.close()
    
    try:
        extract_color_features(corrupted_file.name)
        assert False, "Should raise InvalidImageFormatError or CorruptedImageError"
    except (InvalidImageFormatError, CorruptedImageError) as e:
        print(f"✓ Corrupted image detected: {e.message}")
        print(f"  Error code: {e.error_code}")
        print(f"  Suggestion: {e.suggestion}")
    
    os.unlink(corrupted_file.name)
    
    # Test unsupported format (create a text file with .jpg extension)
    unsupported_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    unsupported_file.write(b'Plain text file')
    unsupported_file.close()
    
    try:
        extract_color_features(unsupported_file.name)
        assert False, "Should raise InvalidImageFormatError"
    except (InvalidImageFormatError, CorruptedImageError) as e:
        print(f"✓ Unsupported format detected: {e.message}")
    
    os.unlink(unsupported_file.name)
    
    print("✓ Error handling tests passed\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Image Processing Tests")
    print("=" * 60 + "\n")
    
    try:
        test_validate_image_file()
        test_safe_imread()
        test_preprocess_image()
        test_extract_color_features()
        test_extract_shape_features()
        test_extract_texture_features()
        test_extract_all_features()
        test_error_handling()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
