import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from typing import Tuple, Optional, Dict
from PIL import Image
import io


class ImageProcessingError(Exception):
    """Base exception for image processing errors"""
    def __init__(self, message: str, error_code: str, suggestion: str = None):
        self.message = message
        self.error_code = error_code
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert error to dictionary for API responses"""
        return {
            'error': self.message,
            'error_code': self.error_code,
            'suggestion': self.suggestion
        }


class InvalidImageFormatError(ImageProcessingError):
    """Raised when image format is not supported or corrupted"""
    def __init__(self, message: str, suggestion: str = None):
        super().__init__(
            message, 
            'INVALID_FORMAT',
            suggestion or 'Please upload a valid JPEG, PNG, or WebP image file.'
        )


class CorruptedImageError(ImageProcessingError):
    """Raised when image file is corrupted or cannot be decoded"""
    def __init__(self, message: str, suggestion: str = None):
        super().__init__(
            message,
            'CORRUPTED_IMAGE',
            suggestion or 'The image file appears to be corrupted. Please try re-saving or re-exporting the image.'
        )


class ImageTooSmallError(ImageProcessingError):
    """Raised when image dimensions are too small for processing"""
    def __init__(self, message: str, suggestion: str = None):
        super().__init__(
            message,
            'IMAGE_TOO_SMALL',
            suggestion or 'Image must be at least 50x50 pixels. Please upload a higher resolution image.'
        )


class ImageProcessingFailedError(ImageProcessingError):
    """Raised when feature extraction fails"""
    def __init__(self, message: str, suggestion: str = None):
        super().__init__(
            message,
            'PROCESSING_FAILED',
            suggestion or 'Failed to process image. Please try a different image or contact support.'
        )


def validate_image_file(image_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate image file before processing.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (is_valid, error_message, error_code)
    """
    # Check if file exists
    if not os.path.exists(image_path):
        return False, f"File not found: {image_path}", "FILE_NOT_FOUND"
    
    # Check file size
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        return False, "Image file is empty (0 bytes)", "EMPTY_FILE"
    # No maximum file size limit - handle large images gracefully
    
    # Validate file format using PIL
    try:
        with Image.open(image_path) as img:
            detected_format = img.format
            supported_formats = ['JPEG', 'PNG', 'WEBP']
            
            if detected_format not in supported_formats:
                return False, f"Unsupported image format: {detected_format}. Supported formats: JPEG, PNG, WebP", "UNSUPPORTED_FORMAT"
    except Exception as e:
        return False, f"Cannot determine image format: {str(e)}", "UNSUPPORTED_FORMAT"
    
    # Try to open with PIL to verify it's not corrupted
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        
        # Re-open to check dimensions (verify() closes the file)
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 50 or height < 50:
                return False, f"Image too small ({width}x{height}). Minimum size is 50x50 pixels.", "IMAGE_TOO_SMALL"
            
            # Check if image has valid mode
            if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                return False, f"Unsupported image mode: {img.mode}", "UNSUPPORTED_MODE"
    
    except (IOError, SyntaxError) as e:
        return False, f"Corrupted or invalid image file: {str(e)}", "CORRUPTED_IMAGE"
    except Exception as e:
        return False, f"Failed to validate image: {str(e)}", "VALIDATION_FAILED"
    
    return True, None, None


def safe_imread(image_path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Safely read image with comprehensive error handling.
    
    Args:
        image_path: Path to image file
        flags: OpenCV imread flags
    
    Returns:
        Image as numpy array
    
    Raises:
        InvalidImageFormatError: If format is not supported
        CorruptedImageError: If image is corrupted
        ImageTooSmallError: If image is too small
    """
    # Validate file first
    is_valid, error_msg, error_code = validate_image_file(image_path)
    if not is_valid:
        if error_code in ['UNSUPPORTED_FORMAT', 'FILE_NOT_FOUND']:
            raise InvalidImageFormatError(error_msg)
        elif error_code in ['CORRUPTED_IMAGE', 'EMPTY_FILE', 'VALIDATION_FAILED']:
            raise CorruptedImageError(error_msg)
        elif error_code == 'IMAGE_TOO_SMALL':
            raise ImageTooSmallError(error_msg)
        elif error_code == 'FILE_TOO_LARGE':
            raise InvalidImageFormatError(error_msg, 'Please reduce the file size to under 10MB.')
        else:
            raise ImageProcessingFailedError(error_msg)
    
    # Try to read with OpenCV
    try:
        img = cv2.imread(image_path, flags)
        
        if img is None:
            # OpenCV failed, try with PIL as fallback
            try:
                pil_img = Image.open(image_path)
                
                # Convert to RGB if needed
                if pil_img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])  # Use alpha channel as mask
                    pil_img = background
                elif pil_img.mode == 'P':
                    pil_img = pil_img.convert('RGB')
                elif pil_img.mode == 'L' and flags == cv2.IMREAD_COLOR:
                    pil_img = pil_img.convert('RGB')
                
                # Convert PIL to OpenCV format
                img = np.array(pil_img)
                
                # Convert RGB to BGR for OpenCV
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            except Exception as e:
                raise CorruptedImageError(f"Failed to read image with both OpenCV and PIL: {str(e)}")
        
        # Final validation
        if img.size == 0:
            raise CorruptedImageError("Image has no data after loading")
        
        return img
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError):
        raise
    except Exception as e:
        raise ImageProcessingFailedError(f"Unexpected error reading image: {str(e)}")


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Normalize image size, remove background, and enhance contrast.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Preprocessed image array (512x512, BGR format)
    
    Raises:
        ImageProcessingError subclasses for various error conditions
    """
    try:
        # Read image with error handling
        img = safe_imread(image_path, cv2.IMREAD_COLOR)
        
        # Resize to standard dimensions (512x512)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        except Exception as e:
            # If contrast enhancement fails, continue with original resized image
            print(f"Warning: Contrast enhancement failed: {e}")
        
        return img
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError):
        raise
    except Exception as e:
        raise ImageProcessingFailedError(f"Failed to preprocess image: {str(e)}")


def extract_color_features(image_path: str) -> np.ndarray:
    """
    Extract color histogram features from image using HSV color space.
    
    Args:
        image_path: Path to image file
    
    Returns:
        256-dimensional feature vector (HSV color histogram)
    
    Raises:
        ImageProcessingError subclasses for various error conditions
    """
    try:
        # Read and preprocess image
        img = safe_imread(image_path, cv2.IMREAD_COLOR)
        
        # Resize for consistent feature extraction
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate 3D histogram (H: 8 bins, S: 8 bins, V: 4 bins = 256 total)
        hist = cv2.calcHist(
            [hsv], 
            [0, 1, 2],  # H, S, V channels
            None, 
            [8, 8, 4],  # Number of bins per channel
            [0, 180, 0, 256, 0, 256]  # Ranges for H, S, V
        )
        
        # Flatten and normalize histogram
        hist = hist.flatten().astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalize to sum to 1
        
        # Ensure exactly 256 dimensions
        if len(hist) != 256:
            if len(hist) < 256:
                hist = np.pad(hist, (0, 256 - len(hist)), mode='constant')
            else:
                hist = hist[:256]
        
        # Ensure float32 for consistency
        hist = hist.astype(np.float32)
        
        return hist
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError):
        raise
    except Exception as e:
        raise ImageProcessingFailedError(f"Failed to extract color features: {str(e)}")


def extract_shape_features(image_path: str) -> np.ndarray:
    """
    Extract shape descriptors using Hu moments.
    
    Args:
        image_path: Path to image file
    
    Returns:
        7-dimensional feature vector (Hu moments)
    
    Raises:
        ImageProcessingError subclasses for various error conditions
    """
    try:
        # Read image in grayscale
        img = safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize for consistent feature extraction
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Apply binary threshold to get shape silhouette
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate moments
        moments = cv2.moments(binary)
        
        # Calculate Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform for better scale invariance and numerical stability
        # Handle zero and negative values
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # Ensure float32 for consistency
        hu_moments = hu_moments.astype(np.float32)
        
        # Verify we have exactly 7 dimensions
        assert len(hu_moments) == 7, f"Expected 7 Hu moments, got {len(hu_moments)}"
        
        return hu_moments
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError):
        raise
    except Exception as e:
        raise ImageProcessingFailedError(f"Failed to extract shape features: {str(e)}")


def extract_texture_features(image_path: str) -> np.ndarray:
    """
    Extract texture features using Local Binary Patterns (LBP).
    
    Args:
        image_path: Path to image file
    
    Returns:
        256-dimensional feature vector (LBP histogram)
    
    Raises:
        ImageProcessingError subclasses for various error conditions
    """
    try:
        # Read image in grayscale
        img = safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize for consistent feature extraction
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Compute LBP
        radius = 3
        n_points = 8 * radius  # 24 points
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        
        # Compute histogram with 256 bins
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        # Normalize histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Avoid division by zero
        
        # Verify we have exactly 256 dimensions
        assert len(hist) == 256, f"Expected 256-dimensional histogram, got {len(hist)}"
        
        return hist
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError):
        raise
    except Exception as e:
        raise ImageProcessingFailedError(f"Failed to extract texture features: {str(e)}")


def extract_all_features(image_path: str) -> Dict[str, np.ndarray]:
    """
    Extract all features (color, shape, texture) from an image.
    
    This is a convenience function that extracts all features at once
    and provides comprehensive error handling.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary with keys 'color_features', 'shape_features', 'texture_features'
    
    Raises:
        ImageProcessingError subclasses for various error conditions
    """
    try:
        color_features = extract_color_features(image_path)
        shape_features = extract_shape_features(image_path)
        texture_features = extract_texture_features(image_path)
        
        return {
            'color_features': color_features,
            'shape_features': shape_features,
            'texture_features': texture_features
        }
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError, ImageProcessingFailedError):
        raise
    except Exception as e:
        raise ImageProcessingFailedError(f"Failed to extract features: {str(e)}")
