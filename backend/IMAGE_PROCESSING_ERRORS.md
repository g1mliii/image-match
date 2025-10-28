# Image Processing Error Handling Guide

This document describes the error handling system for image processing and feature extraction in the Product Matching System.

## Overview

The image processing module provides comprehensive error handling for real-world scenarios where images may be corrupted, in unsupported formats, or otherwise invalid. All errors are structured to provide:

1. **Clear error messages** - Human-readable descriptions of what went wrong
2. **Error codes** - Machine-readable codes for programmatic handling
3. **Actionable suggestions** - Guidance on how to fix the issue

## Error Types

### 1. InvalidImageFormatError

**Error Code:** `INVALID_FORMAT`

**When it occurs:**
- Image file format is not JPEG, PNG, or WebP
- File extension doesn't match actual file content
- File size exceeds 10MB limit

**Example scenarios:**
- User uploads a PDF or text file with .jpg extension
- User uploads a BMP or TIFF image
- User uploads an image larger than 10MB

**Default suggestion:**
> "Please upload a valid JPEG, PNG, or WebP image file."

**API Response Example:**
```json
{
  "error": "Unsupported image format: BMP. Supported formats: JPEG, PNG, WebP",
  "error_code": "INVALID_FORMAT",
  "suggestion": "Please upload a valid JPEG, PNG, or WebP image file."
}
```

### 2. CorruptedImageError

**Error Code:** `CORRUPTED_IMAGE`

**When it occurs:**
- Image file is corrupted or truncated
- Image file cannot be decoded by OpenCV or PIL
- Image data is incomplete or malformed

**Example scenarios:**
- Download was interrupted, resulting in incomplete file
- File was corrupted during transfer
- Image was saved incorrectly by another application

**Default suggestion:**
> "The image file appears to be corrupted. Please try re-saving or re-exporting the image."

**API Response Example:**
```json
{
  "error": "Failed to read image with both OpenCV and PIL: truncated file",
  "error_code": "CORRUPTED_IMAGE",
  "suggestion": "The image file appears to be corrupted. Please try re-saving or re-exporting the image."
}
```

### 3. ImageTooSmallError

**Error Code:** `IMAGE_TOO_SMALL`

**When it occurs:**
- Image dimensions are less than 50x50 pixels

**Example scenarios:**
- User uploads a thumbnail or icon
- User uploads a heavily cropped image
- Image was resized too aggressively

**Default suggestion:**
> "Image must be at least 50x50 pixels. Please upload a higher resolution image."

**API Response Example:**
```json
{
  "error": "Image too small (30x30). Minimum size is 50x50 pixels.",
  "error_code": "IMAGE_TOO_SMALL",
  "suggestion": "Image must be at least 50x50 pixels. Please upload a higher resolution image."
}
```

### 4. ImageProcessingFailedError

**Error Code:** `PROCESSING_FAILED`

**When it occurs:**
- Feature extraction fails for unknown reasons
- Unexpected error during image processing
- System resource issues (memory, disk space)

**Example scenarios:**
- Out of memory during processing
- Disk full when saving features
- Unexpected image characteristics that break processing

**Default suggestion:**
> "Failed to process image. Please try a different image or contact support."

**API Response Example:**
```json
{
  "error": "Failed to extract color features: out of memory",
  "error_code": "PROCESSING_FAILED",
  "suggestion": "Failed to process image. Please try a different image or contact support."
}
```

## Additional Error Codes

### FILE_NOT_FOUND
- File path doesn't exist
- File was deleted before processing

### EMPTY_FILE
- File exists but has 0 bytes
- File was not written correctly

### FILE_TOO_LARGE
- File exceeds 10MB limit
- Suggestion: "Please reduce the file size to under 10MB."

### UNSUPPORTED_MODE
- Image color mode is not supported (e.g., CMYK)
- Rare edge case with unusual image formats

## Using Error Information in UI

### Display Error Messages

```javascript
// Example React component error handling
try {
  const response = await uploadImage(file);
} catch (error) {
  if (error.response?.data) {
    const { error: message, error_code, suggestion } = error.response.data;
    
    // Show user-friendly message
    showErrorNotification({
      title: getErrorTitle(error_code),
      message: message,
      suggestion: suggestion,
      severity: getErrorSeverity(error_code)
    });
  }
}

function getErrorTitle(errorCode) {
  const titles = {
    'INVALID_FORMAT': 'Unsupported File Format',
    'CORRUPTED_IMAGE': 'Corrupted Image File',
    'IMAGE_TOO_SMALL': 'Image Too Small',
    'FILE_TOO_LARGE': 'File Too Large',
    'PROCESSING_FAILED': 'Processing Failed'
  };
  return titles[errorCode] || 'Upload Error';
}

function getErrorSeverity(errorCode) {
  // User-fixable errors are 'warning', system errors are 'error'
  const userFixable = ['INVALID_FORMAT', 'IMAGE_TOO_SMALL', 'FILE_TOO_LARGE'];
  return userFixable.includes(errorCode) ? 'warning' : 'error';
}
```

### Provide Retry Options

```javascript
// Show retry button for user-fixable errors
if (['INVALID_FORMAT', 'IMAGE_TOO_SMALL', 'FILE_TOO_LARGE', 'CORRUPTED_IMAGE'].includes(error_code)) {
  showRetryButton({
    text: 'Try Another Image',
    onClick: () => openFileDialog()
  });
}
```

### Show Image Preview

```javascript
// Preview image before upload to help users verify
function handleFileSelect(file) {
  // Basic client-side validation
  if (file.size > 10 * 1024 * 1024) {
    showError('File too large. Maximum size is 10MB.');
    return;
  }
  
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showError('Unsupported format. Please use JPEG, PNG, or WebP.');
    return;
  }
  
  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      if (img.width < 50 || img.height < 50) {
        showError('Image too small. Minimum size is 50x50 pixels.');
        return;
      }
      showImagePreview(e.target.result);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}
```

## Backend API Integration

### Flask Endpoint Example

```python
from flask import jsonify
from image_processing import extract_all_features, ImageProcessingError

@app.route('/api/products/upload', methods=['POST'])
def upload_product():
    try:
        # ... file handling code ...
        
        # Extract features
        features = extract_all_features(image_path)
        
        # ... store in database ...
        
        return jsonify({
            'success': True,
            'product_id': product_id
        })
    
    except ImageProcessingError as e:
        # Return structured error response
        return jsonify(e.to_dict()), 400
    
    except Exception as e:
        # Unexpected errors
        return jsonify({
            'error': 'Internal server error',
            'error_code': 'INTERNAL_ERROR',
            'suggestion': 'Please try again or contact support.'
        }), 500
```

### Batch Processing Error Handling

```python
from feature_cache import batch_extract_features

# Process multiple products
results = batch_extract_features(product_ids)

# Separate successes and failures
successes = [pid for pid, r in results.items() if r['success']]
failures = [
    {
        'product_id': pid,
        'error': r['error'],
        'error_code': r['error_code'],
        'suggestion': r.get('suggestion')
    }
    for pid, r in results.items()
    if not r['success']
]

return jsonify({
    'total': len(product_ids),
    'successful': len(successes),
    'failed': len(failures),
    'failures': failures
})
```

## Testing Error Scenarios

### Unit Tests

See `test_image_processing.py` for comprehensive error handling tests:
- Invalid file formats
- Corrupted images
- Empty files
- Images too small
- Non-existent files

### Integration Tests

See `test_feature_cache.py` for caching and batch processing error tests.

### Manual Testing Checklist

- [ ] Upload valid JPEG, PNG, WebP images
- [ ] Upload unsupported formats (BMP, TIFF, GIF)
- [ ] Upload non-image files (PDF, TXT, DOCX)
- [ ] Upload corrupted images (truncated files)
- [ ] Upload very small images (< 50x50)
- [ ] Upload very large images (> 10MB)
- [ ] Upload images with unusual color modes (CMYK, grayscale, RGBA)
- [ ] Test with images from different sources (camera, scanner, web)

## Best Practices

1. **Always validate on client-side first** - Catch obvious errors before upload
2. **Show clear error messages** - Use the provided suggestions
3. **Provide retry options** - Let users try again with different files
4. **Log errors for debugging** - Include error codes in logs
5. **Monitor error rates** - Track which errors occur most frequently
6. **Update suggestions based on user feedback** - Improve error messages over time

## Common User Issues and Solutions

| User Report | Likely Error | Solution |
|------------|--------------|----------|
| "Upload doesn't work" | INVALID_FORMAT | Check file format, ensure JPEG/PNG/WebP |
| "Image looks fine but fails" | CORRUPTED_IMAGE | Re-save image in photo editor |
| "Small product photo rejected" | IMAGE_TOO_SMALL | Use higher resolution image |
| "Can't upload from phone" | FILE_TOO_LARGE | Compress image or reduce resolution |
| "Scanned image fails" | UNSUPPORTED_MODE | Convert to RGB color mode |

## Future Enhancements

- [ ] Support for additional formats (TIFF, BMP)
- [ ] Automatic image repair for minor corruption
- [ ] Automatic resizing for oversized images
- [ ] Color mode conversion (CMYK → RGB)
- [ ] Progressive upload with chunking for large files
- [ ] Image quality analysis and warnings
