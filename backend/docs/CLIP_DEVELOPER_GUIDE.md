# CLIP Developer Guide

## Architecture Overview

CatalogMatch uses CLIP (Contrastive Language-Image Pre-training) for visual product matching. This document explains the technical implementation, integration points, and migration path from legacy features.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flask Backend (app.py)                   │
│  - API endpoints for product upload, matching, GPU status    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Feature Extraction Service                        │
│  - feature_extraction_service.py (orchestration)             │
│  - Delegates to CLIP or legacy based on availability         │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  image_processing_clip.py│  │  image_processing.py     │
│  - CLIP embeddings       │  │  - Legacy features       │
│  - GPU acceleration      │  │  - Color histograms      │
│  - Model caching         │  │  - Shape descriptors     │
│  - 512-dim vectors       │  │  - Texture patterns      │
└──────────────────────────┘  └──────────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SQLite Database                           │
│  - products table (metadata)                                 │
│  - features table (BLOB storage for embeddings/features)     │
│  - matches table (similarity scores)                         │
└─────────────────────────────────────────────────────────────┘
```

## CLIP Integration

### Module: `image_processing_clip.py`

**Key Functions:**

```python
def extract_clip_embedding(image_path: str) -> np.ndarray:
    """
    Extract 512-dimensional CLIP embedding from image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy array of shape (512,) with float32 dtype
        
    Raises:
        ImageProcessingError: If image cannot be processed
    """
    
def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute device.
    
    Returns:
        {
            'device': 'cuda' | 'rocm' | 'mps' | 'cpu',
            'gpu_name': str | None,
            'vram_gb': float | None,
            'torch_version': str
        }
    """
    
def is_clip_available() -> bool:
    """Check if CLIP model is available and working."""
    
def get_clip_model() -> SentenceTransformer:
    """
    Get cached CLIP model instance (singleton pattern).
    Model is loaded once and reused for all subsequent calls.
    """
```

### Embedding Storage

CLIP embeddings are stored in the `features` table as BLOBs:

```sql
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    color_features BLOB NOT NULL,  -- For CLIP: stores 512-dim embedding
    shape_features BLOB NOT NULL,  -- For CLIP: empty/null
    texture_features BLOB NOT NULL, -- For CLIP: empty/null
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

**Storage Format:**
- CLIP: `color_features` contains 512-dim float32 array (2048 bytes)
- Legacy: `color_features` (256 bytes) + `shape_features` (28 bytes) + `texture_features` (256 bytes)

**Serialization:**
```python
# Store embedding
embedding_blob = embedding.astype(np.float32).tobytes()

# Load embedding
embedding = np.frombuffer(blob, dtype=np.float32)
```

## GPU Acceleration

### Device Detection

```python
import torch

# Check for NVIDIA CUDA
cuda_available = torch.cuda.is_available()

# Check for Apple MPS
mps_available = torch.backends.mps.is_available()

# Check for AMD ROCm (Windows)
# ROCm appears as CUDA on Windows
rocm_available = cuda_available and 'AMD' in torch.cuda.get_device_name(0)

# Priority: CUDA/ROCm > MPS > CPU
if cuda_available:
    device = 'cuda'
elif mps_available:
    device = 'mps'
else:
    device = 'cpu'
```

### Batch Processing

CLIP supports efficient batch processing on GPU:

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

# Load images
images = [Image.open(path) for path in image_paths]

# Batch encode (GPU automatically used if available)
embeddings = model.encode(
    images,
    batch_size=32,  # Adjust based on VRAM
    show_progress_bar=True,
    convert_to_numpy=True
)
```

**Batch Size Guidelines:**
- 4GB VRAM: batch_size=16
- 8GB VRAM: batch_size=32
- 12GB+ VRAM: batch_size=64

### Memory Management

```python
import torch

# Clear GPU cache after batch processing
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Monitor VRAM usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    print(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

## Similarity Computation

### CLIP Similarity (Cosine Similarity)

```python
def compute_clip_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two CLIP embeddings.
    
    Returns:
        Similarity score in range [0, 100]
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # Convert to 0-100 scale
    # Cosine similarity is in [-1, 1], but CLIP embeddings are typically in [0, 1]
    score = max(0, min(100, similarity * 100))
    
    return score
```

### Batch Similarity (Vectorized)

```python
def compute_batch_similarities(query_embedding: np.ndarray, 
                               catalog_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute similarities between one query and multiple catalog items.
    
    Args:
        query_embedding: Shape (512,)
        catalog_embeddings: Shape (N, 512)
        
    Returns:
        Similarity scores: Shape (N,) in range [0, 100]
    """
    # Normalize
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    catalog_norms = catalog_embeddings / np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    
    # Dot product (vectorized)
    similarities = np.dot(catalog_norms, query_norm)
    
    # Scale to 0-100
    scores = np.clip(similarities * 100, 0, 100)
    
    return scores
```

## Migration from Legacy Features

### Backward Compatibility

The system maintains backward compatibility with legacy features:

```python
def extract_features(image_path: str, use_clip: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract features using CLIP or legacy methods.
    
    Args:
        image_path: Path to image
        use_clip: Whether to use CLIP (default: True)
        
    Returns:
        {
            'color_features': np.ndarray,  # 512-dim for CLIP, 256-dim for legacy
            'shape_features': np.ndarray,  # Empty for CLIP, 7-dim for legacy
            'texture_features': np.ndarray # Empty for CLIP, 256-dim for legacy
        }
    """
    if use_clip and is_clip_available():
        embedding = extract_clip_embedding(image_path)
        return {
            'color_features': embedding,
            'shape_features': np.array([]),
            'texture_features': np.array([])
        }
    else:
        # Fall back to legacy features
        return extract_legacy_features(image_path)
```

### Database Migration

No database migration is required! CLIP embeddings are stored in the same `features` table:

```python
def detect_feature_type(color_features: np.ndarray) -> str:
    """
    Detect whether features are CLIP or legacy based on dimensions.
    
    Returns:
        'clip' if 512-dim, 'legacy' if 256-dim
    """
    if len(color_features) == 512:
        return 'clip'
    elif len(color_features) == 256:
        return 'legacy'
    else:
        raise ValueError(f"Unknown feature dimension: {len(color_features)}")
```

### Mixing CLIP and Legacy

The system can handle catalogs with mixed feature types:

```python
def compute_similarity_mixed(features1: Dict, features2: Dict) -> float:
    """
    Compute similarity between products with potentially different feature types.
    
    If both use CLIP: Use cosine similarity
    If both use legacy: Use weighted combination
    If mixed: Convert legacy to CLIP space (not recommended) or return low score
    """
    type1 = detect_feature_type(features1['color_features'])
    type2 = detect_feature_type(features2['color_features'])
    
    if type1 == 'clip' and type2 == 'clip':
        return compute_clip_similarity(
            features1['color_features'],
            features2['color_features']
        )
    elif type1 == 'legacy' and type2 == 'legacy':
        return compute_legacy_similarity(features1, features2)
    else:
        # Mixed types - not directly comparable
        logger.warning("Comparing CLIP and legacy features - results may be inaccurate")
        return 0.0  # Or implement conversion logic
```

## Performance Optimization

### Model Caching

```python
# Global model cache (singleton)
_clip_model = None

def get_clip_model() -> SentenceTransformer:
    """Get cached CLIP model instance."""
    global _clip_model
    
    if _clip_model is None:
        _clip_model = SentenceTransformer('clip-ViT-B-32')
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _clip_model = _clip_model.to('cuda')
        elif torch.backends.mps.is_available():
            _clip_model = _clip_model.to('mps')
    
    return _clip_model
```

### Feature Caching

Features are cached in the database to avoid recomputation:

```python
def get_or_extract_features(product_id: int, image_path: str) -> np.ndarray:
    """
    Get features from cache or extract if not cached.
    """
    # Check database cache
    features = get_features_from_db(product_id)
    
    if features is not None:
        return features
    
    # Extract and cache
    embedding = extract_clip_embedding(image_path)
    store_features_in_db(product_id, embedding)
    
    return embedding
```

### Batch Processing Pipeline

```python
def process_batch(image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """
    Process multiple images in batches for maximum GPU efficiency.
    """
    model = get_clip_model()
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load images
        images = [Image.open(path) for path in batch_paths]
        
        # Batch encode
        batch_embeddings = model.encode(
            images,
            batch_size=len(images),
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        embeddings.extend(batch_embeddings)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    return embeddings
```

## Testing

### Unit Tests

```python
def test_clip_embedding_shape():
    """Test that CLIP embeddings have correct shape."""
    embedding = extract_clip_embedding('test_image.jpg')
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32

def test_clip_similarity_range():
    """Test that similarity scores are in valid range."""
    emb1 = extract_clip_embedding('image1.jpg')
    emb2 = extract_clip_embedding('image2.jpg')
    
    similarity = compute_clip_similarity(emb1, emb2)
    assert 0 <= similarity <= 100

def test_gpu_detection():
    """Test GPU detection logic."""
    device_info = get_device_info()
    assert device_info['device'] in ['cuda', 'rocm', 'mps', 'cpu']
```

### Integration Tests

```python
def test_end_to_end_matching():
    """Test complete matching pipeline with CLIP."""
    # Upload historical products
    hist_ids = upload_products(['hist1.jpg', 'hist2.jpg'])
    
    # Upload new product
    new_id = upload_product('new.jpg')
    
    # Match
    matches = match_product(new_id, threshold=50, limit=10)
    
    # Verify
    assert len(matches) > 0
    assert all(0 <= m['similarity_score'] <= 100 for m in matches)
```

### Performance Tests

```python
def test_batch_processing_speed():
    """Test batch processing throughput."""
    import time
    
    image_paths = ['test_images/img{}.jpg'.format(i) for i in range(100)]
    
    start = time.time()
    embeddings = process_batch(image_paths, batch_size=32)
    elapsed = time.time() - start
    
    throughput = len(image_paths) / elapsed
    print(f"Throughput: {throughput:.1f} images/sec")
    
    # Verify reasonable performance
    device = get_device_info()['device']
    if device == 'cuda':
        assert throughput > 50  # Should be fast on GPU
    else:
        assert throughput > 5   # Should be reasonable on CPU
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size
embeddings = model.encode(images, batch_size=16)  # Instead of 32

# Or clear cache more frequently
torch.cuda.empty_cache()
```

**2. Model Download Fails**
```python
# Solution: Manual download
from sentence_transformers import SentenceTransformer

# This will download to ~/.cache/torch/sentence_transformers/
model = SentenceTransformer('clip-ViT-B-32')
```

**3. AMD GPU Not Detected**
```python
# Check ROCm installation
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show AMD GPU

# If False, reinstall PyTorch with ROCm:
# pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

**4. Slow Performance on GPU**
```python
# Check if model is actually on GPU
model = get_clip_model()
print(model.device)  # Should be 'cuda' or 'mps', not 'cpu'

# Verify batch processing is used
# Single image processing is slow even on GPU
```

## API Endpoints

### GPU Status

```
GET /api/gpu/status

Response:
{
    "available": true,
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "vram": 12.0,
    "throughput": "150-300",
    "first_run": false,
    "clip_available": true,
    "error": null
}
```

### CLIP Info

```
GET /api/clip/info

Response:
{
    "available": true,
    "model_name": "clip-ViT-B-32",
    "embedding_dim": 512,
    "device": "cuda",
    "cache_size_mb": 350
}
```

### Feature Extraction

```
POST /api/products/upload
Content-Type: multipart/form-data

Form Data:
- image: file
- category: string (optional)
- sku: string (optional)
- name: string (optional)
- is_historical: boolean (optional)

Response:
{
    "product_id": 123,
    "status": "success",
    "features_extracted": true,
    "feature_type": "clip"
}
```

## Configuration

### Environment Variables

```bash
# Force CPU mode (disable GPU)
export CLIP_FORCE_CPU=1

# Set CLIP model
export CLIP_MODEL=clip-ViT-B-32

# Set batch size
export CLIP_BATCH_SIZE=32

# Enable debug logging
export CLIP_DEBUG=1
```

### Config File

```python
# config.py
CLIP_CONFIG = {
    'model_name': 'clip-ViT-B-32',
    'batch_size': 32,
    'force_cpu': False,
    'cache_dir': '~/.cache/torch/sentence_transformers',
    'device_priority': ['cuda', 'mps', 'cpu']
}
```

## Future Enhancements

### Planned Features

1. **Multi-modal Matching**: Combine CLIP with text descriptions
2. **Fine-tuning**: Fine-tune CLIP on product-specific data
3. **Model Selection**: Allow users to choose between CLIP models
4. **Quantization**: Use INT8 quantization for faster inference
5. **ONNX Export**: Export model to ONNX for cross-platform deployment

### Research Directions

1. **Hybrid Matching**: Combine CLIP with legacy features for best of both worlds
2. **Active Learning**: Use user feedback to improve matching
3. **Explainability**: Visualize which image regions contribute to similarity
4. **Cross-domain Transfer**: Pre-train on general products, fine-tune on specific domains

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Apple MPS Documentation](https://developer.apple.com/metal/pytorch/)
