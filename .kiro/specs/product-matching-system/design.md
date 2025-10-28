# Design Document

## Overview

The Product Matching System is a Windows desktop application that enables users to compare new products against a historical catalog using image-based similarity analysis. The system extracts visual features from product images (color histograms, shape descriptors, texture patterns) and computes similarity scores to rank historical products by their visual resemblance to new products. The application is built as a standalone executable with a graphical user interface designed for non-technical users.

### Technology Stack

- **Frontend/GUI**: Electron framework for cross-platform desktop application with web technologies
- **UI Framework**: React for component-based interface development
- **Backend/Processing**: Python for image processing and similarity computation
- **Image Processing**: OpenCV and scikit-image for feature extraction
- **Similarity Computation**: NumPy and SciPy for vector operations and distance calculations
- **Database**: SQLite for local storage of product catalog and metadata
- **Packaging**: Electron Builder for creating Windows executable

### Key Design Decisions

1. **Electron + Python Architecture**: Use Electron for the GUI layer and Python for image processing, communicating via local REST API or IPC
2. **Feature-Based Matching**: Combine multiple visual features (color, shape, texture) with weighted scoring rather than deep learning to ensure fast processing and interpretability
3. **Local-First Storage**: Use SQLite for all data storage to ensure the application works offline without external dependencies
4. **Incremental Processing**: Process and cache image features when products are added to avoid recomputation during matching

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Electron Main Process                    │
│  - Application lifecycle management                          │
│  - Window management                                         │
│  - IPC coordination                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────────┐                 ┌──────────────────────┐
│  Renderer Process    │                 │   Python Backend     │
│  (React UI)          │◄───────────────►│   (Flask Server)     │
│                      │      HTTP        │                      │
│  - Product Upload    │                 │  - Feature Extract   │
│  - Results Display   │                 │  - Similarity Calc   │
│  - Catalog Mgmt      │                 │  - Image Processing  │
└──────────────────────┘                 └──────────────────────┘
                                                    │
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │   SQLite Database    │
                                         │                      │
                                         │  - Products Table    │
                                         │  - Features Table    │
                                         │  - Matches Table     │
                                         └──────────────────────┘
```

### Component Interaction Flow

1. User uploads new product via React UI
2. Electron renderer sends image to Python backend via HTTP
3. Python backend extracts visual features and stores in database
4. Python backend computes similarity against historical products
5. Results returned to React UI for display
6. User views ranked matches and detailed comparisons

## Components and Interfaces

### 1. Electron Main Process

**Responsibilities:**
- Launch and manage application windows
- Start Python backend server on application startup
- Handle file system operations (image storage)
- Manage application lifecycle and cleanup

**Key APIs:**
- `app.on('ready')`: Initialize Python backend and create main window
- `ipcMain.handle()`: Handle IPC requests from renderer process
- `dialog.showOpenDialog()`: File picker for image uploads

### 2. React UI Components

#### ProductUploadComponent
**Responsibilities:**
- Provide drag-and-drop interface for image uploads
- Collect product category and metadata
- Display upload progress and validation errors

**Props/State:**
- `selectedFiles`: Array of uploaded image files
- `category`: Selected product category
- `uploadProgress`: Upload progress percentage
- `validationErrors`: Array of error messages

**Methods:**
- `handleFileDrop(files)`: Process dropped files
- `handleCategorySelect(category)`: Set product category
- `submitProduct()`: Send product to backend for processing

#### MatchResultsComponent
**Responsibilities:**
- Display ranked list of similar products
- Show similarity scores and thumbnails
- Provide filtering and threshold controls

**Props/State:**
- `matches`: Array of match results with scores
- `threshold`: Current similarity threshold value
- `sortOrder`: Ranking sort order

**Methods:**
- `filterByThreshold(threshold)`: Apply threshold filter
- `selectMatch(matchId)`: Navigate to detailed comparison
- `exportResults()`: Export matches to CSV

#### DetailedComparisonComponent
**Responsibilities:**
- Show side-by-side image comparison
- Display feature-level similarity breakdown
- Show historical product metadata

**Props/State:**
- `newProduct`: New product data and image
- `matchedProduct`: Historical product data and image
- `similarityBreakdown`: Object with color, shape, texture scores

**Methods:**
- `navigateToNextMatch()`: Move to next match result
- `navigateToPreviousMatch()`: Move to previous match result

#### CatalogManagementComponent
**Responsibilities:**
- Display historical product catalog
- Allow adding new historical products
- Provide search and filter capabilities

**Props/State:**
- `products`: Array of historical products
- `searchQuery`: Current search text
- `selectedCategory`: Category filter

**Methods:**
- `addHistoricalProduct(product)`: Add product to catalog
- `searchProducts(query)`: Filter products by search
- `filterByCategory(category)`: Filter by category

### 3. Python Backend (Flask Server)

#### Image Processing Service

**Responsibilities:**
- Extract visual features from product images
- Normalize and preprocess images
- Cache extracted features

**Key Functions:**

```python
def extract_color_features(image_path: str) -> np.ndarray:
    """
    Extract color histogram features from image.
    Returns: 256-dimensional feature vector (HSV color histogram)
    """
    pass

def extract_shape_features(image_path: str) -> np.ndarray:
    """
    Extract shape descriptors using Hu moments and contour analysis.
    Returns: 7-dimensional feature vector
    """
    pass

def extract_texture_features(image_path: str) -> np.ndarray:
    """
    Extract texture features using Local Binary Patterns (LBP).
    Returns: 256-dimensional feature vector
    """
    pass

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Normalize image size, remove background, enhance contrast.
    Returns: Preprocessed image array
    """
    pass
```

#### Similarity Computation Service

**Responsibilities:**
- Compute similarity scores between feature vectors
- Rank matches by combined similarity score
- Apply category filtering and thresholds

**Key Functions:**

```python
def compute_color_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute color similarity using histogram intersection.
    Returns: Similarity score 0-100
    """
    pass

def compute_shape_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute shape similarity using Euclidean distance on Hu moments.
    Returns: Similarity score 0-100
    """
    pass

def compute_texture_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute texture similarity using chi-square distance on LBP histograms.
    Returns: Similarity score 0-100
    """
    pass

def compute_combined_similarity(color_sim: float, shape_sim: float, texture_sim: float) -> float:
    """
    Combine individual similarity scores with weights.
    Default weights: color=0.5, shape=0.3, texture=0.2
    Returns: Combined similarity score 0-100
    """
    pass

def find_matches(new_product_id: int, category: str, threshold: float, limit: int) -> List[Dict]:
    """
    Find and rank similar products in the same category.
    Returns: List of match results with scores and metadata
    """
    pass
```

#### REST API Endpoints

```python
POST /api/products/upload
# Upload new product with image and category
# Request: multipart/form-data with image file and category
# Response: { "product_id": int, "status": "success" }

POST /api/products/match
# Find matches for a new product
# Request: { "product_id": int, "threshold": float, "limit": int }
# Response: { "matches": [...], "total_count": int }

GET /api/products/historical
# Get historical product catalog
# Query params: category, search, limit, offset
# Response: { "products": [...], "total_count": int }

POST /api/products/historical
# Add new historical product
# Request: multipart/form-data with image, category, metadata
# Response: { "product_id": int, "status": "success" }

GET /api/products/{product_id}
# Get product details including features
# Response: { "product": {...}, "features": {...} }

POST /api/batch/match
# Batch match multiple products
# Request: { "product_ids": [int], "threshold": float }
# Response: { "results": [...], "summary": {...} }
```

### 4. Database Schema

#### Products Table
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    category TEXT NOT NULL,
    product_name TEXT,
    sku TEXT,
    is_historical BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON string for additional fields
);
```

#### Features Table
```sql
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    color_features BLOB NOT NULL,  -- Serialized numpy array
    shape_features BLOB NOT NULL,
    texture_features BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

#### Matches Table
```sql
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    new_product_id INTEGER NOT NULL,
    matched_product_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    color_score REAL NOT NULL,
    shape_score REAL NOT NULL,
    texture_score REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (new_product_id) REFERENCES products(id),
    FOREIGN KEY (matched_product_id) REFERENCES products(id)
);
```

## Data Models

### Product Model
```typescript
interface Product {
  id: number;
  imagePath: string;
  category: string;
  productName?: string;
  sku?: string;
  isHistorical: boolean;
  createdAt: Date;
  metadata?: Record<string, any>;
}
```

### MatchResult Model
```typescript
interface MatchResult {
  matchedProduct: Product;
  similarityScore: number;
  colorScore: number;
  shapeScore: number;
  textureScore: number;
  isPotentialDuplicate: boolean;  // true if score > 90
}
```

### FeatureVector Model
```typescript
interface FeatureVector {
  productId: number;
  colorFeatures: number[];  // 256-dim HSV histogram
  shapeFeatures: number[];  // 7-dim Hu moments
  textureFeatures: number[]; // 256-dim LBP histogram
}
```

### BatchMatchRequest Model
```typescript
interface BatchMatchRequest {
  products: Array<{
    imagePath: string;
    category: string;
    productName?: string;
  }>;
  threshold: number;
  limit: number;
}
```

### BatchMatchResult Model
```typescript
interface BatchMatchResult {
  productId: number;
  productName: string;
  category: string;
  matchCount: number;
  topMatches: MatchResult[];
  processingTime: number;
}
```

## Error Handling

### Frontend Error Handling

1. **File Upload Errors**
   - Invalid file format: Display error message with supported formats
   - File too large: Display error with size limit
   - Network error: Retry with exponential backoff, show retry button

2. **Validation Errors**
   - Missing category: Highlight required field
   - Invalid threshold: Show valid range (0-100)

3. **Backend Connection Errors**
   - Backend not responding: Display "Service unavailable" message
   - Timeout: Show timeout message with retry option

### Backend Error Handling

1. **Image Processing Errors**
   - Corrupted image: Return 400 error with message
   - Unsupported format: Return 400 error with supported formats
   - Processing failure: Log error, return 500 with generic message

2. **Database Errors**
   - Connection failure: Retry 3 times, return 503 if failed
   - Constraint violation: Return 400 with specific error
   - Query timeout: Return 504 with timeout message

3. **Resource Errors**
   - Disk space full: Return 507 error
   - Memory limit exceeded: Implement pagination, return 413

### Error Response Format
```typescript
interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: any;
  };
}
```

## Testing Strategy

### Unit Tests

1. **Image Processing Functions**
   - Test feature extraction with known images
   - Verify feature vector dimensions
   - Test preprocessing normalization
   - Test edge cases (grayscale, transparent backgrounds)

2. **Similarity Computation**
   - Test similarity functions with known feature vectors
   - Verify score ranges (0-100)
   - Test combined similarity weighting
   - Test threshold filtering logic

3. **Database Operations**
   - Test CRUD operations for products
   - Test feature storage and retrieval
   - Test query performance with large datasets
   - Test transaction rollback on errors

### Integration Tests

1. **API Endpoint Tests**
   - Test product upload flow end-to-end
   - Test matching with various thresholds
   - Test batch processing
   - Test error responses

2. **UI Component Tests**
   - Test file upload component with drag-and-drop
   - Test results display with mock data
   - Test navigation between views
   - Test threshold slider updates

### Performance Tests

1. **Matching Performance**
   - Benchmark matching against 1,000 products
   - Verify completion within 30 seconds
   - Test with various image sizes
   - Profile memory usage during batch operations

2. **Database Performance**
   - Test query performance with 10,000 products
   - Verify index effectiveness
   - Test concurrent read/write operations

### User Acceptance Testing

1. **Usability Testing**
   - Test with non-technical users
   - Verify drag-and-drop intuitiveness
   - Verify error messages are clear
   - Test workflow completion time

2. **Accuracy Testing**
   - Manually verify match quality with known similar products
   - Test with various product types
   - Verify category filtering works correctly
   - Test duplicate detection threshold

## Performance Considerations

### Image Processing Optimization

1. **Caching Strategy**
   - Cache extracted features in database
   - Only recompute if image changes
   - Use in-memory cache for frequently accessed features

2. **Parallel Processing**
   - Process batch uploads in parallel using multiprocessing
   - Limit concurrent processes to CPU count
   - Use thread pool for I/O operations

3. **Image Preprocessing**
   - Resize images to standard dimensions (e.g., 512x512) before processing
   - Use efficient image loading (lazy loading, streaming)
   - Compress stored images to reduce disk usage

### Database Optimization

1. **Indexing**
   - Create index on `products.category` for fast filtering
   - Create index on `products.is_historical` for catalog queries
   - Create composite index on `matches(new_product_id, similarity_score)`

2. **Query Optimization**
   - Use prepared statements for repeated queries
   - Implement pagination for large result sets
   - Use EXPLAIN to analyze slow queries

3. **Connection Management**
   - Use connection pooling
   - Set appropriate timeout values
   - Implement connection retry logic

### UI Performance

1. **Rendering Optimization**
   - Use virtual scrolling for large result lists
   - Lazy load images in results view
   - Debounce search and filter inputs

2. **State Management**
   - Use React.memo for expensive components
   - Implement proper key props for list rendering
   - Avoid unnecessary re-renders

## Security Considerations

1. **File Upload Security**
   - Validate file types using magic numbers, not just extensions
   - Scan uploaded files for malware
   - Limit file sizes to prevent DoS
   - Store uploaded files outside web root

2. **Input Validation**
   - Sanitize all user inputs
   - Validate category names against whitelist
   - Validate numeric inputs (threshold, limits)

3. **Database Security**
   - Use parameterized queries to prevent SQL injection
   - Implement proper error handling to avoid information leakage
   - Regular database backups

4. **Application Security**
   - Run Python backend on localhost only
   - Use random port for backend server
   - Implement request rate limiting
   - Validate all API requests

## Deployment and Packaging

### Build Process

1. **Frontend Build**
   - Bundle React application with Webpack
   - Minify and optimize assets
   - Include all dependencies

2. **Backend Packaging**
   - Package Python application with PyInstaller
   - Include all Python dependencies
   - Bundle OpenCV and image processing libraries

3. **Electron Packaging**
   - Use Electron Builder for Windows executable
   - Include both frontend and backend bundles
   - Configure auto-update mechanism (future enhancement)

### Installation

1. **Single Executable**
   - Create installer with Electron Builder
   - No additional software required
   - Automatic database initialization on first run

2. **Data Storage**
   - Store database in user's AppData folder
   - Store uploaded images in AppData/images
   - Create necessary directories on first launch

### Updates and Maintenance

1. **Version Management**
   - Implement semantic versioning
   - Include version number in UI
   - Log version in database for migration tracking

2. **Database Migrations**
   - Implement migration system for schema changes
   - Automatic migration on application startup
   - Backup database before migrations

## Future Enhancements

1. **Performance Prediction Integration**
   - Add performance data fields to historical products
   - Implement prediction models based on similar products
   - Display predicted performance ranges in match results

2. **Advanced Matching Features**
   - Deep learning-based image embeddings (ResNet, EfficientNet)
   - Text-based matching using product descriptions
   - Custom feature weight configuration per category

3. **Collaboration Features**
   - Multi-user support with user accounts
   - Shared product catalogs
   - Match result annotations and notes

4. **Analytics and Reporting**
   - Match accuracy tracking over time
   - Category-specific performance metrics
   - Export detailed reports
