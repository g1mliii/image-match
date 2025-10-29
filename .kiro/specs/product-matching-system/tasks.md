# Implementation Plan

- [x] 1. Set up project structure and development environment (SIMPLIFIED)
  - Set up Python Flask backend project structure
  - Initialize SQLite database with schema
  - Create requirements.txt for Python dependencies (Flask, OpenCV, numpy, pywebview)
  - Create main.py for desktop launcher (pywebview)
  - No build tools, no npm, no TypeScript - pure Python + vanilla JS
  - _Requirements: 9.2_

- [x] 2. Implement database layer and models
  - Create SQLite database initialization script with products, features, and matches tables
  - Implement Python data access layer with CRUD operations for products
  - Implement feature storage and retrieval functions with numpy array serialization
  - Add database indexes for category and is_historical fields
  - _Requirements: 8.4, 3.1_

- [x] 3. Implement image preprocessing and feature extraction dont assume all images are vaild we may have unopenable images, random format for some reason again its real world data, have some error handilng in the ui to maybe tell user what was wrong and how to reupload image as well. we havent made ui yet but just plan ahead with this idea and maybe update the ui task with this as well

  - Create image preprocessing function to normalize size, remove background, and enhance contrast
  - Implement color feature extraction using HSV histograms (256-dimensional)
  - Implement shape feature extraction using Hu moments (7-dimensional)
  - Implement texture feature extraction using Local Binary Patterns (256-dimensional)
  - Add feature caching mechanism to avoid recomputation
  - _Requirements: 2.1, 2.2, 8.3_

- [x] 4. Implement similarity computation engine
  - Create color similarity function using histogram intersection
  - Create shape similarity function using Euclidean distance on Hu moments
  - Create texture similarity function using chi-square distance on LBP histograms
  - Implement combined similarity scoring with configurable weights (color=0.5, shape=0.3, texture=0.2)
  - Normalize all similarity scores to 0-100 range
  - _Requirements: 2.3, 2.4_

- [x] 5. Implement matching service with category filtering and real-world data handling
  - Create matching function that filters by category before computing similarities (handle NULL/missing categories)
  - Implement graceful handling for products with missing or corrupted features (skip and log errors)
  - Add fallback logic for products without category (use 'unknown' category or match against all)
  - Implement ranking logic to sort matches by similarity score
  - Add threshold filtering to return only matches above configured threshold
  - Implement result limiting to return top N matches
  - Add duplicate detection flag for matches with score > 90
  - Handle edge cases: empty historical catalog, all products filtered out, all similarity computations fail
  - Return detailed error information for failed matches (error codes, suggestions)
  - Log warnings for data quality issues (missing metadata, corrupted features, etc.)
  - _Requirements: 3.1, 3.2, 4.1, 5.1, 5.2, 5.3_

- [x] 6. Implement Flask REST API endpoints (simplified for folder workflow)
  - **POST /api/products/upload** - Upload single product image:
    - Validate image file (format, size, corruption)
    - Accept optional fields: category, product_name, sku, is_historical (all can be NULL)
    - Validate SKU format if provided
    - Extract features and store in database
    - Return product_id and feature extraction status
  - **POST /api/products/match** - Find matches for a product:
    - Validate product_id exists
    - Match against historical products (filter by category if provided)
    - Handle empty catalog and missing features gracefully
    - Return ranked matches with similarity scores
  - **GET /api/products/{id}** - Get product details:
    - Return product metadata (handles NULL fields)
    - Include feature extraction status
  - **GET /api/products/{id}/image** - Serve product image
  - Comprehensive error handling:
    - Structured error responses: {error, error_code, suggestion}
    - Handle image processing errors, missing features, database errors
    - Log all errors with context
  - _Requirements: 1.1, 1.2, 3.3, 4.4, 6.1, 6.4_

- [x] 7. Create lightweight desktop application launcher
  - Replaced Electron with pywebview for lightweight desktop wrapper
  - Created main.py launcher that starts Flask backend and opens webview window
  - No complex build process - just Python!
  - Application runs Flask server in background thread
  - Desktop window wraps web UI at http://127.0.0.1:5000
  - _Requirements: 9.2, 9.3_

- [x] 8. Create folder-based batch matching UI (REDESIGNED FOR ACTUAL WORKFLOW)
  - **Workflow**: Upload historical catalog folder → Upload new products folder → Match → View results
  - Replaced single product upload with folder-based batch processing
  - Created static/index.html with step-by-step wizard interface:
    - Step 1: Upload historical catalog folder (old products)
    - Step 2: Upload new products folder (products to match)
    - Step 3: Configure matching (threshold, limit) and start matching
    - Step 4: View results with all matches for each new product
  - CSV support for metadata (filename, category, sku, name, ...)
    - Parses CSV with multiple fields (all optional except filename)
    - Handles headers automatically
    - Supports quoted values
  - Batch processing with progress bars for each step
  - Results display:
    - Summary statistics (total products, matches found, avg matches)
    - Grid view of all new products with their matches
    - Click to view detailed comparison (side-by-side images, score breakdown)
    - Export all results to CSV
  - Real-world data handling:
    - All metadata fields optional (category, SKU, name)
    - Graceful handling of missing/corrupted images
    - Shows warnings for failed feature extractions
    - Displays data quality issues
    - Escapes HTML to prevent XSS from messy data
    - Shows partial results even if some operations fail
  - No frameworks, no build tools - pure HTML/CSS/JavaScript
  - _Requirements: 1.1, 6.1, 6.2, 9.1, 9.3, 10.2_

- [x] 9-10. Folder upload and results display (COMPLETED in task 8 - folder-based workflow)
  - Folder upload with drag-and-drop
  - CSV parsing for metadata
  - Batch processing with progress tracking
  - Results display with color-coded scores
  - CSV export functionality
  - _Requirements: 1.1, 1.2, 4.1, 4.2, 10.1, 10.2_

- [x] 11. Detailed comparison modal (COMPLETED in task 8)
  - Side-by-side image comparison
  - Similarity score breakdown (color, shape, texture) with visual bars
  - Product metadata display (handles NULL fields)
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 14. Performance optimizations (OPTIONAL - for large catalogs)
  - Add lazy loading for product images
  - Optimize database queries with indexes (might already be done in task 2)
  - Add image compression if needed i dont think we store the images other than for processing so may not need it. 
  - optimize to first match based on category and unknown catgeory for images, instead of matching all right off the bat im pretty sure we are already doing this but just check the matching should be efficient.
  - _Requirements: 8.1, 8.2_![alt text](image.png)

- [ ] 15. Implement error handling and user feedback
  - Add toast notifications for success and error messages
  - Implement retry logic for failed API requests with exponential backoff
  - Create user-friendly error messages for common failures
  - Add tooltips and help text to key UI elements
  - Implement loading spinners and progress indicators for all async operations
  - Add visual feedback for drag-and-drop interactions
  - _Requirements: 9.4, 10.3, 10.4_

- [ ] 16. Package application as Windows executable (SIMPLIFIED)
  - Use PyInstaller to create standalone executable (much simpler than Electron Builder!)
  - Command: `pyinstaller --onefile --windowed --add-data "backend/static;backend/static" main.py`
  - Bundle Python backend, Flask, OpenCV, and all dependencies
  - Configure application to store database in AppData folder
  - Test executable on clean Windows system without Python installed
  - Single .exe file - no complex installer needed!
  - _Requirements: 9.2_

- [ ] 17. Create end-to-end tests for critical workflows

  - Write automated test for uploading new product and viewing matches
  - Write automated test for batch upload workflow
  - Write automated test for adding historical product to catalog
  - Write automated test for threshold filtering and result limiting
  - _Requirements: 1.1, 4.1, 6.1, 8.4_
