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

- [x] 15. Implement error handling and user feedback
  - Add toast notifications for success and error messages
  - Implement retry logic for failed API requests with exponential backoff
  - Create user-friendly error messages for common failures
  - Add tooltips and help text to key UI elements
  - Implement loading spinners and progress indicators for all async operations
  - Add visual feedback for drag-and-drop interactions
  - _Requirements: 9.4, 10.3, 10.4_


- [x] 24. Add price history tracking feature

- [x] 25. Add performance history tracking feature
  - **Performance Metrics Support:**
    - Extend CSV/JSON to include performance metrics per product
    - Schema: `{ "sku": "ABC123", "performance": [{"date": "2024-01-15", "sales": 150, "views": 1200, "conversion": 12.5}, ...] }`
    - Support up to 12 months of performance data per product
  - **Data Storage:**
    - Create new `performance_history` table with columns: product_id, date, sales, views, conversion_rate, revenue
    - Link performance records to products via foreign key relationship
    - Add indexes on product_id and date for efficient querying
  - **Smart Linking:**
    - When matching products, automatically link performance history from matched historical products
    - Display performance trends for matched items in comparison view
    - Calculate performance statistics: avg sales, total revenue, conversion trends
  - **UI Display:**
    - Add performance history chart to product detail/comparison view
    - Show sparkline charts in match results for quick performance visualization
    - Display performance statistics alongside similarity scores
    - Add filter to show only products with performance history data
  - **Export Enhancement:**
    - Include performance history data in CSV exports
    - Add option to export performance trends summary (avg sales, revenue, conversion per product)
  - _Requirements: New feature - extends Requirements 1, 4, 7_


- [x] 26. Add CSV Builder UI tool should be idiot prood and handl broken/missing data just like the rest our code.
  - **Interactive CSV Creator:**
    - Create modal/page with step-by-step CSV builder
    - Allow users to upload image folders and auto-populate filenames
    - Provide form fields for each product: category, sku, name, price, performance
    - Support multiple folder uploads (different categories)
    - Real-time CSV preview as user fills in data
  - **Smart Features:**
    - Auto-detect category from folder name
    - Bulk edit fields (apply same value to multiple products)
    - Copy/paste from Excel support
    - Validation with visual feedback
    - Undo/redo functionality
  - **Price History Builder:**
    - Visual date picker for price entries
    - Add/remove price points easily
    - Quick format: just enter prices, dates auto-generated
    - Import existing price data from clipboard
  - **Performance History Builder:**
    - Visual form for performance metrics
    - Add/remove performance entries
    - Calculate conversion rate automatically
    - Import from clipboard/Excel
  - **Export Options:**
    - Download as CSV
    - Copy to clipboard
    - Save as template for reuse
    - Multiple format options (with/without headers, different separators)
  - **User Experience:**
    - Progress indicator showing completion
    - Save draft functionality
    - Load previous CSV for editing
    - Clear visual examples and tooltips
    - Mobile-friendly interface
  - _Requirements: New feature - improves usability for Requirements 1, 6, 10_

  - **JSON Metadata Support:**
    - Extend CSV/JSON upload to include optional price history data per product
    - Schema: `{ "sku": "ABC123", "prices": [{"date": "2024-01-15", "price": 29.99}, ...] }`
    - Support up to 12 months of historical price data per product
  - **Data Storage:**
    - Create new `price_history` table with columns: product_id, date, price, currency
    - Link price records to products via foreign key relationship
    - Add indexes on product_id and date for efficient querying
  - **Smart Linking:**
    - When matching products, automatically link price history from matched historical products
    - Display price trends for matched items in comparison view
    - Calculate price statistics: min, max, average, current trend (up/down/stable)
  - **UI Display:**
    - Add price history chart to product detail/comparison view
    - Show sparkline charts in match results for quick price trend visualization
    - Display price statistics alongside similarity scores
    - Add filter to show only products with price history data
  - **Export Enhancement:**
    - Include price history data in CSV exports
    - Add option to export price trends summary (avg, min, max per product)
  - _Requirements: New feature - extends Requirements 1, 4, 7_























- [ ] 25. Implement cross-platform GPU acceleration using OpenCL (via OpenCV UMat)
  - **Why OpenCL:** Cross-platform solution that works on Windows (NVIDIA/AMD/Intel), macOS (Apple Silicon M1-M5), and Linux without separate builds or platform-specific code
  - **OpenCV UMat Implementation:**
    - Use `cv2.UMat` instead of regular numpy arrays for automatic GPU acceleration
    - OpenCV automatically uses OpenCL backend when available
    - No additional dependencies needed - OpenCV already includes OpenCL support
    - Graceful automatic fallback to CPU if no GPU/OpenCL available
  - **Image Preprocessing on GPU:**
    - Convert images to UMat: `img_gpu = cv2.UMat(img)`
    - Use standard OpenCV functions - they automatically run on GPU with UMat
    - Accelerate resize, color conversion (BGR to HSV), and normalization operations
    - Implement batch preprocessing for multiple images
  - **Feature Extraction Acceleration:**
    - Move histogram computation to GPU using `cv2.calcHist()` with UMat inputs
    - Color features: HSV histogram computation on GPU
    - Texture features: LBP computation can use UMat for image operations
    - Shape features: Contour detection and moment calculation with UMat
  - **Similarity Computation:**
    - Keep similarity calculations on CPU (numpy) - they're already fast for small vectors
    - GPU overhead not worth it for simple distance calculations
    - Focus GPU usage on image-heavy operations only
  - **Detection and Configuration:**
    - Detect OpenCL availability at startup: `cv2.ocl.haveOpenCL()`
    - Add configuration option to enable/disable GPU acceleration in settings
    - Log GPU status on startup: "OpenCL GPU acceleration: enabled/disabled"
  - **Performance Monitoring:**
    - Log processing times for GPU vs CPU operations
    - Display GPU status in UI: "GPU Acceleration: Active (OpenCL)" or "CPU Mode"
    - Add optional performance metrics to help menu
  - **Fallback Handling:**
    - UMat automatically falls back to CPU if OpenCL fails - no manual handling needed
    - If OpenCL initialization fails, log warning and continue with CPU
    - Provide option to force CPU mode if GPU causes issues
  - **Testing:**
    - Test on Windows with NVIDIA/AMD/Intel GPUs
    - Test on macOS with Apple Silicon (M1/M2/M3/M4/M5)
    - Test fallback behavior when no GPU available
    - Verify performance improvements (expect 2-3x speedup on image operations)
    - Ensure ARM compatibility is maintained
  - **Code Migration Pattern:**
    ```python
    # Before (CPU only):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # After (GPU accelerated):
    img = cv2.imread(path)
    img_gpu = cv2.UMat(img)  # Move to GPU
    hsv_gpu = cv2.cvtColor(img_gpu, cv2.COLOR_BGR2HSV)  # Runs on GPU
    hsv = hsv_gpu.get()  # Get result back to CPU if needed
    ```
  - _Note: This approach is ARM-compatible and works across all platforms without breaking macOS builds_


- [ ] 23. Marketing and launch
  - Create demo video showing workflow
  - Write blog post/case study
  - Post on Product Hunt, Reddit, HackerNews
  - Reach out to relevant communities (e-commerce, retail)
  - SEO optimization for website
  - Social media presence (Twitter, LinkedIn)


- [ ] 16. Write end-to-end tests (backend/tests/test_e2e.py)
  - Create test fixtures: 5 historical images, 3 new images, valid/invalid CSVs
  - Test complete workflow: upload historical → upload new → match → verify results
  - Test CSV metadata handling: valid fields, missing fields, duplicates, invalid format
  - Test category filtering: products match within same category, NULL category handling
  - Test threshold/limit: verify filtering and result limiting work correctly
  - Test error handling: corrupted images, missing features, invalid inputs, empty catalog
  - Test API validation: invalid product_id, threshold, limit return proper 400 errors
  - Test response formats: verify all endpoints return expected fields
  - Run with `pytest backend/tests/test_e2e.py -v` - all tests must pass before packaging
  - _Requirements: 1.1, 4.1, 6.1, 8.4, 10.3_

- [ ] 17. Package Windows executable with PyInstaller for public release whic users will download by clicking link on site and some solution to enable download without triggerig virus issue on chrome and other browsers
  - Modify database.py to store data in `%APPDATA%\ProductMatcher\` (detect PyInstaller with `getattr(sys, 'frozen', False)`)
  - Create product-matcher.spec: `--onefile --windowed --add-data "backend/static;backend/static" --name "Product Matcher"`
  - Create build.bat: `pyinstaller --clean product-matcher.spec`
  - Test packaged exe on clean Windows system: verify launches, creates AppData folders, full workflow works
  - Package: zip exe + README.txt + sample.csv template
  - _Requirements: 9.2_

- [ ] 18. Package macOS application for public release whic users will download by clicking link on site and some solution to enable download without triggerig virus issue on chrome and other browsers
  - Update database.py to detect OS with `platform.system()`: use `~/Library/Application Support/ProductMatcher/` on macOS
  - **ARM (Apple Silicon M1-M5) Build:**
    - Create product-matcher-mac-arm64.spec: `--onefile --windowed --add-data "backend/static:backend/static" --icon=app_icon.icns --name "Product Matcher" --target-arch arm64`
    - Create build-mac-arm.sh: `pyinstaller --clean product-matcher-mac-arm64.spec`
    - Ensure all dependencies (numpy, opencv-python, etc.) are ARM-compatible versions
    - Test on Apple Silicon Mac (M1/M2/M3/M4/M5): verify native ARM performance
  - **Intel (x86_64) Build:**
    - Create product-matcher-mac-x86.spec: `--onefile --windowed --add-data "backend/static:backend/static" --icon=app_icon.icns --name "Product Matcher" --target-arch x86_64`
    - Create build-mac-intel.sh: `pyinstaller --clean product-matcher-mac-x86.spec`
    - Test on Intel Mac or Rosetta 2: verify compatibility
  - **Universal Binary (Optional but Recommended):**
    - Use `lipo` to create universal binary: `lipo -create -output ProductMatcher-Universal ProductMatcher-arm64 ProductMatcher-x86_64`
    - This allows single download for both architectures (larger file size ~100MB)
  - **Distribution Strategy:**
    - Option 1: Provide separate downloads (ARM vs Intel) - requires website detection
    - Option 2: Provide universal binary - works on all Macs but larger download
    - Recommended: Universal binary for simplicity, separate builds for size optimization
  - Optional: code sign with `codesign` to avoid Gatekeeper warnings (requires Apple Developer account)
  - Package: zip .app + README.txt + sample.csv template for each architecture
  - Update download.html to detect architecture and provide correct download link
  - Note: Must build on macOS machine (PyInstaller can't cross-compile). ARM build requires Apple Silicon Mac, Intel build can be done on either with Rosetta 2
  - _Requirements: 9.2_

- [ ] 20. Add advanced features to ui
  - Adjustable similarity weights (color, shape, texture sliders)
  - Batch export with images (not just CSV)
  - Duplicate detection report with rank filters based on performance price history and other thigns that would be usefule for determining what new products would be good to implement.
  - Search/filter results
  - Save/load matching sessions
  - Undo/redo functionality
  - all should be performant and gpu accelerated if necessary taking into account our gpu acceleration plan in task 25 i think on line 236
  - continue to make the ui for like the website its still to0 different even after changes the website looks way better than the ui currently.

- [ ] 22. Implement license key system with LemonSqueezy
  - **LemonSqueezy Setup** (lemonsqueezy.com):
    - Create account and store
    - Create product: "Product Matcher Pro" - $49 one-time payment
    - Enable "Generate unique license keys" feature
    - Get checkout link for GitHub Pages pricing page
  - **License Key Validation in App**:
    - Add "Enter License Key" dialog in Help menu
    - Implement offline validation (check format: XXXX-XXXX-XXXX-XXXX)
    - Store validated key in config file (AppData)
    - Free tier: 50 products limit, Pro tier: unlimited
    - Show upgrade prompt when hitting free tier limit
    - Display license status in app (Free/Pro)


- [ ] 22.5. Set up distribution and analytics
  - GitHub releases for version management
  - Auto-update mechanism in app (optional)
  - Usage analytics - privacy-respecting, opt-in (optional)
  - Crash reporting (optional)
  - User feedback system (optional)
