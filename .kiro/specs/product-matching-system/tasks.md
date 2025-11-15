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

- [ ] 17. Package Windows executable with PyInstaller
  - Modify database.py to store data in `%APPDATA%\ProductMatcher\` (detect PyInstaller with `getattr(sys, 'frozen', False)`)
  - Create product-matcher.spec: `--onefile --windowed --add-data "backend/static;backend/static" --name "Product Matcher"`
  - Create build.bat: `pyinstaller --clean product-matcher.spec`
  - Test packaged exe on clean Windows system: verify launches, creates AppData folders, full workflow works
  - Package: zip exe + README.txt + sample.csv template
  - _Requirements: 9.2_

- [ ] 18. Package macOS application (optional - if needed)
  - Update database.py to detect OS with `platform.system()`: use `~/Library/Application Support/ProductMatcher/` on macOS
  - Create product-matcher-mac.spec: `--onefile --windowed --add-data "backend/static:backend/static" --icon=app_icon.icns --name "Product Matcher"`
  - Create build-mac.sh: `pyinstaller --clean product-matcher-mac.spec`
  - Test packaged .app on clean macOS system: verify launches, creates folders, full workflow works
  - Optional: code sign with `codesign` to avoid Gatekeeper warnings (requires Apple Developer account)
  - Package: zip .app + README.txt + sample.csv template
  - Note: Must build on macOS machine (PyInstaller can't cross-compile)
  - _Requirements: 9.2_

## Future Enhancements (Public Release)

- [ ] 19. Polish UI for public release
  - Professional color scheme and branding style in the deign language of google android with rounded buttons things of that nature.
  - gpu acceleratoin to speed up tasks, and ui if need be.
  - Better onboarding flow with tutorial/step explanation area/or tooltops and other ways to guide user.
  - Improved results visualization (charts, graphs)

- [ ] 20. Add advanced features
  - Adjustable similarity weights (color, shape, texture sliders)
  - Batch export with images (not just CSV)
  - Duplicate detection report
  - Side-by-side comparison grid view
  - Search/filter results
  - Save/load matching sessions
  - Undo/redo functionality


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

- [ ] 23. Marketing and launch
  - Create demo video showing workflow
  - Write blog post/case study
  - Post on Product Hunt, Reddit, HackerNews
  - Reach out to relevant communities (e-commerce, retail)
  - SEO optimization for website
  - Social media presence (Twitter, LinkedIn)
