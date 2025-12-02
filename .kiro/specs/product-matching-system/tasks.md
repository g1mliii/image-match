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
 
- [x] 27. Implement flexible metadata linking system in CSV Builder


  - **Goal:** Support different business data organization methods - adapt to their system, not force them into ours
  - **Core Problem:** Businesses store product data differently (SKU in filename, folder structure, separate database, etc.)
  - **Solution:** Multiple linking strategies with auto-detection and preview
  
  - **Linking Strategies:**
    - **Strategy 1: Filename = SKU** (e.g., PM-001.jpg → SKU: PM-001)
      - Remove file extension, use as SKU
      - Match against imported product data
    - **Strategy 2: Filename contains SKU** (e.g., photo_PM-001_front.jpg → SKU: PM-001)
      - Regex pattern matching: `[A-Z]+-\d+`, `\d{4,}`, custom patterns
      - Extract SKU from anywhere in filename
    - **Strategy 3: Folder name = SKU** (e.g., /PM-001/image.jpg → SKU: PM-001)
      - Use parent folder name as SKU
      - Useful for multi-image products
    - **Strategy 4: Fuzzy name matching** (e.g., blue-plate.jpg → "Blue Ceramic Plate")
      - Match image filename to product name (case-insensitive, ignore special chars)
      - Less reliable but useful when no SKU system exists
    - **Strategy 5: Manual linking** (fallback for unmatched products)
      - Side-by-side UI: image preview + product list
      - Click/drag to link
      - Keyboard shortcuts for speed
  
  - **Workflow:**
    - **Step 1:** User uploads images (filenames + categories auto-detected)
    - **Step 2:** User imports existing product data (CSV/Excel with SKU, name, price, etc.)
    - **Step 3:** System analyzes and suggests best linking method
      - Shows preview: "95/100 products matched using 'Filename = SKU'"
      - User can try different strategies
    - **Step 4:** User reviews and applies linking
    - **Step 5:** Manual linking UI for unmatched products
    - **Step 6:** Export final CSV with all metadata linked
  
  - **Import Product Data Options:**
    - Upload CSV/Excel file with existing product data
    - Paste from clipboard (from ERP, database export, etc.)
    - Enter manually in CSV Builder (current method)
  
  - **UI Components:**
    - Linking strategy selector with match count preview
    - Visual preview of matches before applying
    - Unmatched products list with manual linking interface
    - Progress indicator: "Linked: 95/100 ✓ | Unlinked: 5 ⚠️"
  
  - **Smart Features:**
    - Auto-detect best strategy based on match count
    - Learn from user's manual links and suggest patterns
    - Validate: warn if duplicate SKUs or missing required fields
    - Undo/redo for linking operations
  
  - _Requirements: Improves Requirements 1, 6, 10 - makes system work with any business data structure_

- [x] 27.1. Add Excel workflow to CSV Builder (Export → Edit → Import)



  - **Goal:** Let users work in Excel for bulk metadata entry (faster for large datasets)
  
  - **Export Template Feature:**
    - After uploading images, add "Export Template CSV" button
    - Generates CSV with: filename, category (auto-detected), empty columns for sku, name, price, price_history, performance_history
    - Downloads as `product-template.csv`
    - Shows instructions: "Fill in Excel and re-import when done"
  
  - **Import Completed CSV Feature:**
    - Add "Import Completed CSV" button
    - User uploads CSV with filled metadata
    - System matches by filename (exact match required)
    - Updates products with imported data
    - Shows results: "Imported metadata for 95/100 products ✓ | 5 not found ⚠️"
    - Validates data: SKU format, price is number, dates are valid, etc.
  

  - **Excel Workflow Benefits:**
    - Users can copy/paste from their existing systems (ERP, database)
    - Use Excel formulas (e.g., auto-increment SKUs, calculate prices)
    - Sort, filter, find/replace for bulk operations
    - Work offline, share with team for collaboration
    - Handle 1000s of products easily
  
  - **Hybrid Workflow Support:**
    - User can export template, merge with existing data in Excel (VLOOKUP), then import
    - Or import existing data first, then export to add missing fields
    - Flexible to match user's preferred workflow
  
  - **Error Handling:**
    - Clear error messages for mismatched filenames
    - Validation warnings for invalid data (negative prices, bad dates, etc.)
    - Option to skip invalid rows or fix in browser
    - Preview changes before applying
  
  - _Requirements: Improves Requirements 1, 6, 10 - enables Excel power users and large datasets_

- [x] 27.2. Update UI tooltips, help text, and CSV format documentation


  - **Goal:** Guide users through new linking and Excel workflows with clear instructions
  
  - **CSV Builder Updates:**
    - Add help section explaining linking strategies with examples
    - Tooltip on "Import Product Data": "Upload CSV/Excel with your existing product data (SKU, name, price). We'll match it to your images."
    - Tooltip on "Export Template": "Download CSV with filenames. Fill in Excel and re-import."
    - Add visual examples of each linking strategy (screenshots or diagrams)
    - Update progress steps to include linking step
  
  - **Main App CSV Format Popup Updates:**
    - Update CSV format help modal to explain new flexible linking
    - Add section: "Don't have a CSV? Use CSV Builder to create one!"
    - Show examples of different CSV formats that work (with/without SKU, different column orders)
    - Explain that filename is the only required field for linking
  
  - **Main App Upload Section:**
    - Add tooltip: "CSV is optional. Upload images only for simple mode, or use CSV Builder to add metadata."
    - Link to CSV Builder from main app: "Need help creating a CSV? → CSV Builder"
    - Update advanced mode explanation to mention linking options
  
  - **CSV Builder Step-by-Step Guidance:**
    - Step 1 (Upload): "Upload your product images. We'll extract filenames and detect categories from folders."
    - Step 2 (Link): "Import your product data or export template to fill in Excel. We'll help you link images to metadata."
    - Step 3 (Review): "Review linked products and fix any unmatched items."
    - Step 4 (Export): "Export final CSV for use in main app."
  
  - **Help Button Content:**
    - Add FAQ: "What if my images don't have SKUs in the filename?"
    - Add FAQ: "Can I use my existing product database?"
    - Add FAQ: "What's the fastest way to add metadata for 1000 products?"
    - Add examples of common business scenarios and recommended workflows
  
  - _Requirements: Improves Requirements 9.4, 10.3 - better user guidance and onboarding_

- [x] 28. Implement catalog management and database cleanup

  - **Goal:** Let users view, manage, and clear their persistent catalog to prevent database bloat and RAM issues
  - **Problem:** Database grows indefinitely, users don't know data persists, no way to clear old products. Test data also accumulates when running tests repeatedly.
  
  - **Catalog Overview UI:**
    - Show catalog statistics in main app maybe seperate tab or button that opesn it should be differnt file simiar to csv builder otherwise our main app will be too large:
      - Total products: "1,247 historical products | 53 new products"
      - Database size: "156 MB" (calculate from file size)
      - Last updated: "2024-01-15 14:30"
      - Breakdown by category: "Plates: 450 | Placemats: 320 | ..."
    - Add "View Catalog" button to browse all products
    - Add "Refresh" button to reload catalog stats
  
  - **Catalog Loading Options:**
    - Radio buttons in Step 1 (Historical Catalog):
      - ○ Use existing catalog (1,247 products) - Skip upload, use saved data
      - ○ Add to existing catalog - Upload new products, merge with existing
      - ○ Replace catalog - Clear all and upload fresh
    - Default to "Use existing" if catalog exists
    - Show warning before replacing: "This will delete 1,247 products. Continue?"
  
  - **Database Cleanup Features:**
    - **Clear All Products:** Delete everything, start fresh
    - **Clear Historical Only:** Keep new products, delete historical
    - **Clear New Only:** Keep historical, delete new products
    - **Clear Matches Only:** Delete match results, keep products and features
    - **Clear by Category:** Select categories to delete
    - **Clear by Date:** Delete products older than X days
    - **Vacuum Database:** Reclaim disk space after deletions (SQLite VACUUM command)
    - **Test Data Cleanup:** Utility to clear test data (for development/testing)
  
  - **Cleanup UI (Settings/Tools Menu):**
    ```
    Database Management
    ├─ Current Size: 156 MB
    ├─ Products: 1,247 historical | 53 new
    ├─ Matches: 2,450 stored
    └─ Actions:
       [Clear All Products]
       [Clear Historical Only]
       [Clear New Only]
       [Clear Matches]
       [Clear by Categor
       [Vacuum Database]
       [Export Backup]
    ```
  
  - **Safety Features:**
    - Confirmation dialogs before any deletion
    - Show what will be deleted: "This will delete 450 products in 'Plates' c   - Ext
    - Option to export backup before clearing
    - Undo not possible - make this clear to users
    - Disable cleanup during active matching operations
  
  - **Automatic Maintenance:**
    - Warn when database exceeds 500 MB: "Database is large (523 MB). Consider cleaning up old products."
    - Warn when RAM usage is high during matching: "Large catalog detected. Consider filtering by category."
    - Option to auto-delete products older than X days (user configurable)
    - Option to auto-vacuum database weekly
  
  - **Performance Optimizations:**
    - low ram mode maybe for low ram systems 
    - Don't load all products into RAM at once
    - Use pagination for catalog view (show 100 at a time)
    - Load features only for current category during matching
    - Add database indexes for faster queries maybe alread done
    - Consider archiving old products to separate database file
  
  - **Export/Backup Features:**
    - Export catalog to CSV (all products + metend CSV/JSON upload to include optional price history data per product
    - Schema: `{ "sku": "ABC123", "prices": [{"date": "2024-01-15", "price": 29.99}, ...] }`
    - Support up to 12 months of historical price data per product
  - **Data Stod auto-backups (optional)
  
  - **Database File Mrage:**
    - Create newase file location: "C:\Users\...\ProductMatcher\product_matching.db"
    - Button to open folder in file explorer
    - Button to copy database path to clipboard
    - Show uploads folder size: "Images: 2.3 GB in 1,247 files"
    - Option to clear uploaded images (keep metadata only)
  
  - **Implementation Details:**
    ```python
    # Database size
    def get_database_size():
        return os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
    
    # Clear products
    def clear_products(filter_type='all', category=None, older_than_days=None):
        # Delete from products, features, matches tables
        # Delete associated image files
        # Return count of deleted items
    
    # Vacuum database
    def vacuum_database():
        conn.execute('VACUUM')  # Reclaim space
    
    # Category-based loading (prevent RAM issues)
    def get_features_by_category(category, limit=1000):
        # Only load subset of products
        # Use for matching to prevent loading 10K products into RAM
    ```
  
  - **UI Placement:**
    - Catalog stats: Top of Step 1 (Historical Catalog section)
    - Cleanup options: Settings menu or Tools menu
    - Quick actions: "Clear All" button in catalog view
    - Warnings: Toast notifications when database is large
  
  - _Requirements: Improves Requirements 8.4, 9.1, 10.3 - prevents database bloat, better resource management_


- [x] 28.1. Add catalog browser and product management UI

  - **Goal:** Let users view and manage individual products in their catalog
  
  - **Catalog Browser:**
    - Grid view of all products with thumbnails
    - Filter by: category, date added, has features, has matches
    - Sort by: date, name, SKU, category
    - Search by: filename, SKU, name
    - Pagination: 50 products per page
  
  - **Product Actions:**
    - View details: metadata, features status, match history
    - Edit metadata: update SKU, name, category, i think we have some form of this in csv builder or maybe not so just check.
    - Delete product: remove from database + delete image file same as above
    - Re-extract features: if feature extraction failed same as above
    - Bulk actions: select multiple products, delete/edit in bulk same as above
  
  - **Product Details Modal:**
    ```
    Product Details
    ├─ Image preview
    ├─ Filename: IMG_1234.jpg
    ├─ Category: Plates
    ├─ SKU: PLT-001
    ├─ Name: Blue Ceramic Plate
    ├─ Price: $29.99
    ├─ Features: ✓ Extracted (CLIP embedding)
    ├─ Matches: 15 matches found
    ├─ Created: 2024-01-15 14:30
    └─ Actions: [Edit] [Delete] [Re-extract Features]
    ```
  
  - **Bulk Operations:**
    - Select products with checkboxes
    - Bulk delete: "Delete 25 selected products"
    - Bulk edit category: "Change category for 25 products"
    - Bulk re-extract: "Re-extract features for 25 products"
  
  - _Requirements: Improves Requirements 10.1, 10.2 - better catalog visibility and control_



- [ ] 25. Cross-platform GPU acceleration for CLIP (COMPLETED)

  - **Status:** ✅ CLIP GPU acceleration fully implemented and working
  - **Implemented Features:**
    - **PyTorch GPU Auto-Detection:** Works across CUDA (NVIDIA), ROCm (AMD), and MPS (Apple Silicon)
    - **AMD ROCm Support:** Detects AMD GPUs via `torch.cuda.is_available()` and distinguishes from NVIDIA
    - **Device Detection:** `detect_device()` function with priority: CUDA/ROCm > MPS > CPU
    - **Batch Processing:** Configurable batch size (default 32) for maximum GPU efficiency
    - **Automatic Mixed Precision (AMP):** Faster GPU inference with `torch.amp.autocast()`
    - **GPU Memory Management:** VRAM monitoring, periodic cache clearing, OOM prevention
    - **Graceful CPU Fallback:** Automatic fallback if GPU unavailable or fails
    - **Performance:** 10-50x speedup on GPU vs CPU (0.01-0.05s vs 0.5-1.0s per image)
  - **Implementation Details:**
    - File: `backend/image_processing_clip.py`
    - Functions: `detect_device()`, `get_device_info()`, `extract_clip_embedding()`, `batch_extract_clip_embeddings()`
    - GPU status displayed in UI: "⚡ AMD GPU Active (ROCm)" / "⚡ NVIDIA GPU Active (CUDA)" / "⚡ Apple Silicon Active (MPS)" / "💻 CPU Mode"
    - Model caching: `~/.cache/clip-models/` (~350MB one-time download)
    - Config file: `~/.cache/clip-models/config.json`
  - _Note: This implementation is production-ready and works across all platforms_

- [ ] 25.1. GPU acceleration performance monitoring and metrics
  - Log processing times for GPU vs CPU operations
  - Add optional performance metrics to help menu
  - Track and display: images/sec, embeddings extracted/sec


- [ ] 25.2. GPU acceleration testing and validation
  - Test on Windows with NVIDIA GPUs (CUDA)
  - Test on Windows with AMD GPUs (ROCm via PyTorch ROCm build)
  - Test on macOS with Apple Silicon (M1/M2/M3/M4/M5 via MPS)
  - Test fallback behavior when no GPU available
  - Ensure ARM compatibility is maintained
  - Verify Mode 2 (Metadata) unaffected by GPU settings
  - Test CLIP batch processing with different batch sizes (8, 16, 32, 64)

















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


  - [ ] 16.5. Create application icons for cross-platform packaging
  - **Windows Icon (.ico):**
    - Create `app_icon.ico` with multiple sizes: 16x16, 32x32, 48x48, 256x256
    - Use online tool (e.g., icoconvert.com) or GIMP to create multi-resolution .ico
    - Place in project root for development, will be bundled during packaging
  - **macOS Icon (.icns):**
    - Create `app_icon.icns` with multiple sizes for Retina displays
    - Use `iconutil` on macOS or online converter to create .icns from PNG
    - Required sizes: 16x16, 32x32, 128x128, 256x256, 512x512, 1024x1024
  - **Source Icon:**
    - Create or obtain a 1024x1024 PNG as master icon
    - Simple, recognizable design that works at small sizes
    - Suggestion: Product/box icon or "PM" monogram
  - **Testing:**
    - Uncomment `icon=icon_path` line in main.py
    - Run `python main.py` to verify icon appears in title bar
    - Test on both Windows and macOS if possible
  - _Note: Icons will be bundled into executables during tasks 17-18_

  - [ ] 17. Package Windows executable with PyInstaller for public release whic users will download by clicking link on site and some solution to enable download without triggerig virus issue on chrome and other browsers
  - Modify database.py to store data in `%APPDATA%\ProductMatcher\` (detect PyInstaller with `getattr(sys, 'frozen', False)`)
  - Create product-matcher.spec: `--onefile --windowed --add-data "backend/static;backend/static" --icon=app_icon.ico --name "Product Matcher"`
  - Create build.bat: `pyinstaller --clean product-matcher.spec`
  - Test packaged exe on clean Windows system: verify launches, creates AppData folders, full workflow works
  - Package: zip exe + README.txt + sample.csv template
  - It should be a 1 click solution for everybody so that all features including CLIP and GPU acceleration (AMD, NVIDIA, Intel, ARM) and all dependencies/requirements are installed and work correctly.
  - **Pre-Packaging Testing (CRITICAL):**
    - Test `python setup_gpu.py` on clean Windows system (no Python packages installed)
    - Verify PyTorch auto-installation works for NVIDIA GPUs (CUDA detection + install)
    - Verify PyTorch auto-installation works for AMD GPUs (ROCm detection + HIP SDK guidance)
    - Verify Intel GPU support (optional): `intel-extension-for-pytorch` can be installed separately
    - Verify all dependencies from `requirements.txt` and `backend/requirements.txt` install correctly
    - Test GPU detection: NVIDIA (nvidia-smi check), AMD (ROCm DLL check), Intel (IPEX check), CPU fallback
    - Test CLIP model download and caching (first run downloads ~350MB model)
    - Verify `python check_gpu.py` shows correct GPU status after setup
    - Test full workflow after setup: upload images → extract CLIP features → match → verify results
    - Test on systems with/without GPU to ensure graceful CPU fallback
    - Test multiprocessing/multithreading: Verify batch matching uses CPU parallelization (built-in, no extra deps)
    - Document any manual steps required (e.g., AMD HIP SDK installation, Intel IPEX for Intel GPU users)
  - _Requirements: 9.2_

- [ ] 18. Package macOS application for public release whic users will download by clicking link on site and some solution to enable download without triggerig virus issue on chrome and other browsers
  - It should be a 1 click solution for everybody so that all features including CLIP and GPU acceleration (Apple Silicon MPS, Intel GPU optional) and all dependencies/requirements are installed and work correctly.
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
  - **Pre-Packaging Testing (CRITICAL):**
    - Test `python setup_gpu.py` on clean macOS system (no Python packages installed)
    - **Apple Silicon (M1/M2/M3/M4/M5) Testing:**
      - Verify PyTorch auto-installation with MPS (Metal Performance Shaders) support
      - Verify MPS GPU detection: `torch.backends.mps.is_available()` returns True
      - Test CLIP model runs on MPS backend (should be automatic, no config needed)
      - Verify all dependencies install correctly (ARM-native versions)
      - Test performance: CLIP should be 5-10x faster than CPU on Apple Silicon
    - **Intel Mac Testing:**
      - Verify PyTorch CPU installation (no GPU acceleration on Intel Macs)
      - Verify all dependencies install correctly (x86_64 versions)
      - Test Rosetta 2 compatibility if building ARM binary on Intel Mac
    - **Intel Mac with Intel GPU (Optional):**
      - Intel GPU support via `intel-extension-for-pytorch` (optional, user can install separately)
      - Verify graceful fallback to CPU if Intel GPU extension not installed
    - **Universal Testing:**
      - Test CLIP model download and caching (~350MB, should work on both architectures)
      - Verify `python check_gpu.py` shows correct status (MPS for Apple Silicon, CPU for Intel)
      - Test full workflow: upload images → extract CLIP features → match → verify results
      - Verify database paths work correctly: `~/Library/Application Support/ProductMatcher/`
      - Test app launches without terminal/console window
      - Test multiprocessing/multithreading: Verify batch matching uses CPU parallelization (built-in, no extra deps)
    - Document any manual steps required (should be none for macOS - everything auto-installs)
  - _Requirements: 9.2_




- [ ] 31. Implement mobile photo upload with QR code pairing

  - **Goal:** Enable users to upload product photos from their phone to the desktop app via QR code pairing
  - **Use Case:** Client takes photos in warehouse/store, scans QR code on desktop, uploads directly to catalog
  
  - **Architecture:**
    - Desktop app generates unique session token and displays QR code
    - QR code contains URL: `http://{local_ip}:5000/mobile?session={token}`
    - User scans QR with phone camera → Opens mobile upload page in browser
    - Mobile page uploads photos to desktop app via REST API
    - Desktop app receives photos and adds to selected snapshot
    - No mobile app installation needed - works in any browser
  
  - **Backend Changes (app.py):**
    - **New API Endpoints:**
      - `POST /api/mobile/session/create` - Generate new mobile session
        - Returns: session_token, qr_code_url, expires_at (30 min)
        - Store active sessions in memory: `{token: {created_at, expires_at, is_used}}`
      - `GET /api/mobile/session/{token}/validate` - Check if session is valid
        - Returns: {valid: true/false, expired: true/false}
      - `POST /api/mobile/upload` - Upload photos from mobile
        - Requires: session_token, images (multiple files), snapshot_name, is_historical, category (optional)
        - Validates session token before accepting upload
        - Creates snapshot if doesn't exist or adds to existing
        - Returns: {success: true, products_added: 5, snapshot_name: "..."}
      - `POST /api/mobile/session/{token}/close` - Invalidate session token
      - `GET /api/mobile/session/active` - Get active session info for desktop UI
    - **Session Management:**
      - Sessions expire after 30 minutes
      - One-time use option: Session invalidated after first upload
      - Reusable option: Session stays active until manually closed or expired
      - Cleanup expired sessions every 5 minutes (background task)
  
  - **QR Code Generation:**
    - Use Python library: `qrcode` (add to requirements.txt)
    - Generate QR code as base64 image for display in UI
    - QR code contains: `http://{local_ip}:5000/mobile?session={token}`
    - Auto-detect local IP address (prefer 192.168.x.x or 10.x.x.x)
    - Handle multiple network interfaces (show all available IPs)
  
  - **Desktop UI Changes (index.html or new mobile-pairing.html):**
    - **Mobile Upload Button:**
      - Add button in main app: "Upload from Phone"
      - Opens modal with QR code and instructions
    -and 
      
    - **Upload Notifications:**
      - Show toast when photos uploaded from mobile
      - "5 photos added to 'Summer 2024 Plates' from mobile"
      - Update catalog stats in real-time
  
  - **Mobile Upload Page (new file: static/mobile-upload.html):**
    - **Mobile-Optimized UI:**
      - brutalist design simliar to our app


    - **Features:**
      - Multiple photo selection from gallery
      - Take photo with camera (use `<input type="file" accept="image/*" capture="camera">`)
      - Preview selected photos before upload
      - Remove photos from selection
      - Progress bar during upload
      - Success/error messages
      - Auto-generated snapshot name: "Mobile Upload - {date} {time}"
      - Option to select existing snapshot from dropdown
      - Category dropdown (loads from desktop app)
    - **Session Validation:**
      - On page load, validate session token
      - If invalid/expired: Show error "Session expired. Please scan QR code again."
      - If valid: Show "Connected to Desktop ✓"
      - Poll session status every 30 seconds
  
  - **Network Discovery:**
    - **Auto-detect local IP:**
      ```python
      import socket
      def get_local_ip():
          s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
          s.connect(("8.8.8.8", 80))
          ip = s.getsockname()[0]
          s.close()
          return ip
      ```
    - **Handle multiple IPs:**
      - If multiple network interfaces, show all in modal
      - User can select which IP to use for QR code
      - Prefer 192.168.x.x (home WiFi) over 10.x.x.x (VPN)
    - **Firewall Warning:**
      - If connection fails, show troubleshooting tips
      - "Make sure phone and computer are on same WiFi network"
      - "Check firewall settings if connection fails"
  
  - **Security:**
    - **Session Tokens:**
      - Generate cryptographically secure random tokens (32 characters)
      - Use `secrets.token_urlsafe(32)` in Python
      - Store tokens in memory only (not in database)
      - Expire after 30 minutes
    - **CORS:**
      - Allow mobile uploads from any origin (already enabled with CORS)
      - Validate session token on every request
    - **Rate Limiting:**
      - Limit uploads per session (e.g., max 100 photos per session)
      - Prevent abuse with rate limiting (max 10 uploads per minute)
    - **HTTPS (Optional):**
      - For production, consider self-signed certificate for HTTPS
      - Prevents man-in-the-middle attacks on local network
      - Not critical for local network use
  
  - **Snapshot Integration:**
    - Mobile uploads automatically create snapshots
    - Default name: "Mobile Upload - {date} {time}"
    - User can rename later in Catalog Manager
    - Snapshot type (historical/new) selected on mobile page
    - Snapshot appears in Catalog Manager immediately after upload
    - Auto-selected in active catalogs by default
  
  - **Multiple Photo Support:**
    - Accept multiple files in single upload request
    - Process photos in batch (extract features for all)
    - Show progress: "Processing 3/5 photos..."
    - If some photos fail (corrupted, wrong format), show partial success
    - "3 photos uploaded successfully, 2 failed (invalid format)"
  
  - **Metadata Support:**
    - Category: Dropdown of existing categories (loaded from desktop)
    - SKU: Optional text field (not recommended on mobile - tedious)
    - Product name: Optional text field (not recommended on mobile)
    - Keep mobile UI simple - add detailed metadata later on desktop
  
  - **Error Handling:**
    - Invalid session token: "Session expired. Please scan QR code again."
    - Network error: "Connection lost. Check WiFi and try again."
    - Upload failed: "Upload failed. Check file format and try again."
    - File too large: "Photo too large (max 10 MB per photo)"
    - Invalid format: "Invalid format. Use JPEG, PNG, or WebP."
    - Show clear error messages with retry button
  
  - **Testing:**
    - Test QR code generation and scanning
    - Test session creation and expiration
    - Test mobile upload with single photo
    - Test mobile upload with multiple photos (5, 10, 20)
    - Test session validation (valid, expired, invalid token)
    - Test network discovery (multiple IPs, WiFi vs Ethernet)
    - Test on different phones (iOS Safari, Android Chrome)
    - Test with different image formats (JPEG, PNG, HEIC)
    - Test with large images (10+ MB)
    - Test snapshot creation from mobile uploads
    - Test category dropdown loading
    - Test connection loss during upload (retry logic)
  
  - **Dependencies:**
    - Add to requirements.txt:
      - `qrcode[pil]` - QR code generation
      - `pillow` - Image processing (already included)
  
  - **Future Enhancements (Optional):**
    - Push notifications to desktop when photos uploaded
    - Live preview of uploaded photos on desktop
    - Batch metadata editing on desktop after mobile upload
    - Support for video uploads (for product demos)
    - OCR to extract SKU from product labels in photos
  
  - _Requirements: New feature - improves Requirements 1.1, 6.1, 9.1 - mobile convenience and flexibility_



