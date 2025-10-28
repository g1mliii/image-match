# Project Structure

```
product-matching-system/
├── backend/
│   ├── docs/                       # Documentation
│   │   ├── DATABASE_DESIGN.md
│   │   ├── IMAGE_PROCESSING_ERRORS.md
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── MATCHING_SERVICE.md
│   │   ├── REAL_WORLD_DATA_HANDLING.md
│   │   ├── SIMILARITY_ERROR_HANDLING.md
│   │   └── SKU_IMPLEMENTATION.md
│   ├── static/                     # Frontend files
│   │   ├── index.html              # Main UI
│   │   ├── styles.css              # Styling
│   │   └── app.js                  # Frontend logic
│   ├── tests/                      # Test files
│   │   ├── test_database.py
│   │   ├── test_image_processing.py
│   │   ├── test_similarity.py
│   │   ├── test_matching.py
│   │   └── ...
│   ├── uploads/                    # Uploaded product images
│   ├── app.py                      # Flask REST API
│   ├── database.py                 # Database layer
│   ├── image_processing.py         # Feature extraction
│   ├── similarity.py               # Similarity computation
│   ├── product_matching.py         # Matching logic
│   ├── feature_cache.py            # Feature caching
│   ├── product_matching.db         # SQLite database
│   └── requirements.txt            # Backend dependencies
├── .kiro/                          # Kiro IDE configuration
│   └── specs/                      # Project specifications
├── main.py                         # Desktop app launcher
├── requirements.txt                # All Python dependencies
├── README.md                       # Project overview
├── SETUP_SIMPLE.md                 # Setup instructions
└── .gitignore                      # Git ignore rules
```

## Core Files

### Backend
- **app.py**: Flask REST API with endpoints for upload, match, and product details
- **database.py**: SQLite database operations (CRUD, features, matches)
- **image_processing.py**: Image preprocessing and feature extraction (color, shape, texture)
- **similarity.py**: Similarity computation between feature vectors
- **product_matching.py**: Matching logic with category filtering and error handling
- **feature_cache.py**: Caching mechanism for extracted features

### Frontend
- **static/index.html**: Folder-based workflow UI (4-step wizard)
- **static/styles.css**: Modern, responsive styling
- **static/app.js**: Vanilla JavaScript (no frameworks)

### Desktop
- **main.py**: Pywebview launcher that wraps Flask app as desktop application

## Workflow

1. Upload historical catalog folder → Process images
2. Upload new products folder → Process images
3. Configure matching (threshold, limit) → Find matches
4. View results → Export to CSV
