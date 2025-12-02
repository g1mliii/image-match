# Simple Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode (Browser)
Run Flask server and open in your browser:
```bash
python backend/app.py
```
Then open http://127.0.0.1:5000 in your browser

### Desktop Mode
Run as a desktop application:
```bash
python main.py
```

## Project Structure

```
product-matching-system/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── database.py         # Database operations
│   ├── image_processing.py # Feature extraction
│   ├── similarity.py       # Similarity computation
│   ├── product_matching.py # Matching logic
│   ├── static/             # Frontend files
│   │   ├── index.html      # Main UI
│   │   ├── styles.css      # Styling
│   │   └── app.js          # Frontend logic
│   └── uploads/            # Uploaded images
├── main.py                 # Desktop app launcher
└── requirements.txt        # Python dependencies
```

## Features

- Upload product images with optional metadata (name, SKU, category)
- Find similar products using CLIP visual embeddings (GPU-accelerated)
- Manage historical product catalog
- Batch upload multiple products with CSV support
- Filter results by similarity threshold
- Cross-platform GPU support (AMD, NVIDIA, Intel, Apple Silicon)
- No complex build process - just Python!

## Building Executable (Optional)

To create a standalone executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "backend/static;backend/static" main.py
```

The executable will be in the `dist/` folder.
