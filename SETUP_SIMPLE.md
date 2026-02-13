# Simple Setup Guide

## Prerequisites

- **Python 3.8 - 3.12** (Python 3.13+ is NOT supported)
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

### Remote Mobile Access (ngrok, auto-start)

For Desktop Mode, ngrok now auto-starts in the background when the app launches (best effort).

One-time setup on host machine:
```bash
ngrok config add-authtoken <YOUR_TOKEN>
```

Port mapping:
- App server: `127.0.0.1:8000`
- ngrok local API/status: `127.0.0.1:4040`
- Public HTTPS ngrok URL forwards to `127.0.0.1:8000`

In the app:
1. Open `CONNECT PHONE`
2. Click `AUTO NGROK`
3. Share remote URL + PIN

Optional: disable ngrok auto-start
```bash
AUTO_START_NGROK=false python main.py
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
