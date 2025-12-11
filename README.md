# CatalogMatch

**Smart Product Comparison for Inventory Management**

CatalogMatch is a desktop application that helps businesses compare new products against their existing inventory using AI-powered visual similarity analysis. Make smarter purchasing and inventory decisions by finding visually similar items in your catalog.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-Proprietary-red)

## Website

Visit our website: **https://g1mliii.github.io/image-match/**

## Features

- **Visual Similarity Matching** - Compare products based on color, shape, and texture
- **Category Filtering** - Match products only within the same category
- **Batch Processing** - Process up to 100 products at once
- **Offline Desktop App** - No internet required, your data stays private
- **CSV Import/Export** - Easy integration with your existing systems
- **Detailed Scoring** - See similarity breakdowns by feature type
- **Side-by-Side Comparison** - Visual comparison of matched products

## Quick Start

1. **Download** the app from [our website](https://g1mliii.github.io/image-match/download.html)
2. **Install** by running the executable (Windows only)
3. **Upload** your historical product catalog
4. **Match** new products against your catalog
5. **Review** results and export to CSV

## Documentation

### Getting Started
- **[INSTALLATION.md](INSTALLATION.md)** - Complete installation guide for all platforms
- **[START_SERVER_README.md](START_SERVER_README.md)** - How to start the server
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference guide

### User Guides
- **[Website Documentation](https://g1mliii.github.io/image-match/docs.html)** - Online user guide
- **[GPU Setup Guide](gpu/GPU_SETUP_GUIDE.md)** - GPU acceleration setup (AMD/NVIDIA/Intel/Apple Silicon)

### Developer Documentation
- **[CLIP Developer Guide](backend/docs/CLIP_DEVELOPER_GUIDE.md)** - Technical guide for CLIP integration
- **[Database Design](backend/docs/DATABASE_DESIGN.md)** - Database schema and design
- **[Matching Service](backend/docs/MATCHING_SERVICE.md)** - Matching algorithm details

## System Requirements

### Windows
- Windows 10 or later (64-bit)
- 4 GB RAM minimum (8 GB recommended)
- 500 MB free disk space (1 GB with CLIP model)
- Display resolution: 1280x720 or higher

### GPU Acceleration (Optional but Recommended)
- **AMD GPU**: Radeon RX 6000/7000/9000 series + Python 3.12 + ROCm 6.4
- **NVIDIA GPU**: GeForce/RTX series (any recent GPU)
- **Intel GPU**: Arc/Iris Xe/UHD (11th gen+) + Intel Extension
- **Apple Silicon**: M1/M2/M3/M4/M5 (built-in MPS support)

**Performance**: 
- AMD/NVIDIA GPU: 150-300 img/s (fastest)
- Intel GPU: 30-80 img/s (3-5x faster than CPU)
- CPU only: 5-20 img/s (slowest but works everywhere)

**Setup GPU**: See [INSTALLATION.md](INSTALLATION.md) or [gpu/GPU_SETUP_GUIDE.md](gpu/GPU_SETUP_GUIDE.md)

**Quick GPU Setup**:
```bash
cd gpu
python setup_gpu.py
```

## Pricing

- **Free**: Up to 50 products
- **Pro**: $29 one-time payment for unlimited products

[View full pricing details](https://g1mliii.github.io/image-match/pricing.html)

## More Information

Full documentation is available at: https://g1mliii.github.io/image-match/docs.html

## Technology Stack

- **Backend**: Python, Flask, OpenCV, NumPy
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Database**: SQLite
- **Desktop**: PyWebView

## What's Included

- Visual similarity engine with color, shape, and texture analysis
- Category-based filtering system
- Batch upload and processing
- CSV metadata import/export
- Local SQLite database
- Desktop GUI application

## Privacy & Security

- **100% Offline** - All processing happens on your computer
- **No Cloud Storage** - Your data never leaves your machine
- **Local Database** - SQLite database stored in your app data folder
- **No Tracking** - We don't collect any usage data

## Support

- **Documentation**: https://g1mliii.github.io/image-match/docs.html
- **Bug Reports**: https://github.com/g1mliii/image-match/issues
- **Business Inquiries**: info@anchored.site

## Roadmap

- [ ] macOS support
- [ ] Price history tracking
- [ ] Advanced filtering options
- [ ] Duplicate detection reports
- [ ] GPU acceleration
- [ ] API access for Pro users

## License

Proprietary software. See [Terms of Service](https://g1mliii.github.io/image-match/terms.html) for details.

- **Free Version**: Personal use only, up to 50 products
- **Pro Version**: Commercial use allowed, unlimited products

## About

CatalogMatch is developed to help businesses make smarter inventory decisions through visual product comparison. Whether you're managing e-commerce inventory, retail stock, or product catalogs, CatalogMatch helps you quickly identify similar items and avoid duplicate purchases.

---

**© 2024 CatalogMatch. All rights reserved.**

[Website](https://g1mliii.github.io/image-match/) • [Download](https://g1mliii.github.io/image-match/download.html) • [Pricing](https://g1mliii.github.io/image-match/pricing.html) • [Docs](https://g1mliii.github.io/image-match/docs.html)
