# Setup Instructions

## Initial Setup

Follow these steps to set up the Product Matching System development environment:

### 1. Install Node.js Dependencies

```bash
npm install
```

This will install all required packages including:
- Electron and Electron Builder
- React and React Router
- TypeScript and build tools
- Webpack and loaders

### 2. Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

Or if using a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r backend/requirements.txt
```

This will install:
- Flask and Flask-CORS
- NumPy and SciPy
- OpenCV and scikit-image
- Pillow

### 3. Initialize Database

The database will be automatically initialized when you first run the backend, but you can verify it manually:

```bash
python backend/test_db.py
```

Expected output:
```
Initializing database...
Database initialized successfully

Verifying tables...
Tables created: ['products', 'sqlite_sequence', 'features', 'matches']
Indexes created: ['idx_products_category', 'idx_products_is_historical', 'idx_matches_new_product']

Database setup complete!
```

### 4. Build Electron Main Process

```bash
npm run build:electron
```

This compiles the TypeScript files for the Electron main process.

### 5. Run Development Environment

```bash
npm run electron:dev
```

This will:
1. Build the Electron main process
2. Start the Python Flask backend on http://localhost:5000
3. Launch the Electron application

## Verification Checklist

- [ ] Node.js dependencies installed (`node_modules/` exists)
- [ ] Python dependencies installed (test with `python -c "import flask, cv2, numpy"`)
- [ ] Database initialized (`backend/product_matching.db` exists)
- [ ] Electron main process builds without errors
- [ ] Backend starts and responds at http://localhost:5000/api/health
- [ ] Electron application launches successfully

## Troubleshooting

### Python Backend Won't Start

- Ensure Python 3.9+ is installed: `python --version`
- Check all dependencies are installed: `pip list`
- Try running backend directly: `python backend/app.py`

### Electron Won't Launch

- Ensure TypeScript compiled: `npm run build:electron`
- Check for errors in `dist/electron/` directory
- Verify Node.js version: `node --version` (should be 18+)

### Database Errors

- Delete `backend/product_matching.db` and run `python backend/test_db.py` again
- Check file permissions on the backend directory

## Next Steps

After successful setup, you can:
1. Start implementing UI components in `src/renderer/`
2. Add API endpoints in `backend/app.py`
3. Implement image processing features in `backend/image_processing.py`
4. Test the application with sample product images
