# Product Matching System

A desktop application for comparing new products against a historical catalog using image-based similarity analysis.

## Features

- Upload product images with category classification
- Visual similarity matching using color, shape, and texture features
- Category-based filtering
- Batch processing support
- Historical product catalog management
- Detailed comparison views with similarity breakdowns

## Technology Stack

- **Frontend**: Electron + React + TypeScript
- **Backend**: Python + Flask
- **Image Processing**: OpenCV, scikit-image
- **Database**: SQLite

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.9 or higher)
- npm or yarn

## Installation

1. Clone the repository
2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

## Development

Run the application in development mode:

```bash
npm run electron:dev
```

This will:
- Start the Python Flask backend on port 5000
- Launch the Electron application with hot reload

## Building

Build the application for production:

```bash
npm run build
npm run package
```

The packaged application will be in the `release/` directory.

## Project Structure

```
product-matching-system/
├── src/
│   ├── electron/          # Electron main process
│   │   ├── main.ts
│   │   └── preload.ts
│   ├── renderer/          # React UI
│   │   ├── App.tsx
│   │   ├── index.tsx
│   │   └── styles.css
│   └── types/             # TypeScript definitions
├── backend/               # Python Flask backend
│   ├── app.py            # Main Flask application
│   ├── database.py       # Database operations
│   ├── image_processing.py  # Feature extraction
│   ├── similarity.py     # Similarity computation
│   └── requirements.txt
├── package.json
├── tsconfig.json
└── webpack.config.js
```

## License

MIT
