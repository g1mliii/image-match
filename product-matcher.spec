# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

home_dir = str(Path.home())
CLIP_CACHE_PATH = os.path.join(home_dir, '.cache', 'clip-models')

ngrok_binaries = []
if os.name == 'nt':
    ngrok_win = os.path.join('third_party', 'ngrok', 'windows', 'ngrok.exe')
    if os.path.exists(ngrok_win):
        ngrok_binaries.append((ngrok_win, os.path.join('third_party', 'ngrok', 'windows')))
elif os.name == 'posix':
    if sys.platform == 'darwin':
        ngrok_macos = os.path.join('third_party', 'ngrok', 'macos', 'ngrok')
        if os.path.exists(ngrok_macos):
            ngrok_binaries.append((ngrok_macos, os.path.join('third_party', 'ngrok', 'macos')))
    else:
        ngrok_linux = os.path.join('third_party', 'ngrok', 'linux', 'ngrok')
        if os.path.exists(ngrok_linux):
            ngrok_binaries.append((ngrok_linux, os.path.join('third_party', 'ngrok', 'linux')))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=ngrok_binaries,
    datas=[
        ('backend/static', 'backend/static'),
        # CRITICAL: Bundle CLIP model (~350MB) - downloaded at ~/.cache/clip-models/
        # Run first: python scripts/download_clip_model.py
        # This ensures instant matching on first app run (no model download needed)
        (os.path.expanduser('~/.cache/clip-models'), '.cache/clip-models'),
    ],
    hiddenimports=[
        # Backend modules
        'backend.app',
        'backend.database',
        'backend.config',
        'backend.similarity',
        'backend.image_processing',
        'backend.image_processing_clip',
        'backend.product_matching',
        'backend.hybrid_matching',
        'backend.feature_extraction_service',
        'backend.feature_cache',
        'backend.faiss_index',
        'backend.snapshot_manager',
        'backend.matching_utils',
        'backend.validation_utils',
        'backend.path_manager',

        # Web Framework (from requirements.txt)
        'flask',
        'flask_cors',
        'werkzeug',
        'urllib3',

        # Core Scientific (from requirements.txt)
        'numpy',
        'scipy',
        'scipy.sparse',
        'scipy.spatial',

        # Image Processing (from requirements.txt)
        'cv2',
        'skimage',
        'PIL',
        'PIL.Image',

        # Deep Learning (from requirements.txt)
        'torch',
        'torch._C',
        'torch.distributed',
        'torchvision',
        'torchaudio',

        # CLIP Model (from requirements.txt - CRITICAL: <3.0.0)
        'sentence_transformers',
        'sentence_transformers.models',
        'transformers',

        # Fast Search (from requirements.txt)
        'faiss',

        # Desktop (from requirements.txt)
        'webview',

        # Utilities (from requirements.txt)
        'psutil',
        'fuzzywuzzy',
        'Levenshtein',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['pytest', 'unittest', 'test', 'tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ProductMatcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ProductMatcher'
)
