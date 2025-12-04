#!/usr/bin/env python3
"""
Pre-download CLIP Model
Downloads the CLIP model to cache before first run
"""

import sys
import os
from pathlib import Path

def download_clip_model():
    """Download CLIP model to cache"""
    print("="*80)
    print("Downloading CLIP Model")
    print("="*80)
    
    # Set cache directory
    cache_dir = Path.home() / '.cache' / 'clip-models'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    
    print(f"\nCache directory: {cache_dir}")
    
    # Check if already downloaded
    model_name = 'clip-ViT-B-32'
    model_path = cache_dir / f'models--sentence-transformers--{model_name}'
    
    if model_path.exists():
        print(f"\n✓ Model already downloaded: {model_name}")
        print(f"  Location: {model_path}")
        return True
    
    print(f"\nDownloading {model_name} (~350 MB)...")
    print("This may take 1-2 minutes depending on your internet speed...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download model
        print("\nInitializing download...")
        model = SentenceTransformer(model_name)
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"  Location: {cache_dir}")
        print(f"  Model: {model_name}")
        
        # Verify
        if model_path.exists():
            print("\n✓ Verification passed - model is cached")
        else:
            print("\n⚠ Warning: Model downloaded but cache location unexpected")
            print(f"  Expected: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try again later")
        print("3. The model will auto-download on first app launch")
        return False


if __name__ == "__main__":
    print("\nCLIP Model Pre-Download Script")
    print("This will download the AI model used for image matching")
    print()
    
    success = download_clip_model()
    
    if success:
        print("\n" + "="*80)
        print("Download Complete!")
        print("="*80)
        print("\nThe CLIP model is now cached and ready to use.")
        print("The app will launch instantly without downloading.")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("Download Failed")
        print("="*80)
        print("\nDon't worry - the app will still work!")
        print("The model will download automatically on first launch.")
        sys.exit(1)
