#!/usr/bin/env python3
"""
Setup script to pre-compute main category embeddings
Run this once during deployment to avoid first-interaction delays
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def main():
    """Pre-compute and save main category embeddings"""
    try:
        from core.llm_updater import precompute_main_category_embeddings
        
        print("🚀 Setting up main category embeddings...")
        print("This will pre-compute embeddings for all academic categories.")
        print("This only needs to be done once during setup/deployment.")
        print()
        
        # Pre-compute embeddings
        embeddings = precompute_main_category_embeddings()
        
        print()
        print("✅ Setup complete!")
        print(f"📁 Embeddings saved to: main_category_embeddings.pkl")
        print(f"📊 Total categories processed: {len(embeddings)}")
        print()
        print("🎯 Benefits:")
        print("  • First user interaction will be fast")
        print("  • No delay from computing embeddings")
        print("  • Consistent performance across all requests")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory")
        return 1
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 