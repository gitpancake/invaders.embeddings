#!/usr/bin/env python3
"""
Test the flash identification system.

Downloads a random flash image from the reference set and tests
if the system can correctly identify it.
"""

import asyncio
import aiohttp
import json
import sys
import random
from pathlib import Path
from io import BytesIO
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encoder import CLIPEncoder
from src.index import FAISSIndexManager

# Paths
INVADERS_JSON = Path(__file__).parent.parent.parent.parent / "flashcastr/public/data/json/invaders.json"
INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "flash_index"


async def download_image(url: str) -> bytes:
    """Download an image from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            return await response.read()


async def main():
    # Load flash data
    print(f"Loading flash data from {INVADERS_JSON}")
    with open(INVADERS_JSON, "r") as f:
        flashes = json.load(f)
    
    print(f"Loaded {len(flashes)} flashes")
    
    # Pick a random flash to test
    test_flash = random.choice(flashes)
    print(f"\nTest flash: {test_flash['n']} (ID: {test_flash['i']})")
    print(f"Image URL: {test_flash['t']}")
    
    # Download the test image
    print("\nDownloading test image...")
    image_bytes = await download_image(test_flash['t'])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Initialize encoder and index
    print("\nLoading CLIP encoder...")
    encoder = CLIPEncoder()
    
    print(f"Loading index from {INDEX_PATH}...")
    index_manager = FAISSIndexManager()
    index_manager.load(str(INDEX_PATH))
    print(f"Index contains {index_manager.size} flashes")
    
    # Generate embedding and search
    print("\nGenerating embedding...")
    import time
    start = time.time()
    embedding = encoder.encode_image(image)
    embed_time = (time.time() - start) * 1000
    print(f"Embedding generated in {embed_time:.1f}ms")
    
    # Search
    start = time.time()
    results = index_manager.search(embedding, top_k=5)
    search_time = (time.time() - start) * 1000
    print(f"Search completed in {search_time:.1f}ms")
    
    # Print results
    print("\n" + "="*60)
    print("TOP 5 MATCHES:")
    print("="*60)
    
    for i, match in enumerate(results, 1):
        correct = "✓" if match["flash_id"] == test_flash["i"] else ""
        print(f"{i}. {match['flash_name']} (ID: {match['flash_id']}) - "
              f"Similarity: {match['similarity']:.4f}, "
              f"Confidence: {match['confidence']:.2%} {correct}")
    
    # Check if correct
    if results and results[0]["flash_id"] == test_flash["i"]:
        print("\n✅ SUCCESS: Correct flash identified as top match!")
    else:
        print(f"\n❌ FAILED: Expected {test_flash['n']} but got {results[0]['flash_name']}")


if __name__ == "__main__":
    asyncio.run(main())
