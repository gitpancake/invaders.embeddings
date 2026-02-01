#!/usr/bin/env python3
"""
Build the FAISS index from reference flash images.

This downloads all reference images from invaders.json and creates
embeddings using CLIP, then builds a FAISS index for similarity search.
"""

import asyncio
import aiohttp
import json
import logging
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encoder import CLIPEncoder
from src.index import FAISSIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
INVADERS_JSON = Path(__file__).parent.parent.parent.parent / "flashcastr/public/data/json/invaders.json"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_PATH = DATA_DIR / "flash_index"


async def download_image(session, url, timeout=30):
    """Download an image from URL."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        return None


async def main():
    """Main function to build the index."""
    
    # Load flash data
    logger.info(f"Loading flash data from {INVADERS_JSON}")
    with open(INVADERS_JSON, "r") as f:
        flashes = json.load(f)
    
    logger.info(f"Loaded {len(flashes)} flashes")
    
    # Initialize encoder
    logger.info("Initializing CLIP encoder...")
    encoder = CLIPEncoder()
    
    # Download images and generate embeddings
    logger.info("Downloading images and generating embeddings...")
    
    all_embeddings = []
    valid_flashes = []
    failed_count = 0
    
    async with aiohttp.ClientSession() as session:
        # Process in batches
        batch_size = 32
        
        for i in tqdm(range(0, len(flashes), batch_size), desc="Processing"):
            batch = flashes[i:i + batch_size]
            
            # Download images concurrently
            tasks = [download_image(session, f["t"]) for f in batch]
            results = await asyncio.gather(*tasks)
            
            # Process successful downloads
            images = []
            batch_valid = []
            
            for flash, image_bytes in zip(batch, results):
                if image_bytes is not None:
                    try:
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        images.append(image)
                        batch_valid.append(flash)
                    except Exception as e:
                        failed_count += 1
                else:
                    failed_count += 1
            
            if images:
                embeddings = encoder.encode_images(images)
                all_embeddings.append(embeddings)
                valid_flashes.extend(batch_valid)
    
    # Combine all embeddings
    logger.info("Building FAISS index...")
    all_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Generated {len(all_embeddings)} embeddings")
    logger.info(f"Failed downloads: {failed_count}")
    
    # Build index
    index_manager = FAISSIndexManager()
    index_manager.build_index(all_embeddings, valid_flashes)
    
    # Save index
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index_manager.save(str(INDEX_PATH))
    
    logger.info(f"Index saved to {INDEX_PATH}")
    logger.info(f"Total flashes indexed: {index_manager.size}")


if __name__ == "__main__":
    asyncio.run(main())
