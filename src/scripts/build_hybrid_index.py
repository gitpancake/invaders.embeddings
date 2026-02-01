#!/usr/bin/env python3
"""Build a hybrid index using DINOv2 embeddings and perceptual hashes."""

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encoder import DINOv2Encoder, PerceptualHasher
from src.index import HybridIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INVADERS_JSON = Path(__file__).parent.parent.parent.parent / "flashcastr/public/data/json/invaders.json"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_PATH = DATA_DIR / "hybrid_index"


async def download_image(session, url, timeout=30):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        return None


async def main():
    logger.info(f"Loading flash data from {INVADERS_JSON}")
    with open(INVADERS_JSON, "r") as f:
        flashes = json.load(f)

    logger.info(f"Loaded {len(flashes)} flashes")

    logger.info("Initializing DINOv2 encoder...")
    encoder = DINOv2Encoder()
    
    logger.info("Initializing perceptual hasher...")
    hasher = PerceptualHasher()

    logger.info("Downloading images and generating embeddings + hashes...")

    all_embeddings = []
    valid_flashes = []
    flash_hashes = {}
    failed_count = 0

    async with aiohttp.ClientSession() as session:
        batch_size = 32

        for i in tqdm(range(0, len(flashes), batch_size), desc="Processing"):
            batch = flashes[i:i + batch_size]

            tasks = [download_image(session, f["t"]) for f in batch]
            results = await asyncio.gather(*tasks)

            images = []
            batch_valid = []

            for flash, image_bytes in zip(batch, results):
                if image_bytes is not None:
                    try:
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        images.append(image)
                        batch_valid.append(flash)
                        
                        # Compute perceptual hashes
                        flash_hashes[flash["i"]] = hasher.compute_hashes(image)
                    except Exception as e:
                        failed_count += 1
                else:
                    failed_count += 1

            if images:
                embeddings = encoder.encode_images(images)
                all_embeddings.append(embeddings)
                valid_flashes.extend(batch_valid)

    logger.info("Building hybrid index...")
    all_embeddings = np.vstack(all_embeddings)

    logger.info(f"Generated {len(all_embeddings)} embeddings")
    logger.info(f"Generated {len(flash_hashes)} hash sets")
    logger.info(f"Failed downloads: {failed_count}")

    index_manager = HybridIndexManager(dimension=encoder.EMBEDDING_DIM)
    index_manager.build_index(all_embeddings, valid_flashes, flash_hashes)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index_manager.save(str(INDEX_PATH))

    logger.info(f"Hybrid index saved to {INDEX_PATH}")
    logger.info(f"Total flashes indexed: {index_manager.size}")


if __name__ == "__main__":
    asyncio.run(main())
