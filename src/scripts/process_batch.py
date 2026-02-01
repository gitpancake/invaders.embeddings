#!/usr/bin/env python3
"""
Batch process IPFS images to identify Space Invader flashes.

Reads a CSV of IPFS hashes, downloads each image, identifies the flash,
and outputs the results to a CSV file.

Usage:
    python -m src.scripts.process_batch --input ipfs_hashes.csv --output results.csv
"""

import asyncio
import aiohttp
import csv
import json
import logging
import sys
import time
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import click

# Enable AVIF support
try:
    import pillow_avif
except ImportError:
    pass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encoder import CLIPEncoder
from src.index import FAISSIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "flash_index"

# IPFS Gateway - can be configured
DEFAULT_IPFS_GATEWAY = "https://gateway.pinata.cloud/ipfs/"


class BatchProcessor:
    """Process batches of IPFS images for flash identification."""
    
    def __init__(
        self,
        ipfs_gateway: str = DEFAULT_IPFS_GATEWAY,
        concurrency: int = 5,
        timeout: int = 30,
        top_k: int = 1,
    ):
        self.ipfs_gateway = ipfs_gateway
        self.concurrency = concurrency
        self.timeout = timeout
        self.top_k = top_k
        
        # Load encoder and index
        logger.info("Loading CLIP encoder...")
        self.encoder = CLIPEncoder()
        
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        self.index = FAISSIndexManager()
        self.index.load(str(INDEX_PATH))
        logger.info(f"Index contains {self.index.size} flashes")
        
        # Stats
        self.processed = 0
        self.failed = 0
        self.start_time = None
    
    async def download_image(
        self,
        session: aiohttp.ClientSession,
        ipfs_hash: str,
    ) -> bytes | None:
        """Download an image from IPFS."""
        url = f"{self.ipfs_gateway}{ipfs_hash}"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.warning(f"HTTP {response.status} for {ipfs_hash}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to download {ipfs_hash}: {e}")
            return None
    
    def identify_image(self, image_bytes: bytes) -> dict | None:
        """Identify a flash from image bytes."""
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            embedding = self.encoder.encode_image(image)
            matches = self.index.search(embedding, top_k=self.top_k)
            
            if matches:
                return {
                    "flash_id": matches[0]["flash_id"],
                    "flash_name": matches[0]["flash_name"],
                    "similarity": matches[0]["similarity"],
                    "confidence": matches[0]["confidence"],
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to identify image: {e}")
            return None
    
    async def process_batch(
        self,
        ipfs_hashes: list[str],
        progress_callback=None,
    ) -> list[dict]:
        """Process a batch of IPFS hashes."""
        results = []
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_one(ipfs_hash: str) -> dict:
            async with semaphore:
                result = {
                    "ipfs_hash": ipfs_hash,
                    "flash_id": None,
                    "flash_name": None,
                    "similarity": None,
                    "confidence": None,
                    "status": "failed",
                }
                
                async with aiohttp.ClientSession() as session:
                    image_bytes = await self.download_image(session, ipfs_hash)
                
                if image_bytes:
                    match = self.identify_image(image_bytes)
                    if match:
                        result.update(match)
                        result["status"] = "success"
                        self.processed += 1
                    else:
                        self.failed += 1
                else:
                    self.failed += 1
                
                if progress_callback:
                    progress_callback()
                
                return result
        
        tasks = [process_one(h) for h in ipfs_hashes]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def process_file(
        self,
        input_file: Path,
        output_file: Path,
        checkpoint_file: Path | None = None,
        batch_size: int = 100,
    ):
        """Process a CSV file of IPFS hashes."""
        self.start_time = time.time()
        
        # Load input
        logger.info(f"Loading IPFS hashes from {input_file}...")
        with open(input_file, "r") as f:
            reader = csv.DictReader(f)
            # Assume column name is 'ipfs_hash' or first column
            fieldnames = reader.fieldnames
            hash_column = "ipfs_hash" if "ipfs_hash" in fieldnames else fieldnames[0]
            all_hashes = [row[hash_column] for row in reader]
        
        logger.info(f"Loaded {len(all_hashes)} IPFS hashes")
        
        # Load checkpoint if exists
        processed_hashes = set()
        if checkpoint_file and checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                processed_hashes = set(json.load(f))
            logger.info(f"Resuming from checkpoint: {len(processed_hashes)} already processed")
        
        # Filter out already processed
        remaining = [h for h in all_hashes if h not in processed_hashes]
        logger.info(f"Processing {len(remaining)} remaining hashes")
        
        # Open output file
        write_header = not output_file.exists()
        
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ipfs_hash", "flash_id", "flash_name", "similarity", "confidence", "status"
            ])
            
            if write_header:
                writer.writeheader()
            
            # Process in batches
            with tqdm(total=len(remaining), desc="Processing") as pbar:
                for i in range(0, len(remaining), batch_size):
                    batch = remaining[i:i + batch_size]
                    
                    results = await self.process_batch(
                        batch,
                        progress_callback=lambda: pbar.update(1)
                    )
                    
                    # Write results
                    for result in results:
                        writer.writerow(result)
                        processed_hashes.add(result["ipfs_hash"])
                    
                    f.flush()
                    
                    # Save checkpoint
                    if checkpoint_file:
                        with open(checkpoint_file, "w") as cf:
                            json.dump(list(processed_hashes), cf)
        
        # Stats
        elapsed = time.time() - self.start_time
        rate = len(remaining) / elapsed if elapsed > 0 else 0
        
        logger.info(f"Completed processing {len(remaining)} images")
        logger.info(f"Successful: {self.processed}, Failed: {self.failed}")
        logger.info(f"Time: {elapsed:.1f}s, Rate: {rate:.1f} images/sec")


@click.command()
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True),
              help="Input CSV file with IPFS hashes")
@click.option("--output", "-o", "output_file", required=True, type=click.Path(),
              help="Output CSV file for results")
@click.option("--checkpoint", "-c", "checkpoint_file", type=click.Path(),
              help="Checkpoint file for resume capability")
@click.option("--gateway", "-g", default=DEFAULT_IPFS_GATEWAY,
              help="IPFS gateway URL")
@click.option("--concurrency", default=5, type=int,
              help="Number of concurrent downloads")
@click.option("--batch-size", default=100, type=int,
              help="Batch size for processing")
def main(input_file, output_file, checkpoint_file, gateway, concurrency, batch_size):
    """Process IPFS images to identify Space Invader flashes."""
    processor = BatchProcessor(
        ipfs_gateway=gateway,
        concurrency=concurrency,
    )
    
    asyncio.run(processor.process_file(
        Path(input_file),
        Path(output_file),
        Path(checkpoint_file) if checkpoint_file else None,
        batch_size=batch_size,
    ))


if __name__ == "__main__":
    main()
