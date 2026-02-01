"""Template matching for accurate flash verification."""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
import aiohttp
import asyncio
from io import BytesIO

try:
    import pillow_avif
except ImportError:
    pass

logger = logging.getLogger(__name__)


def extract_mosaic_features(image: Image.Image, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    img = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img)
    
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang[ang < 0] += 180
    
    block_size = size // 4
    n_bins = 9
    grid_features = []
    
    for by in range(4):
        for bx in range(4):
            y1, y2 = by * block_size, (by + 1) * block_size
            x1, x2 = bx * block_size, (bx + 1) * block_size
            
            block_mag = mag[y1:y2, x1:x2].flatten()
            block_ang = ang[y1:y2, x1:x2].flatten()
            
            hist, _ = np.histogram(block_ang, bins=n_bins, range=(0, 180), weights=block_mag)
            hist = hist / (hist.sum() + 1e-10)
            grid_features.append(hist)
    
    grid_features = np.concatenate(grid_features)
    
    pixels = arr.reshape(-1, 3)
    quantized = (pixels // 64).astype(np.uint8)
    color_ids = quantized[:, 0] * 16 + quantized[:, 1] * 4 + quantized[:, 2]
    color_hist, _ = np.histogram(color_ids, bins=64, range=(0, 64))
    color_features = color_hist.astype(np.float32) / color_hist.sum()
    
    return grid_features.astype(np.float32), color_features


def compute_feature_similarity(
    query_grid: np.ndarray, 
    query_color: np.ndarray,
    ref_grid: np.ndarray, 
    ref_color: np.ndarray,
    grid_weight: float = 0.5
) -> Tuple[float, float, float]:
    grid_sim = np.sum(np.minimum(query_grid, ref_grid))
    color_sim = np.sum(np.minimum(query_color, ref_color))
    combined = grid_weight * grid_sim + (1 - grid_weight) * color_sim
    return combined, grid_sim, color_sim


async def download_image_with_retry(session: aiohttp.ClientSession, url: str, retries: int = 2) -> Optional[Image.Image]:
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.read()
                    return Image.open(BytesIO(data)).convert('RGB')
        except asyncio.TimeoutError:
            if attempt < retries:
                await asyncio.sleep(0.5)
                continue
        except Exception as e:
            logger.debug(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries:
                await asyncio.sleep(0.5)
                continue
    return None


async def rerank_with_template_matching(
    query_image: Image.Image,
    candidates: List[Dict],
    top_k: int = 5,
    grid_weight: float = 0.5,
    max_concurrent: int = 10
) -> List[Dict]:
    query_grid, query_color = extract_mosaic_features(query_image)
    
    # Use semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_with_semaphore(session, url):
        async with semaphore:
            return await download_image_with_retry(session, url)
    
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_with_semaphore(session, c.get('image_url', '')) for c in candidates]
        ref_images = await asyncio.gather(*tasks)
    
    results = []
    downloaded = 0
    failed = 0
    
    for candidate, ref_image in zip(candidates, ref_images):
        if ref_image is None:
            failed += 1
            continue
        
        downloaded += 1
        ref_grid, ref_color = extract_mosaic_features(ref_image)
        combined, grid_sim, color_sim = compute_feature_similarity(
            query_grid, query_color, ref_grid, ref_color, grid_weight
        )
        
        result = candidate.copy()
        result['template_similarity'] = combined
        result['structure_similarity'] = grid_sim
        result['color_similarity'] = color_sim
        results.append(result)
    
    logger.info(f"Template matching: {downloaded} downloaded, {failed} failed")
    results.sort(key=lambda x: x['template_similarity'], reverse=True)
    return results[:top_k]


def rerank_sync(query_image: Image.Image, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    return asyncio.run(rerank_with_template_matching(query_image, candidates, top_k))
