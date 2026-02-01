"""Tile pattern re-ranking for flash identification.

Re-ranks CLIP embedding candidates using tile pattern similarity.
"""

import asyncio
import aiohttp
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional
import logging

try:
    import pillow_avif
except ImportError:
    pass

from .tile_extract import extract_tile_pattern_v2
from .grid_detect import preprocess_with_grid_detection

logger = logging.getLogger(__name__)


async def download_image(session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
    """Download an image from URL."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status == 200:
                data = await response.read()
                return Image.open(BytesIO(data)).convert('RGB')
            else:
                logger.debug(f'HTTP {response.status} for {url}')
    except Exception as e:
        logger.debug(f'Failed to download {url}: {e}')
    return None


def extract_pattern_safe(image: Image.Image, max_grid: int = 30) -> Optional[Dict]:
    """
    Safely extract tile pattern, handling errors.
    Uses grid_detect to isolate the mosaic first.
    """
    try:
        # Use grid detection to find and crop to mosaic
        cropped = preprocess_with_grid_detection(image)
        
        # Extract pattern from cropped image (no bounds detection needed)
        pattern = extract_tile_pattern_v2(cropped, find_bounds=False)

        # Validate pattern
        rows, cols = pattern['grid_shape']
        if rows < 3 or cols < 3:
            logger.debug(f'Grid too small: {rows}x{cols}')
            return None
        if rows > max_grid or cols > max_grid:
            logger.debug(f'Grid too large: {rows}x{cols}')
            return None

        return pattern
    except Exception as e:
        logger.debug(f'Pattern extraction failed: {e}')
        return None


def compute_pattern_similarity(query_pattern: Dict, ref_pattern: Dict) -> float:
    """
    Compute similarity between query and reference patterns.
    Returns a score from 0 to 1.
    """
    if query_pattern is None or ref_pattern is None:
        return 0.0

    q_rows, q_cols = query_pattern['grid_shape']
    r_rows, r_cols = ref_pattern['grid_shape']

    # Grid dimension similarity
    row_ratio = min(q_rows, r_rows) / max(q_rows, r_rows)
    col_ratio = min(q_cols, r_cols) / max(q_cols, r_cols)
    dim_sim = row_ratio * col_ratio

    # If dimensions are very different, low similarity
    if dim_sim < 0.5:
        return dim_sim * 0.3

    q_pat = query_pattern['pattern']
    r_pat = ref_pattern['pattern']

    # Center-aligned comparison
    q_center_r, q_center_c = q_rows // 2, q_cols // 2
    r_center_r, r_center_c = r_rows // 2, r_cols // 2

    window_r = min(q_rows, r_rows) // 2
    window_c = min(q_cols, r_cols) // 2

    matches = 0
    total = 0

    for dr in range(-window_r, window_r + 1):
        for dc in range(-window_c, window_c + 1):
            qr, qc = q_center_r + dr, q_center_c + dc
            rr, rc = r_center_r + dr, r_center_c + dc

            if 0 <= qr < q_rows and 0 <= qc < q_cols and 0 <= rr < r_rows and 0 <= rc < r_cols:
                total += 1
                q_color = q_pat[qr][qc]
                r_color = r_pat[rr][rc]

                if q_color == r_color:
                    matches += 1
                elif are_similar_colors(q_color, r_color):
                    matches += 0.5

    color_sim = matches / total if total > 0 else 0

    return dim_sim * 0.3 + color_sim * 0.7


def are_similar_colors(c1: str, c2: str) -> bool:
    """Check if two color names are similar."""
    dark_colors = {'black', 'navy', 'brown', 'gray'}
    light_colors = {'white', 'beige', 'light_blue', 'pink'}
    warm_colors = {'red', 'orange', 'yellow'}
    cool_colors = {'blue', 'cyan', 'green', 'lime', 'purple'}

    for group in [dark_colors, light_colors, warm_colors, cool_colors]:
        if c1 in group and c2 in group:
            return True
    return False


async def rerank_with_tiles(
    query_image: Image.Image,
    candidates: List[Dict],
    top_k: int = 10,
    tile_weight: float = 0.4,
    max_concurrent: int = 10
) -> List[Dict]:
    """
    Re-rank candidates using tile pattern matching.
    """
    logger.info('Extracting query tile pattern...')
    query_pattern = extract_pattern_safe(query_image)

    if query_pattern is None:
        logger.warning('Could not extract query pattern, returning original ranking')
        return candidates[:top_k]

    logger.info(f'Query pattern: {query_pattern["grid_shape"]} tiles')

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_candidate(session: aiohttp.ClientSession, candidate: Dict) -> Dict:
        async with semaphore:
            result = candidate.copy()
            result['tile_similarity'] = 0.0
            result['ref_grid_shape'] = None

            url = candidate.get('image_url', '')
            if not url:
                return result

            ref_image = await download_image(session, url)
            if ref_image is None:
                return result

            # Use same grid_detect for reference images
            ref_pattern = extract_pattern_safe(ref_image)

            if ref_pattern is not None:
                result['tile_similarity'] = compute_pattern_similarity(query_pattern, ref_pattern)
                result['ref_grid_shape'] = ref_pattern['grid_shape']
                logger.debug(f'{candidate.get("flash_name")}: grid={ref_pattern["grid_shape"]}, tile_sim={result["tile_similarity"]:.3f}')

            return result

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_candidate(session, c) for c in candidates]
        results = await asyncio.gather(*tasks)

    # Compute combined score
    for r in results:
        emb_sim = r.get('similarity', 0)
        tile_sim = r.get('tile_similarity', 0)
        r['combined_score'] = (1 - tile_weight) * emb_sim + tile_weight * tile_sim

    results.sort(key=lambda x: x['combined_score'], reverse=True)

    logger.info('Re-ranking complete:')
    for i, r in enumerate(results[:top_k], 1):
        logger.info(f'  {i}. {r["flash_name"]} - combined: {r["combined_score"]:.4f} '
                   f'(emb: {r["similarity"]:.4f}, tile: {r["tile_similarity"]:.4f}, '
                   f'grid: {r.get("ref_grid_shape")})')

    return results[:top_k]
