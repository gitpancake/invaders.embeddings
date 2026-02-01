"""Improved tile pattern extraction from Space Invader mosaics.

Handles both clean reference images and real-world photos by:
1. Finding mosaic bounds using edge density (optional)
2. Grid detection using autocorrelation
3. Color mapping to standard palette
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Standard Space Invader tile colors
STANDARD_PALETTE = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 200, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'orange': (255, 165, 0),
    'pink': (255, 150, 180),
    'purple': (128, 0, 128),
    'navy': (0, 0, 128),
    'gray': (128, 128, 128),
    'lime': (180, 255, 0),
    'brown': (139, 90, 43),
    'beige': (210, 200, 190),
    'light_blue': (135, 206, 235),
}


def find_mosaic_bounds_edge(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find mosaic bounds using edge density.
    Returns (x, y, width, height).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find edges using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    
    # Sum edges along rows and columns
    row_edges = edges.sum(axis=1)
    col_edges = edges.sum(axis=0)
    
    # Find where edges are concentrated
    row_threshold = row_edges.max() * 0.15
    col_threshold = col_edges.max() * 0.15
    
    row_active = row_edges > row_threshold
    col_active = col_edges > col_threshold
    
    # Find bounds
    y1 = int(np.argmax(row_active))
    y2 = int(len(row_active) - np.argmax(row_active[::-1]))
    x1 = int(np.argmax(col_active))
    x2 = int(len(col_active) - np.argmax(col_active[::-1]))
    
    # Ensure valid bounds
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    
    # Add small margin
    margin = 2
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    return x1, y1, x2 - x1, y2 - y1


def find_tile_size(image: np.ndarray, min_tile: int = 15, max_tile: int = 80) -> int:
    """
    Find tile size using autocorrelation on edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape

    # Autocorrelation
    f = np.fft.fft2(edges.astype(np.float32))
    autocorr = np.abs(np.fft.ifft2(f * np.conj(f)))
    autocorr = np.fft.fftshift(autocorr)
    autocorr = autocorr / (autocorr.max() + 1e-10)

    cy, cx = h // 2, w // 2

    # Average horizontal and vertical profiles
    max_range = min(max_tile + 10, min(h, w) // 2)
    h_profile = autocorr[cy, cx:cx+max_range]
    v_profile = autocorr[cy:cy+max_range, cx]
    profile = (h_profile + v_profile) / 2

    # Find the strongest peak in valid range
    best_tile = min_tile
    best_score = 0

    for i in range(min_tile, min(max_tile, len(profile) - 2)):
        if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
            # Prefer larger tiles (more likely to be actual tile size)
            harmonic = profile[i*2] if i*2 < len(profile) else 0
            score = profile[i] * (1 + harmonic) * (i / min_tile) ** 0.3

            if score > best_score:
                best_score = score
                best_tile = i

    return best_tile


def map_to_palette(color: np.ndarray, palette: dict = STANDARD_PALETTE) -> Tuple[str, Tuple[int, int, int]]:
    """Map a color to the nearest palette color."""
    min_dist = float('inf')
    best_name = 'gray'
    best_color = (128, 128, 128)

    for name, pal_color in palette.items():
        dist = np.sqrt(np.sum((color.astype(float) - np.array(pal_color)) ** 2))
        if dist < min_dist:
            min_dist = dist
            best_name = name
            best_color = pal_color

    return best_name, best_color


def extract_tile_pattern_v2(image: Image.Image, find_bounds: bool = True) -> Dict:
    """
    Extract tile pattern from mosaic image.

    Args:
        image: PIL Image
        find_bounds: If True, detect mosaic bounds within image. 
                     Set False if image is already cropped to mosaic.

    Returns:
        dict with tile_size, grid_shape, colors, pattern, signature
    """
    arr = np.array(image.convert('RGB'))
    h, w = arr.shape[:2]

    # Optionally find mosaic bounds
    if find_bounds:
        x, y, mw, mh = find_mosaic_bounds_edge(arr)
        logger.info(f'Mosaic bounds: ({x}, {y}, {mw}, {mh})')
        mosaic_arr = arr[y:y+mh, x:x+mw]
    else:
        x, y, mw, mh = 0, 0, w, h
        mosaic_arr = arr

    # Find tile size
    tile_size = find_tile_size(mosaic_arr)
    logger.info(f'Tile size: {tile_size} pixels')

    # Calculate grid dimensions
    n_rows = max(1, round(mh / tile_size))
    n_cols = max(1, round(mw / tile_size))
    logger.info(f'Grid: {n_rows} x {n_cols}')

    # Sample colors at tile centers
    actual_tile_h = mh / n_rows
    actual_tile_w = mw / n_cols

    colors = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    pattern = []

    for row in range(n_rows):
        row_pattern = []
        for col in range(n_cols):
            cy = int((row + 0.5) * actual_tile_h)
            cx = int((col + 0.5) * actual_tile_w)

            # Sample small region around center
            sample_r = max(1, int(min(actual_tile_h, actual_tile_w) * 0.25))
            y1 = max(0, cy - sample_r)
            y2 = min(mh, cy + sample_r + 1)
            x1 = max(0, cx - sample_r)
            x2 = min(mw, cx + sample_r + 1)

            region = mosaic_arr[y1:y2, x1:x2]
            color = np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)
            colors[row, col] = color

            # Map to palette
            name, _ = map_to_palette(color)
            row_pattern.append(name)

        pattern.append(row_pattern)

    # Create signature
    signature = '|'.join([''.join(n[0].upper() for n in row) for row in pattern])

    return {
        'tile_size': tile_size,
        'grid_shape': (n_rows, n_cols),
        'bounds': (x, y, mw, mh),
        'colors': colors,
        'pattern': pattern,
        'signature': signature,
    }


def pattern_to_image(pattern_data: Dict, tile_px: int = 20) -> Image.Image:
    """Render pattern as an image."""
    pattern = pattern_data['pattern']
    n_rows = len(pattern)
    n_cols = len(pattern[0]) if pattern else 0

    img = np.zeros((n_rows * tile_px, n_cols * tile_px, 3), dtype=np.uint8)

    for row in range(n_rows):
        for col in range(n_cols):
            name = pattern[row][col]
            color = STANDARD_PALETTE.get(name, (128, 128, 128))

            y1, y2 = row * tile_px, (row + 1) * tile_px
            x1, x2 = col * tile_px, (col + 1) * tile_px
            img[y1:y2, x1:x2] = color

    return Image.fromarray(img)


def compare_patterns(p1: Dict, p2: Dict) -> float:
    """Compare two patterns, return similarity 0-1."""
    pat1 = p1['pattern']
    pat2 = p2['pattern']

    rows1, cols1 = len(pat1), len(pat1[0]) if pat1 else 0
    rows2, cols2 = len(pat2), len(pat2[0]) if pat2 else 0

    # Penalize dimension mismatch
    if rows1 != rows2 or cols1 != cols2:
        dim_sim = min(rows1, rows2) / max(rows1, rows2) * min(cols1, cols2) / max(cols1, cols2)
    else:
        dim_sim = 1.0

    # Compare at smaller dimensions
    min_rows = min(rows1, rows2)
    min_cols = min(cols1, cols2)

    matches = sum(
        1 for r in range(min_rows) for c in range(min_cols)
        if pat1[r][c] == pat2[r][c]
    )

    total = min_rows * min_cols
    tile_sim = matches / total if total > 0 else 0

    return tile_sim * dim_sim
