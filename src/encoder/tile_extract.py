"""Improved tile pattern extraction from Space Invader mosaics.

Handles both clean reference images and real-world photos by:
1. Color quantization to identify tile colors
2. Grid detection using autocorrelation
3. Automatic mosaic bounds detection
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict
from sklearn.cluster import KMeans
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
    'light_blue': (135, 206, 235),
}


def quantize_image(image: np.ndarray, n_colors: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image to n_colors using k-means.
    Returns quantized image and cluster centers.
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    quantized = centers[labels].reshape(h, w, 3)
    return quantized, centers


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
    h_profile = autocorr[cy, cx:cx+max_tile+5]
    v_profile = autocorr[cy:cy+max_tile+5, cx]
    profile = (h_profile + v_profile) / 2
    
    # Find the strongest peak in valid range
    best_tile = min_tile
    best_score = 0
    
    for i in range(min_tile, min(max_tile, len(profile) - 2)):
        if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
            # Check for harmonic at 2x
            harmonic = profile[i*2] if i*2 < len(profile) else 0
            score = profile[i] * (1 + harmonic)
            
            if score > best_score:
                best_score = score
                best_tile = i
    
    return best_tile


def find_mosaic_bounds(image: np.ndarray, quantized: np.ndarray, centers: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounds of the actual mosaic (excluding background).
    
    Strategy: Find the most common color (likely background) and find
    the bounding box of everything else.
    """
    h, w = image.shape[:2]
    
    # Find which cluster is background (most common AND low saturation)
    pixels = quantized.reshape(-1, 3)
    labels = np.zeros(len(pixels), dtype=int)
    
    for i, pixel in enumerate(pixels):
        for j, center in enumerate(centers):
            if np.array_equal(pixel, center):
                labels[i] = j
                break
    
    label_counts = np.bincount(labels, minlength=len(centers))
    
    # Calculate saturation for each center
    centers_hsv = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
    saturations = centers_hsv[:, 1]
    values = centers_hsv[:, 2]
    
    # Background is typically: most common AND (low saturation OR very uniform)
    # For mosaics, we want to keep: colorful tiles, black tiles, white tiles
    
    # Score each cluster - high score = likely background
    bg_scores = []
    for i in range(len(centers)):
        count_score = label_counts[i] / len(pixels)
        # Low saturation suggests gray background
        sat_score = 1 - saturations[i] / 255
        # Middle value suggests gray (not black or white tiles)
        mid_value_score = 1 - abs(values[i] - 128) / 128
        
        bg_score = count_score * 0.5 + sat_score * 0.3 + mid_value_score * 0.2
        bg_scores.append(bg_score)
    
    # Top 2 bg candidates
    bg_candidates = np.argsort(bg_scores)[-2:]
    
    # Create mask of non-background pixels
    label_img = labels.reshape(h, w)
    fg_mask = np.ones((h, w), dtype=bool)
    for bg_idx in bg_candidates:
        # Only exclude if it is really dominant
        if label_counts[bg_idx] / len(pixels) > 0.15:
            fg_mask &= (label_img != bg_idx)
    
    # Find bounding box
    coords = np.column_stack(np.where(fg_mask))
    if len(coords) < 100:
        # Not enough foreground, return full image
        return 0, 0, w, h
    
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    
    # Add small padding
    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    return x1, y1, x2 - x1, y2 - y1


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


def extract_tile_pattern_v2(image: Image.Image, n_colors: int = 10) -> Dict:
    """
    Extract tile pattern with improved algorithm.
    
    Args:
        image: PIL Image (can be full photo or cropped)
        n_colors: Number of colors for quantization
    
    Returns:
        dict with tile_size, grid_shape, colors, pattern, signature
    """
    arr = np.array(image.convert('RGB'))
    h, w = arr.shape[:2]
    
    # Step 1: Quantize colors
    quantized, centers = quantize_image(arr, n_colors)
    
    # Step 2: Find mosaic bounds
    x, y, mw, mh = find_mosaic_bounds(arr, quantized, centers)
    logger.info(f'Mosaic bounds: ({x}, {y}, {mw}, {mh})')
    
    # Crop to mosaic
    mosaic_arr = arr[y:y+mh, x:x+mw]
    mosaic_quant = quantized[y:y+mh, x:x+mw]
    
    # Step 3: Find tile size
    tile_size = find_tile_size(mosaic_quant)
    logger.info(f'Tile size: {tile_size} pixels')
    
    # Step 4: Calculate grid dimensions
    n_rows = max(1, round(mh / tile_size))
    n_cols = max(1, round(mw / tile_size))
    logger.info(f'Grid: {n_rows} x {n_cols}')
    
    # Step 5: Sample colors at tile centers
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
