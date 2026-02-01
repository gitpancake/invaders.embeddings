"""Grid-based mosaic detection for Space Invader images.

Detects the mosaic region by finding areas with regular grid patterns
characteristic of tile-based artwork.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_tile_size(edges: np.ndarray, min_tile: int = 8, max_tile: int = 40) -> Tuple[int, float]:
    """
    Estimate tile size from edge image using autocorrelation.
    
    Returns (tile_size, confidence)
    """
    # Autocorrelation via FFT
    f = np.fft.fft2(edges.astype(np.float32))
    autocorr = np.abs(np.fft.ifft2(f * np.conj(f)))
    autocorr = np.fft.fftshift(autocorr)
    autocorr = autocorr / (autocorr.max() + 1e-10)
    
    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    
    # Average horizontal and vertical profiles
    h_profile = autocorr[cy, cx:cx+max_tile+10]
    v_profile = autocorr[cy:cy+max_tile+10, cx]
    avg_profile = (h_profile + v_profile) / 2
    
    # Find strongest peak in valid range
    best_tile = min_tile
    best_strength = 0
    
    for i in range(min_tile, min(max_tile, len(avg_profile) - 1)):
        # Check if this is a local maximum
        if avg_profile[i] > avg_profile[i-1] and avg_profile[i] > avg_profile[i+1]:
            # Check for harmonic (2x should also have a peak for real grid)
            harmonic_idx = i * 2
            harmonic_strength = avg_profile[harmonic_idx] if harmonic_idx < len(avg_profile) else 0
            
            # Score: primary peak + harmonic confirmation
            strength = avg_profile[i] * (1 + harmonic_strength)
            
            if strength > best_strength:
                best_strength = strength
                best_tile = i
    
    return best_tile, best_strength


def compute_grid_score(edges: np.ndarray, tile_size: int) -> float:
    """
    Compute how well the edges match a regular grid pattern.
    
    Higher score = more grid-like structure.
    """
    h, w = edges.shape
    
    # Sum edges along rows and columns
    row_sums = np.sum(edges, axis=1)
    col_sums = np.sum(edges, axis=0)
    
    # A regular grid should have peaks at multiples of tile_size
    # Check periodicity by computing autocorrelation and checking peak at tile_size
    
    def periodicity_score(sums, period):
        if len(sums) < period * 2:
            return 0
        autocorr = np.correlate(sums, sums, mode='full')
        center = len(autocorr) // 2
        autocorr = autocorr / (autocorr[center] + 1e-10)
        
        # Check for peak at the expected period
        if center + period < len(autocorr):
            return autocorr[center + period]
        return 0
    
    row_score = periodicity_score(row_sums, tile_size)
    col_score = periodicity_score(col_sums, tile_size)
    
    # Also factor in edge density (mosaics have lots of edges)
    edge_density = np.sum(edges > 0) / edges.size
    
    return (row_score + col_score) / 2 * (1 + edge_density * 2)


def detect_mosaic_by_grid(image: Image.Image) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    """
    Detect mosaic region by finding areas with strong grid patterns.
    
    Args:
        image: PIL Image
        
    Returns:
        (x, y, width, height) of detected region, or None
        tile_size estimate
    """
    arr = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # First pass: estimate tile size from center region
    center_size = min(w, h) // 2
    cy, cx = h // 2, w // 2
    center_edges = edges[cy-center_size//2:cy+center_size//2, 
                         cx-center_size//2:cx+center_size//2]
    
    tile_size, tile_conf = estimate_tile_size(center_edges)
    logger.info(f'Estimated tile size: {tile_size} pixels (confidence: {tile_conf:.3f})')
    
    # Second pass: sliding window to find region with best grid pattern
    best_score = 0
    best_region = None
    
    # Try different window sizes
    for size_ratio in [0.5, 0.6, 0.4, 0.7]:
        window_size = int(min(w, h) * size_ratio)
        # Make window size a multiple of estimated tile size
        window_size = (window_size // tile_size) * tile_size
        if window_size < tile_size * 4:
            continue
            
        step = max(tile_size, window_size // 6)
        
        for y in range(0, h - window_size + 1, step):
            for x in range(0, w - window_size + 1, step):
                window_edges = edges[y:y+window_size, x:x+window_size]
                
                # Skip if too few edges
                if np.sum(window_edges > 0) / window_edges.size < 0.03:
                    continue
                
                score = compute_grid_score(window_edges, tile_size)
                
                if score > best_score:
                    best_score = score
                    best_region = (x, y, window_size, window_size)
    
    if best_region is None:
        return None, tile_size
    
    # Refine the region bounds
    x, y, rw, rh = best_region
    refined = refine_grid_bounds(edges, x, y, rw, rh, tile_size)
    if refined:
        best_region = refined
    
    logger.info(f'Detected grid region: {best_region} with score {best_score:.4f}')
    return best_region, tile_size


def refine_grid_bounds(edges: np.ndarray, x: int, y: int, w: int, h: int, 
                       tile_size: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Refine the detected region to tighter bounds based on edge density.
    """
    img_h, img_w = edges.shape
    region = edges[y:y+h, x:x+w]
    
    # Sum edges along rows and columns
    row_sums = np.sum(region, axis=1)
    col_sums = np.sum(region, axis=0)
    
    # Smooth with kernel size of tile
    kernel = np.ones(tile_size) / tile_size
    row_smooth = np.convolve(row_sums, kernel, mode='same')
    col_smooth = np.convolve(col_sums, kernel, mode='same')
    
    # Find where density drops significantly
    row_thresh = np.max(row_smooth) * 0.2
    col_thresh = np.max(col_smooth) * 0.2
    
    row_active = np.where(row_smooth > row_thresh)[0]
    col_active = np.where(col_smooth > col_thresh)[0]
    
    if len(row_active) < tile_size or len(col_active) < tile_size:
        return None
    
    # Get bounds
    top = row_active[0]
    bottom = row_active[-1]
    left = col_active[0]
    right = col_active[-1]
    
    # Add small padding
    pad = tile_size // 2
    new_x = max(0, x + left - pad)
    new_y = max(0, y + top - pad)
    new_w = min(img_w - new_x, right - left + 2 * pad)
    new_h = min(img_h - new_y, bottom - top + 2 * pad)
    
    # Sanity check
    if new_w < tile_size * 3 or new_h < tile_size * 3:
        return None
    
    return (new_x, new_y, new_w, new_h)


def preprocess_with_grid_detection(image: Image.Image) -> Image.Image:
    """
    Main preprocessing function using grid detection.
    """
    region, tile_size = detect_mosaic_by_grid(image)
    
    if region is not None:
        x, y, w, h = region
        cropped = image.crop((x, y, x + w, y + h))
        logger.info(f'Cropped from {image.size} to {cropped.size}')
        return cropped
    else:
        # Fallback to center crop
        logger.info('Grid detection failed, using center crop')
        w, h = image.size
        crop_size = int(min(w, h) * 0.7)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return image.crop((left, top, left + crop_size, top + crop_size))
