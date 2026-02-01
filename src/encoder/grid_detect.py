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
    if edges.size == 0 or edges.max() == 0:
        return min_tile, 0.0
        
    f = np.fft.fft2(edges.astype(np.float32))
    autocorr = np.abs(np.fft.ifft2(f * np.conj(f)))
    autocorr = np.fft.fftshift(autocorr)
    autocorr = autocorr / (autocorr.max() + 1e-10)

    h, w = autocorr.shape
    cy, cx = h // 2, w // 2

    max_range = min(max_tile + 10, min(h, w) // 2)
    h_profile = autocorr[cy, cx:cx+max_range]
    v_profile = autocorr[cy:cy+max_range, cx]
    avg_profile = (h_profile + v_profile) / 2

    best_tile = min_tile
    best_strength = 0

    for i in range(min_tile, min(max_tile, len(avg_profile) - 1)):
        if avg_profile[i] > avg_profile[i-1] and avg_profile[i] > avg_profile[i+1]:
            harmonic_idx = i * 2
            harmonic_strength = avg_profile[harmonic_idx] if harmonic_idx < len(avg_profile) else 0
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
    if h < tile_size * 2 or w < tile_size * 2:
        return 0.0

    row_sums = np.sum(edges, axis=1)
    col_sums = np.sum(edges, axis=0)

    def periodicity_score(sums, period):
        if len(sums) < period * 2:
            return 0
        autocorr = np.correlate(sums, sums, mode='full')
        center = len(autocorr) // 2
        autocorr = autocorr / (autocorr[center] + 1e-10)
        if center + period < len(autocorr):
            return autocorr[center + period]
        return 0

    row_score = periodicity_score(row_sums, tile_size)
    col_score = periodicity_score(col_sums, tile_size)

    # Edge density factor
    edge_density = np.sum(edges > 0) / edges.size
    
    density_factor = 1.0
    if edge_density < 0.03:
        density_factor = edge_density / 0.03
    elif edge_density > 0.25:
        density_factor = 0.25 / edge_density

    return (row_score + col_score) / 2 * (1 + density_factor)


def compute_rectangularity_score(edges: np.ndarray) -> float:
    """
    Compute how rectangular/grid-like the edges are.
    Mosaics have mostly horizontal and vertical lines.
    Graffiti has many diagonal/curved lines.
    Returns a score 0-1 where 1 = highly rectangular.
    """
    if edges.size == 0 or edges.max() == 0:
        return 0.0
    
    # Use Hough line detection to find line angles
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                            minLineLength=10, maxLineGap=5)
    
    if lines is None or len(lines) < 3:
        return 0.5  # Not enough lines to judge
    
    # Count horizontal, vertical, and diagonal lines
    horiz_count = 0
    vert_count = 0
    diag_count = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
        
        # Horizontal: 0-15 degrees, Vertical: 75-90 degrees, Diagonal: 15-75 degrees
        if angle < 15:
            horiz_count += 1
        elif angle > 75:
            vert_count += 1
        else:
            diag_count += 1
    
    total = horiz_count + vert_count + diag_count
    if total == 0:
        return 0.5
    
    # High rectangularity = mostly horizontal + vertical, few diagonals
    rect_ratio = (horiz_count + vert_count) / total
    return rect_ratio


def compute_color_variance(image_region: np.ndarray) -> float:
    """
    Compute color variance in a region.
    Mosaics typically have distinct, saturated colors.
    Returns a score where higher = more colorful.
    """
    if image_region.size == 0:
        return 0.0
    
    # Convert to HSV
    hsv = cv2.cvtColor(image_region, cv2.COLOR_RGB2HSV)
    
    # Saturation channel - mosaics tend to have higher saturation
    saturation = hsv[:, :, 1]
    mean_sat = np.mean(saturation)
    
    # Value channel variance - mosaics have distinct tile colors
    value = hsv[:, :, 2]
    value_std = np.std(value)
    
    # Combine: prefer moderate-high saturation and some value variance
    sat_score = min(mean_sat / 100, 1.0)
    var_score = min(value_std / 50, 1.0)
    
    return (sat_score + var_score) / 2


def detect_mosaic_by_grid(image: Image.Image, 
                          min_window_ratio: float = 0.08,
                          max_window_ratio: float = 0.35) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    """
    Detect mosaic region by finding areas with strong grid patterns.
    """
    arr = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    min_dim = min(w, h)
    cx, cy = w // 2, h // 2

    edges = cv2.Canny(gray, 50, 150)

    # Estimate tile size from center region
    center_size = min_dim // 2
    center_edges = edges[cy-center_size//2:cy+center_size//2,
                         cx-center_size//2:cx+center_size//2]
    tile_size, _ = estimate_tile_size(center_edges)
    
    tile_estimates = [tile_size]
    for ratio in [0.25, 0.4]:
        size = int(min_dim * ratio)
        for offset_y in [-size//2, 0, size//2]:
            for offset_x in [-size//2, 0, size//2]:
                sy = max(0, cy + offset_y - size//2)
                sx = max(0, cx + offset_x - size//2)
                ey = min(h, sy + size)
                ex = min(w, sx + size)
                if ey - sy > 50 and ex - sx > 50:
                    sample_edges = edges[sy:ey, sx:ex]
                    ts, conf = estimate_tile_size(sample_edges)
                    if conf > 0.15:
                        tile_estimates.append(ts)
    
    tile_size = int(np.median(tile_estimates))
    tile_size = max(8, min(40, tile_size))
    
    logger.info(f'Estimated tile size: {tile_size} pixels')

    best_score = 0
    best_region = None

    min_window = max(int(min_dim * min_window_ratio), tile_size * 4)
    max_window = int(min_dim * max_window_ratio)
    
    window_sizes = []
    current = min_window
    while current <= max_window:
        rounded = (current // tile_size) * tile_size
        if rounded >= tile_size * 4 and rounded not in window_sizes:
            window_sizes.append(rounded)
        current = int(current * 1.2)
    
    # Exclude top/bottom 10% of image (often sky/ground with false patterns)
    margin_y = int(h * 0.10)
    search_y_start = margin_y
    search_y_end = h - margin_y
    
    for window_size in window_sizes:
        step = max(tile_size // 2, window_size // 10)

        for y in range(search_y_start, search_y_end - window_size + 1, step):
            for x in range(0, w - window_size + 1, step):
                window_edges = edges[y:y+window_size, x:x+window_size]

                score = compute_grid_score(window_edges, tile_size)
                
                if score < 0.1:
                    continue

                # Rectangularity score - penalize graffiti/curved patterns
                rect_score = compute_rectangularity_score(window_edges)

                # Color score - prefer colorful regions
                window_rgb = arr[y:y+window_size, x:x+window_size]
                color_score = compute_color_variance(window_rgb)
                
                # Center preference (both horizontal and vertical)
                window_cy = y + window_size // 2
                window_cx = x + window_size // 2
                
                vert_dist = abs(window_cy - cy) / cy
                horiz_dist = abs(window_cx - cx) / cx
                
                # Strong center preference - mosaic is usually near image center
                center_factor = 1.0 - 1.0 * (vert_dist + horiz_dist) / 2
                
                # Prefer smaller windows
                size_factor = 1.0 - 0.1 * (window_size - min_window) / (max_window - min_window + 1)
                
                # Rectangularity factor: only penalize very low rectangularity (likely graffiti)
                # rect >= 0.7: no penalty, 0.5-0.7: mild penalty, < 0.5: strong penalty
                if rect_score >= 0.7:
                    rect_factor = 1.0
                elif rect_score >= 0.5:
                    rect_factor = 0.7 + (rect_score - 0.5) * 1.5  # 0.7 to 1.0
                else:
                    rect_factor = rect_score * 1.4  # 0 to 0.7

                # Combined score - increased center preference weight
                adjusted_score = score * rect_factor * (0.8 + 0.2 * color_score) * center_factor * size_factor

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_region = (x, y, window_size, window_size)

    if best_region is None:
        return None, tile_size

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

    row_sums = np.sum(region, axis=1)
    col_sums = np.sum(region, axis=0)

    kernel = np.ones(max(1, tile_size)) / max(1, tile_size)
    row_smooth = np.convolve(row_sums, kernel, mode='same')
    col_smooth = np.convolve(col_sums, kernel, mode='same')

    row_thresh = np.max(row_smooth) * 0.25
    col_thresh = np.max(col_smooth) * 0.25

    row_active = np.where(row_smooth > row_thresh)[0]
    col_active = np.where(col_smooth > col_thresh)[0]

    if len(row_active) < tile_size or len(col_active) < tile_size:
        return None

    top = row_active[0]
    bottom = row_active[-1]
    left = col_active[0]
    right = col_active[-1]

    pad = tile_size // 2
    new_x = max(0, x + left - pad)
    new_y = max(0, y + top - pad)
    new_w = min(img_w - new_x, right - left + 2 * pad)
    new_h = min(img_h - new_y, bottom - top + 2 * pad)

    if new_w < tile_size * 3 or new_h < tile_size * 3:
        return None

    return (new_x, new_y, new_w, new_h)


def preprocess_with_grid_detection(image: Image.Image, 
                                   min_window_ratio: float = 0.08,
                                   max_window_ratio: float = 0.35) -> Image.Image:
    """
    Main preprocessing function using grid detection.
    """
    region, tile_size = detect_mosaic_by_grid(image, min_window_ratio, max_window_ratio)

    if region is not None:
        x, y, w, h = region
        cropped = image.crop((x, y, x + w, y + h))
        logger.info(f'Cropped from {image.size} to {cropped.size}')
        return cropped
    else:
        logger.info('Grid detection failed, using center crop')
        w, h = image.size
        crop_size = int(min(w, h) * 0.3)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return image.crop((left, top, left + crop_size, top + crop_size))
