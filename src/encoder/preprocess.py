"""Image preprocessing for Space Invader mosaic detection and cropping."""

import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def preprocess_flash_image(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Preprocess a flash photo to extract and enhance the mosaic region.
    
    This attempts to:
    1. Detect the mosaic region in the image
    2. Crop to that region
    3. Return the cropped image for embedding
    
    Args:
        image: Input PIL Image
        target_size: Target size for the output image
        
    Returns:
        Preprocessed PIL Image focused on the mosaic
    """
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Try to detect and crop the mosaic region
        cropped = detect_and_crop_mosaic(image)
        
        if cropped is not None:
            logger.info(f"Detected mosaic region, cropped from {image.size} to {cropped.size}")
            return cropped
        else:
            logger.info("Could not detect mosaic, using center crop")
            return center_crop(image)
            
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}, returning original")
        return image


def detect_and_crop_mosaic(image: Image.Image, min_size_ratio: float = 0.2) -> Image.Image:
    """
    Detect the mosaic region in an image using grid pattern and color analysis.
    
    Space Invader mosaics typically have:
    - High color saturation (bright, distinct colors)
    - Dense grid pattern (many horizontal and vertical lines)
    - Located somewhere in the frame
    
    Args:
        image: Input PIL Image
        min_size_ratio: Minimum size of detected region relative to image
        
    Returns:
        Cropped PIL Image or None if detection failed
    """
    import cv2
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_array.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Use a sliding window approach to find the region with highest "mosaic score"
    # Score = combination of edge density + color saturation
    
    window_sizes = [
        (int(img_w * 0.5), int(img_h * 0.5)),  # 50% of image
        (int(img_w * 0.4), int(img_h * 0.4)),  # 40% of image
        (int(img_w * 0.6), int(img_h * 0.6)),  # 60% of image
    ]
    
    best_score = 0
    best_region = None
    
    for win_w, win_h in window_sizes:
        if win_w < 50 or win_h < 50:
            continue
            
        step = max(win_w // 4, 20)  # Step size for sliding window
        
        for y in range(0, img_h - win_h + 1, step):
            for x in range(0, img_w - win_w + 1, step):
                # Extract window
                edge_window = edges[y:y+win_h, x:x+win_w]
                sat_window = saturation[y:y+win_h, x:x+win_w]
                
                # Calculate edge density (more edges = more likely mosaic grid)
                edge_density = np.sum(edge_window > 0) / edge_window.size
                
                # Calculate saturation score (mosaics are colorful)
                # Use mean of pixels above a threshold
                high_sat_mask = sat_window > 80
                if np.sum(high_sat_mask) > 0:
                    sat_score = np.mean(sat_window[high_sat_mask]) / 255.0
                else:
                    sat_score = 0
                
                # Calculate color variance (mosaics have distinct color blocks)
                color_window = img_array[y:y+win_h, x:x+win_w]
                color_variance = np.std(color_window) / 128.0  # Normalize
                
                # Combined score
                # Weight edge density highly (grid pattern is key)
                score = (edge_density * 2.0) + (sat_score * 1.0) + (color_variance * 0.5)
                
                if score > best_score:
                    best_score = score
                    best_region = (x, y, win_w, win_h)
    
    if best_region is None:
        return None
    
    x, y, w, h = best_region
    
    # Refine the region: try to find tighter bounds using edge concentration
    refined = refine_mosaic_bounds(edges, saturation, x, y, w, h, img_w, img_h)
    if refined:
        x, y, w, h = refined
    
    # Add small padding
    padding = int(min(w, h) * 0.05)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_w - x, w + 2 * padding)
    h = min(img_h - y, h + 2 * padding)
    
    # Minimum size check
    if w < img_w * min_size_ratio or h < img_h * min_size_ratio:
        return None
    
    # Crop the region
    cropped = image.crop((x, y, x + w, y + h))
    
    return cropped


def refine_mosaic_bounds(edges: np.ndarray, saturation: np.ndarray, 
                          x: int, y: int, w: int, h: int, 
                          img_w: int, img_h: int) -> tuple:
    """
    Refine the detected region bounds by finding where edge/color density drops off.
    """
    # Get the region
    edge_region = edges[y:y+h, x:x+w]
    sat_region = saturation[y:y+h, x:x+w]
    
    # Calculate row and column profiles
    row_edges = np.sum(edge_region > 0, axis=1)
    col_edges = np.sum(edge_region > 0, axis=0)
    
    row_sat = np.mean(sat_region > 80, axis=1)
    col_sat = np.mean(sat_region > 80, axis=0)
    
    # Combine profiles
    row_score = row_edges / (w + 1) + row_sat
    col_score = col_edges / (h + 1) + col_sat
    
    # Find bounds where score is above threshold
    threshold = 0.1
    
    # Find top/bottom bounds
    active_rows = np.where(row_score > threshold)[0]
    if len(active_rows) < 10:
        return None
    top = active_rows[0]
    bottom = active_rows[-1]
    
    # Find left/right bounds
    active_cols = np.where(col_score > threshold)[0]
    if len(active_cols) < 10:
        return None
    left = active_cols[0]
    right = active_cols[-1]
    
    # Convert back to image coordinates
    new_x = x + left
    new_y = y + top
    new_w = right - left
    new_h = bottom - top
    
    # Sanity check
    if new_w < 50 or new_h < 50:
        return None
    
    return (new_x, new_y, new_w, new_h)


def center_crop(image: Image.Image, crop_ratio: float = 0.7) -> Image.Image:
    """
    Perform a center crop on the image.
    
    Args:
        image: Input PIL Image
        crop_ratio: Ratio of the image to keep (0.7 = keep center 70%)
        
    Returns:
        Center-cropped PIL Image
    """
    width, height = image.size
    
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    return image.crop((left, top, right, bottom))
