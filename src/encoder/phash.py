"""Perceptual hashing for image similarity.

Perceptual hashes capture structural patterns and are more robust
to minor color/lighting changes than raw pixel comparison.
"""

import numpy as np
from PIL import Image
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


def compute_phash(image: Union[Image.Image, str], hash_size: int = 16) -> np.ndarray:
    """
    Compute perceptual hash of an image using DCT.
    
    Args:
        image: PIL Image or path to image file
        hash_size: Size of the hash (default 16 = 256 bits)
    
    Returns:
        Binary hash as numpy array of shape (hash_size * hash_size,)
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert to grayscale and resize
    img = image.convert('L').resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    pixels = np.array(img, dtype=np.float64)
    
    # Compute 2D DCT
    from scipy.fftpack import dct
    dct_result = dct(dct(pixels, axis=0), axis=1)
    
    # Keep only top-left (low frequency) components
    dct_low = dct_result[:hash_size, :hash_size]
    
    # Compute hash: 1 if above median, 0 otherwise
    median = np.median(dct_low)
    hash_bits = (dct_low > median).flatten().astype(np.uint8)
    
    return hash_bits


def compute_dhash(image: Union[Image.Image, str], hash_size: int = 16) -> np.ndarray:
    """
    Compute difference hash of an image.
    
    Compares adjacent pixels to detect gradients/edges.
    
    Args:
        image: PIL Image or path to image file
        hash_size: Size of the hash (default 16 = 256 bits)
    
    Returns:
        Binary hash as numpy array
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Resize to hash_size + 1 width (for horizontal gradient)
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    
    # Compute horizontal gradient (is left pixel brighter than right?)
    diff = pixels[:, 1:] > pixels[:, :-1]
    
    return diff.flatten().astype(np.uint8)


def compute_color_hash(image: Union[Image.Image, str], bins: int = 8) -> np.ndarray:
    """
    Compute color histogram hash.
    
    Captures the color distribution which is important for Space Invader mosaics.
    
    Args:
        image: PIL Image or path to image file
        bins: Number of bins per channel (default 8 = 512 total features)
    
    Returns:
        Normalized histogram as numpy array
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    img = image.convert('RGB')
    pixels = np.array(img)
    
    # Compute histogram for each channel
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, :, channel], bins=bins, range=(0, 256))
        histograms.append(hist)
    
    # Concatenate and normalize
    full_hist = np.concatenate(histograms).astype(np.float32)
    full_hist = full_hist / (full_hist.sum() + 1e-10)
    
    return full_hist


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Compute Hamming distance between two binary hashes."""
    return np.sum(hash1 != hash2)


def hamming_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Compute similarity (0-1) from Hamming distance."""
    distance = hamming_distance(hash1, hash2)
    max_distance = len(hash1)
    return 1.0 - (distance / max_distance)


def histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute histogram intersection similarity."""
    return np.sum(np.minimum(hist1, hist2))


class PerceptualHasher:
    """
    Combines multiple perceptual hashing methods for robust similarity.
    """
    
    def __init__(self, phash_size: int = 16, dhash_size: int = 16, color_bins: int = 8):
        self.phash_size = phash_size
        self.dhash_size = dhash_size
        self.color_bins = color_bins
    
    def compute_hashes(self, image: Union[Image.Image, str]) -> dict:
        """Compute all hash types for an image."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return {
            'phash': compute_phash(image, self.phash_size),
            'dhash': compute_dhash(image, self.dhash_size),
            'color': compute_color_hash(image, self.color_bins),
        }
    
    def similarity(self, hashes1: dict, hashes2: dict, weights: dict = None) -> float:
        """
        Compute weighted similarity between two hash sets.
        
        Args:
            hashes1: First hash dict from compute_hashes()
            hashes2: Second hash dict from compute_hashes()
            weights: Optional weights for each hash type
        
        Returns:
            Combined similarity score (0-1)
        """
        if weights is None:
            # Default weights emphasize structural similarity
            weights = {'phash': 0.4, 'dhash': 0.4, 'color': 0.2}
        
        total_weight = sum(weights.values())
        
        similarity = 0.0
        
        if 'phash' in weights and 'phash' in hashes1 and 'phash' in hashes2:
            similarity += weights['phash'] * hamming_similarity(hashes1['phash'], hashes2['phash'])
        
        if 'dhash' in weights and 'dhash' in hashes1 and 'dhash' in hashes2:
            similarity += weights['dhash'] * hamming_similarity(hashes1['dhash'], hashes2['dhash'])
        
        if 'color' in weights and 'color' in hashes1 and 'color' in hashes2:
            similarity += weights['color'] * histogram_similarity(hashes1['color'], hashes2['color'])
        
        return similarity / total_weight
