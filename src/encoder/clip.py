"""CLIP encoder wrapper for generating image embeddings."""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union
import logging

from .preprocess import preprocess_flash_image

# Enable AVIF support
try:
    import pillow_avif
except ImportError:
    pass

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    CLIP encoder for generating image embeddings.
    Uses clip-vit-base-patch32 with float16 for lower memory usage (~300MB vs 600MB).
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM = 512  # base model uses 512-dim embeddings

    def __init__(self, device: str = None, use_fp16: bool = True):
        """
        Initialize the CLIP encoder.

        Args:
            device: Device to use ('mps' for M1, 'cuda' for NVIDIA, 'cpu' for CPU).
                   If None, auto-detects the best available device.
            use_fp16: Use half precision to reduce memory (default True)
        """
        if device is None:
            device = self._get_best_device()

        self.device = device
        self.use_fp16 = use_fp16 and device != "mps"  # MPS doesn't fully support fp16

        logger.info(f"Initializing CLIP encoder on device: {device}, fp16: {self.use_fp16}")

        # Load model and processor
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)

        if self.use_fp16:
            self.model = CLIPModel.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)

        self.model.eval()

        logger.info(f"CLIP model loaded: {self.MODEL_NAME}")

    def _get_best_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"

    def encode_image(self, image: Union[Image.Image, str], preprocess: bool = False) -> np.ndarray:
        """
        Encode a single image to an embedding vector.

        Args:
            image: PIL Image or path to image file
            preprocess: Whether to apply mosaic detection preprocessing

        Returns:
            Normalized embedding vector of shape (512,)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return self.encode_images([image], preprocess=preprocess)[0]

    def encode_images(self, images: List[Image.Image], batch_size: int = 32, preprocess: bool = False) -> np.ndarray:
        """
        Encode multiple images to embedding vectors.

        Args:
            images: List of PIL Images
            batch_size: Number of images to process at once
            preprocess: Whether to apply mosaic detection preprocessing

        Returns:
            Normalized embedding vectors of shape (n_images, 512)
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Ensure all images are RGB
            batch = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]
            
            # Apply preprocessing if requested
            if preprocess:
                batch = [preprocess_flash_image(img) for img in batch]

            # Process batch
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Convert to fp16 if enabled
            if self.use_fp16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                # Get image features - returns a tensor directly
                outputs = self.model.get_image_features(**inputs)

                # Handle both tensor and object return types
                if hasattr(outputs, 'last_hidden_state'):
                    # If it returns an object, get the pooled output
                    image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
                else:
                    # It's already a tensor
                    image_features = outputs

                # L2 normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert back to float32 for numpy
            all_embeddings.append(image_features.float().cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_image_from_bytes(self, image_bytes: bytes, preprocess: bool = False) -> np.ndarray:
        """
        Encode an image from raw bytes.

        Args:
            image_bytes: Raw image bytes
            preprocess: Whether to apply mosaic detection preprocessing

        Returns:
            Normalized embedding vector of shape (512,)
        """
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.encode_image(image, preprocess=preprocess)
