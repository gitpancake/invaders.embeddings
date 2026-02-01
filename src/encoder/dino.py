"""DINOv2 encoder for generating image embeddings.

DINOv2 is better at capturing structural/geometric features compared to CLIP,
making it more suitable for fine-grained visual matching of mosaic patterns.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union
import logging

from .preprocess import preprocess_flash_image

try:
    import pillow_avif
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DINOv2Encoder:
    MODEL_NAME = "facebook/dinov2-small"
    EMBEDDING_DIM = 384

    def __init__(self, device: str = None, use_fp16: bool = True):
        from transformers import AutoImageProcessor, AutoModel

        if device is None:
            device = self._get_best_device()

        self.device = device
        self.use_fp16 = use_fp16 and device != "mps"

        logger.info(f"Initializing DINOv2 encoder on device: {device}, fp16: {self.use_fp16}")

        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)

        if self.use_fp16:
            self.model = AutoModel.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)

        self.model.eval()
        logger.info(f"DINOv2 model loaded: {self.MODEL_NAME}")

    def _get_best_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def encode_image(self, image: Union[Image.Image, str], preprocess: bool = False) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return self.encode_images([image], preprocess=preprocess)[0]

    def encode_images(self, images: List[Image.Image], batch_size: int = 32, preprocess: bool = False) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]

            if preprocess:
                batch = [preprocess_flash_image(img) for img in batch]

            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.use_fp16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state[:, 0]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.float().cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_image_from_bytes(self, image_bytes: bytes, preprocess: bool = False) -> np.ndarray:
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.encode_image(image, preprocess=preprocess)
