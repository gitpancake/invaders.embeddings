"""FAISS index manager for similarity search."""

import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Manages FAISS index for flash image similarity search.
    Uses inner product (cosine similarity on normalized vectors).
    """
    
    def __init__(self, dimension: int = 768):
        """
        Initialize the FAISS index manager.
        
        Args:
            dimension: Embedding dimension (768 for CLIP ViT-L/14)
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.flash_metadata: List[Dict[str, Any]] = []  # Maps index position to flash info
    
    def build_index(
        self, 
        embeddings: np.ndarray, 
        flash_metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Array of shape (n_flashes, dimension)
            flash_metadata: List of dicts with flash info (id, name, etc.)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        
        if len(embeddings) != len(flash_metadata):
            raise ValueError(f"Embeddings ({len(embeddings)}) and metadata ({len(flash_metadata)}) count mismatch")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # For ~4000 images, use exact search (IndexFlatIP is fast enough)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.flash_metadata = flash_metadata
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar flashes.
        
        Args:
            query_embedding: Query embedding of shape (dimension,) or (1, dimension)
            top_k: Number of results to return
            
        Returns:
            List of dicts with flash_id, flash_name, similarity, confidence
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32 and normalized
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for not found
                metadata = self.flash_metadata[idx]
                results.append({
                    "flash_id": metadata["i"],
                    "flash_name": metadata["n"],
                    "similarity": float(sim),
                    "confidence": self._similarity_to_confidence(sim),
                    "location": metadata.get("l"),
                    "image_url": metadata.get("t"),
                })
        
        return results
    
    def _similarity_to_confidence(self, similarity: float) -> float:
        """
        Convert cosine similarity to a confidence score.
        
        Similarity ranges from -1 to 1 (normalized vectors).
        We map this to 0-1 confidence, with thresholds:
        - >0.9: Very high confidence (likely same flash)
        - 0.7-0.9: High confidence
        - 0.5-0.7: Medium confidence
        - <0.5: Low confidence
        """
        # Simple linear mapping from [0, 1] to [0, 1]
        # (since we use normalized vectors, similarity is typically positive)
        return max(0.0, min(1.0, similarity))
    
    def save(self, path: str) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            path: Base path (will create {path}.index and {path}.meta.json)
        """
        if self.index is None:
            raise RuntimeError("No index to save")
        
        path = Path(path)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix(".index")))
        
        # Save metadata
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump({
                "dimension": self.dimension,
                "flash_metadata": self.flash_metadata
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the index and metadata from disk.
        
        Args:
            path: Base path (expects {path}.index and {path}.meta.json)
        """
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix(".index")))
        
        # Load metadata
        with open(path.with_suffix(".meta.json"), "r") as f:
            data = json.load(f)
            self.dimension = data["dimension"]
            self.flash_metadata = data["flash_metadata"]
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors from {path}")
    
    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
