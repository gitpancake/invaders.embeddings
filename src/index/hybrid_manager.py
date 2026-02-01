"""Hybrid index manager combining DINOv2 embeddings with perceptual hash re-ranking."""

import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HybridIndexManager:
    """
    Hybrid index that uses DINOv2 embeddings for initial retrieval
    and perceptual hashing for re-ranking.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.flash_metadata: List[Dict[str, Any]] = []
        self.flash_hashes: Dict[int, Dict] = {}

    def build_index(
        self,
        embeddings: np.ndarray,
        flash_metadata: List[Dict[str, Any]],
        flash_hashes: Dict[int, Dict] = None
    ) -> None:
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.flash_metadata = flash_metadata
        self.flash_hashes = flash_hashes or {}

        logger.info(f"Built hybrid index with {self.index.ntotal} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_hashes: Dict = None,
        rerank_k: int = 20,
        hash_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        fetch_k = rerank_k if query_hashes and self.flash_hashes else top_k
        similarities, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
                
            metadata = self.flash_metadata[idx]
            flash_id = metadata["i"]
            
            embedding_sim = float(sim)
            hash_sim = 0.0
            
            if query_hashes and flash_id in self.flash_hashes:
                hash_sim = self._compute_hash_similarity(
                    query_hashes, 
                    self.flash_hashes[flash_id]
                )
                combined_sim = (1 - hash_weight) * embedding_sim + hash_weight * hash_sim
            else:
                combined_sim = embedding_sim
            
            results.append({
                "flash_id": flash_id,
                "flash_name": metadata["n"],
                "similarity": combined_sim,
                "embedding_similarity": embedding_sim,
                "hash_similarity": hash_sim,
                "confidence": max(0.0, min(1.0, combined_sim)),
                "location": metadata.get("l"),
                "image_url": metadata.get("t"),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _compute_hash_similarity(self, hashes1: Dict, hashes2: Dict) -> float:
        weights = {"phash": 0.4, "dhash": 0.4, "color": 0.2}
        total_weight = 0.0
        similarity = 0.0
        
        for hash_type, weight in weights.items():
            if hash_type in hashes1 and hash_type in hashes2:
                h1 = np.array(hashes1[hash_type])
                h2 = np.array(hashes2[hash_type])
                
                if hash_type == "color":
                    sim = float(np.sum(np.minimum(h1, h2)))
                else:
                    sim = 1.0 - (np.sum(h1 != h2) / len(h1))
                
                similarity += weight * sim
                total_weight += weight
        
        return similarity / total_weight if total_weight > 0 else 0.0

    def save(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("No index to save")

        path = Path(path)
        faiss.write_index(self.index, str(path.with_suffix(".index")))

        serializable_hashes = {}
        for flash_id, hashes in self.flash_hashes.items():
            serializable_hashes[str(flash_id)] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in hashes.items()
            }

        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump({
                "dimension": self.dimension,
                "flash_metadata": self.flash_metadata,
                "flash_hashes": serializable_hashes
            }, f)

        logger.info(f"Saved hybrid index to {path}")

    def load(self, path: str) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path.with_suffix(".index")))

        with open(path.with_suffix(".meta.json"), "r") as f:
            data = json.load(f)
            self.dimension = data["dimension"]
            self.flash_metadata = data["flash_metadata"]
            
            self.flash_hashes = {}
            for flash_id, hashes in data.get("flash_hashes", {}).items():
                self.flash_hashes[int(flash_id)] = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in hashes.items()
                }

        logger.info(f"Loaded hybrid index with {self.index.ntotal} vectors")

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
