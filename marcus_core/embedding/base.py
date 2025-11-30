"""
Base Embedding Extractor Interface
===================================

Abstract base class for face embedding extraction models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class EmbeddingExtractor(ABC):
    """
    Abstract base class for face embedding extraction.
    
    All embedding extractors must implement the extract() method.
    Embeddings should be L2 normalized for cosine similarity.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 512,
        normalize: bool = True,
        device: str = "auto",
        fp16: bool = True,
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to model weights
            embedding_dim: Output embedding dimension
            normalize: Whether to L2 normalize embeddings
            device: Compute device (auto, cuda, mps, cpu)
            fp16: Use FP16 inference for speed
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self._device = device
        self.fp16 = fp16
        self._resolved_device: Optional[str] = None
    
    @property
    def device(self) -> str:
        """Get the resolved compute device."""
        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()
        return self._resolved_device
    
    def _resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self._device != "auto":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @abstractmethod
    def extract(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        Extract embedding from a single face image.
        
        Args:
            face_image: Aligned face image (RGB, uint8, typically 112x112)
        
        Returns:
            Embedding vector (embedding_dim,)
        """
        pass
    
    @abstractmethod
    def extract_batch(
        self,
        face_images: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_images: List of aligned face images
        
        Returns:
            Embeddings array (N, embedding_dim)
        """
        pass
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding(s)."""
        if not self.normalize:
            return embedding
        
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            return embedding / (norm + 1e-10)
        else:
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding / (norms + 1e-10)
    
    @abstractmethod
    def warmup(self) -> None:
        """Warm up the model with dummy inference."""
        pass
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Cosine similarity [-1, 1], higher is more similar
        """
        # Ensure normalized
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        return float(np.dot(e1, e2))
    
    def compute_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """
        Compute distance between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Distance metric (cosine, euclidean)
        
        Returns:
            Distance value (lower is more similar)
        """
        if metric == "cosine":
            return 1.0 - self.compute_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            return float(np.linalg.norm(embedding1 - embedding2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dim={self.embedding_dim}, "
            f"normalize={self.normalize}, "
            f"device={self.device})"
        )
