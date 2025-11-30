"""
Base Vector Store Interface
============================

Abstract base class for vector database implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid
import numpy as np


@dataclass
class SearchResult:
    """
    Result from a vector similarity search.
    
    Attributes:
        id: Unique identifier of the matched item
        score: Similarity score (higher = more similar for cosine)
        distance: Distance value (lower = more similar)
        metadata: Associated metadata
        embedding: The matched embedding vector (optional)
    """
    id: str
    score: float
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding embedding)."""
        return {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "metadata": self.metadata,
        }


@dataclass  
class VectorEntry:
    """
    Entry in the vector database.
    
    Attributes:
        id: Unique identifier
        embedding: Vector embedding
        metadata: Associated metadata
        created_at: Creation timestamp
    """
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class VectorStore(ABC):
    """
    Abstract base class for vector storage and similarity search.
    
    All vector stores must implement methods for adding, searching,
    and managing vector embeddings with associated metadata.
    """
    
    def __init__(
        self,
        dimension: int = 512,
        metric: str = "cosine",
        persist_path: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, inner_product)
            persist_path: Path for persistent storage
        """
        self.dimension = dimension
        self.metric = metric
        self.persist_path = persist_path
    
    @abstractmethod
    def add(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Add a single embedding to the store.
        
        Args:
            embedding: Vector embedding
            metadata: Associated metadata
            id: Optional ID (generated if not provided)
        
        Returns:
            ID of the added embedding
        """
        pass
    
    @abstractmethod
    def add_batch(
        self,
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add multiple embeddings to the store.
        
        Args:
            embeddings: Array of embeddings (N, dimension)
            metadatas: List of metadata dicts
            ids: Optional list of IDs
        
        Returns:
            List of IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Optional similarity threshold
            filter_metadata: Optional metadata filter
        
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            id: ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def delete_batch(self, ids: List[str]) -> int:
        """
        Delete multiple embeddings.
        
        Args:
            ids: List of IDs to delete
        
        Returns:
            Number of embeddings deleted
        """
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[VectorEntry]:
        """
        Get an entry by ID.
        
        Args:
            id: ID to retrieve
        
        Returns:
            VectorEntry or None if not found
        """
        pass
    
    @abstractmethod
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an entry.
        
        Args:
            id: ID to update
            metadata: New metadata (merged with existing)
        
        Returns:
            True if updated, False if not found
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the number of entries in the store."""
        pass
    
    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the store to disk.
        
        Args:
            path: Save path (uses persist_path if not provided)
        """
        pass
    
    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the store from disk.
        
        Args:
            path: Load path (uses persist_path if not provided)
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the store."""
        pass
    
    def generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())
    
    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate and normalize embedding dimension."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}"
                )
        elif embedding.ndim == 2:
            if embedding.shape[1] != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[1]}"
                )
        else:
            raise ValueError(f"Invalid embedding shape: {embedding.shape}")
        return embedding
    
    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score."""
        if self.metric == "cosine":
            # Cosine distance to similarity
            return 1.0 - distance
        elif self.metric == "inner_product":
            # Inner product is already similarity-like
            return distance
        else:
            # Euclidean: convert to similarity
            return 1.0 / (1.0 + distance)
    
    def __len__(self) -> int:
        return self.count()
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"count={self.count()})"
        )
