"""
FAISS Vector Store
===================

High-performance vector similarity search using Facebook AI Similarity Search.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from marcus_core.vectordb.base import VectorStore, SearchResult, VectorEntry


class FAISSStore(VectorStore):
    """
    FAISS-based vector store for efficient similarity search.
    
    Supports multiple index types:
    - Flat: Exact search, best accuracy, O(n) search
    - IVF_Flat: Approximate search with inverted file, faster for large datasets
    - IVF_PQ: Product quantization for memory efficiency
    - HNSW: Graph-based approximate search
    
    Example:
        >>> store = FAISSStore(dimension=512, index_type="IVF_Flat")
        >>> id = store.add(embedding, {"name": "John Doe"})
        >>> results = store.search(query_embedding, top_k=5)
    """
    
    INDEX_TYPES = ["Flat", "IVF_Flat", "IVF_PQ", "HNSW"]
    
    def __init__(
        self,
        dimension: int = 512,
        metric: str = "cosine",
        index_type: str = "Flat",
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 8,  # PQ subquantizers
        nbits: int = 8,  # PQ bits per subquantizer
        persist_path: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        Initialize FAISS store.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, inner_product)
            index_type: FAISS index type (Flat, IVF_Flat, IVF_PQ, HNSW)
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search
            m: Number of subquantizers for PQ
            nbits: Bits per subquantizer for PQ
            persist_path: Path for saving/loading
            auto_save: Automatically save after modifications
        """
        super().__init__(dimension=dimension, metric=metric, persist_path=persist_path)
        
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.auto_save = auto_save
        
        # Storage
        self._index = None
        self._id_map: Dict[int, str] = {}  # FAISS internal ID -> string ID
        self._reverse_id_map: Dict[str, int] = {}  # string ID -> FAISS internal ID
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}  # Store embeddings for retrieval
        self._next_id = 0
        self._is_trained = False
        
        # Initialize index
        self._init_index()
    
    def _init_index(self):
        """Initialize the FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss package is required. "
                "Install with: pip install faiss-cpu or faiss-gpu"
            )
        
        # Determine metric
        if self.metric in ["cosine", "inner_product"]:
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2
        
        # Create index based on type
        if self.index_type == "Flat":
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                self._index = faiss.IndexFlatIP(self.dimension)
            else:
                self._index = faiss.IndexFlatL2(self.dimension)
            self._is_trained = True
            
        elif self.index_type == "IVF_Flat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, metric_type
            )
            
        elif self.index_type == "IVF_PQ":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, self.m, self.nbits
            )
            
        elif self.index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(self.dimension, 32)  # M=32
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                # Wrap for inner product
                self._index = faiss.IndexIDMap(self._index)
            self._is_trained = True
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Set search parameters
        if hasattr(self._index, 'nprobe'):
            self._index.nprobe = self.nprobe
    
    def _normalize_for_cosine(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity (used with inner product)."""
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / (norms + 1e-10)
        return vectors
    
    def _train_if_needed(self, embeddings: np.ndarray):
        """Train index if required (for IVF indices)."""
        if self._is_trained:
            return
        
        import faiss
        
        # Need enough samples to train
        min_samples = max(self.nlist * 39, 256)
        if len(embeddings) < min_samples:
            # Not enough data yet, use Flat index temporarily
            return
        
        # Train the index
        training_data = self._normalize_for_cosine(embeddings.astype(np.float32))
        self._index.train(training_data)
        self._is_trained = True
    
    def add(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """Add a single embedding."""
        embedding = self._validate_embedding(embedding)
        
        if id is None:
            id = self.generate_id()
        
        # Check for duplicate
        if id in self._reverse_id_map:
            # Update existing
            self.delete(id)
        
        # Prepare embedding
        embedding = embedding.reshape(1, -1).astype(np.float32)
        embedding = self._normalize_for_cosine(embedding)
        
        # Train if needed (for IVF)
        if not self._is_trained:
            # Collect embeddings for training
            all_embeddings = np.vstack(
                [embedding] + [e.reshape(1, -1) for e in self._embeddings.values()]
            ) if self._embeddings else embedding
            self._train_if_needed(all_embeddings)
        
        # Add to index
        if self._is_trained:
            self._index.add(embedding)
        
        # Update mappings
        internal_id = self._next_id
        self._id_map[internal_id] = id
        self._reverse_id_map[id] = internal_id
        self._embeddings[id] = embedding.flatten()
        self._metadata[id] = metadata or {}
        self._next_id += 1
        
        # Auto-save
        if self.auto_save and self.persist_path:
            self.save()
        
        return id
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add multiple embeddings."""
        embeddings = self._validate_embedding(embeddings)
        n = len(embeddings)
        
        if ids is None:
            ids = [self.generate_id() for _ in range(n)]
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        
        # Prepare embeddings
        embeddings = embeddings.astype(np.float32)
        embeddings = self._normalize_for_cosine(embeddings)
        
        # Train if needed
        if not self._is_trained:
            self._train_if_needed(embeddings)
        
        # Add to index
        if self._is_trained:
            self._index.add(embeddings)
        
        # Update mappings
        added_ids = []
        for i, (id, metadata) in enumerate(zip(ids, metadatas)):
            if id in self._reverse_id_map:
                continue  # Skip duplicates
            
            internal_id = self._next_id
            self._id_map[internal_id] = id
            self._reverse_id_map[id] = internal_id
            self._embeddings[id] = embeddings[i]
            self._metadata[id] = metadata
            self._next_id += 1
            added_ids.append(id)
        
        # Auto-save
        if self.auto_save and self.persist_path:
            self.save()
        
        return added_ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        if self.count() == 0:
            return []
        
        query = self._validate_embedding(query_embedding)
        query = query.reshape(1, -1).astype(np.float32)
        query = self._normalize_for_cosine(query)
        
        # Search
        k = min(top_k * 2, self.count())  # Get extra for filtering
        distances, indices = self._index.search(query, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            
            id = self._id_map.get(idx)
            if id is None:
                continue
            
            metadata = self._metadata.get(id, {})
            
            # Apply metadata filter
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            # Calculate score
            if self.metric in ["cosine", "inner_product"]:
                score = float(dist)  # Already similarity
                distance = 1.0 - score
            else:
                distance = float(dist)
                score = 1.0 / (1.0 + distance)
            
            # Apply threshold
            if threshold is not None and score < threshold:
                continue
            
            results.append(SearchResult(
                id=id,
                score=score,
                distance=distance,
                metadata=metadata,
                embedding=self._embeddings.get(id),
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete(self, id: str) -> bool:
        """Delete an embedding by ID."""
        if id not in self._reverse_id_map:
            return False
        
        # Remove from mappings
        internal_id = self._reverse_id_map.pop(id)
        del self._id_map[internal_id]
        del self._metadata[id]
        del self._embeddings[id]
        
        # Note: FAISS doesn't support direct deletion for most index types
        # For production use, consider using IndexIDMap or rebuilding periodically
        
        if self.auto_save and self.persist_path:
            self.save()
        
        return True
    
    def delete_batch(self, ids: List[str]) -> int:
        """Delete multiple embeddings."""
        count = 0
        for id in ids:
            if self.delete(id):
                count += 1
        return count
    
    def get(self, id: str) -> Optional[VectorEntry]:
        """Get an entry by ID."""
        if id not in self._reverse_id_map:
            return None
        
        return VectorEntry(
            id=id,
            embedding=self._embeddings[id],
            metadata=self._metadata[id],
        )
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for an entry."""
        if id not in self._reverse_id_map:
            return False
        
        self._metadata[id].update(metadata)
        
        if self.auto_save and self.persist_path:
            self.save()
        
        return True
    
    def count(self) -> int:
        """Get number of entries."""
        return len(self._reverse_id_map)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the store to disk."""
        import faiss
        
        path = path or self.persist_path
        if path is None:
            raise ValueError("No save path specified")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._is_trained and self._index.ntotal > 0:
            faiss.write_index(self._index, str(path / "index.faiss"))
        
        # Save metadata and mappings
        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "reverse_id_map": self._reverse_id_map,
            "metadata": self._metadata,
            "next_id": self._next_id,
            "is_trained": self._is_trained,
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings
        if self._embeddings:
            embeddings_array = np.array([self._embeddings[id] for id in self._embeddings])
            embedding_ids = list(self._embeddings.keys())
            np.savez(
                path / "embeddings.npz",
                embeddings=embeddings_array,
                ids=embedding_ids,
            )
    
    def load(self, path: Optional[str] = None) -> None:
        """Load the store from disk."""
        import faiss
        
        path = path or self.persist_path
        if path is None:
            raise ValueError("No load path specified")
        
        path = Path(path)
        if not path.exists():
            return  # Nothing to load
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
            
            self.dimension = data.get("dimension", self.dimension)
            self.metric = data.get("metric", self.metric)
            self.index_type = data.get("index_type", self.index_type)
            self._id_map = {int(k): v for k, v in data.get("id_map", {}).items()}
            self._reverse_id_map = data.get("reverse_id_map", {})
            self._metadata = data.get("metadata", {})
            self._next_id = data.get("next_id", 0)
            self._is_trained = data.get("is_trained", False)
        
        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            if hasattr(self._index, 'nprobe'):
                self._index.nprobe = self.nprobe
        else:
            self._init_index()
        
        # Load embeddings
        embeddings_path = path / "embeddings.npz"
        if embeddings_path.exists():
            data = np.load(embeddings_path, allow_pickle=True)
            embeddings = data["embeddings"]
            ids = data["ids"]
            self._embeddings = {id: emb for id, emb in zip(ids, embeddings)}
        
        # If we have embeddings but no index, rebuild it
        if self._embeddings and self._index.ntotal == 0:
            self.rebuild_index()
    
    def clear(self) -> None:
        """Clear all entries."""
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._metadata.clear()
        self._embeddings.clear()
        self._next_id = 0
        self._is_trained = False
        self._init_index()
        
        if self.auto_save and self.persist_path:
            self.save()
    
    def rebuild_index(self) -> None:
        """Rebuild the FAISS index from stored embeddings."""
        if not self._embeddings:
            return
        
        # Re-initialise creates fresh index and sets _is_trained for Flat
        self._init_index()
        
        # Re-add all embeddings
        ids = list(self._embeddings.keys())
        embeddings = np.array([self._embeddings[id] for id in ids])
        embeddings = self._normalize_for_cosine(embeddings.astype(np.float32))
        
        # For Flat index, _is_trained is already True from _init_index
        # For IVF/PQ, we need to train
        if not self._is_trained:
            self._train_if_needed(embeddings)
        
        # Add embeddings to index
        if self._is_trained:
            self._index.add(embeddings)
        
        # Rebuild mappings
        self._id_map = {i: id for i, id in enumerate(ids)}
        self._reverse_id_map = {id: i for i, id in enumerate(ids)}
        self._next_id = len(ids)
