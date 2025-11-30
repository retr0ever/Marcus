"""
Identity Matcher
=================

Core matching logic for identity recognition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from marcus_core.matching.identity import Identity, IdentityStore
from marcus_core.vectordb.base import VectorStore, SearchResult


@dataclass
class MatchResult:
    """
    Result of an identity match.
    
    Attributes:
        identity: Matched identity
        score: Match confidence score [0, 1]
        distance: Embedding distance
        matched_embedding_idx: Index of the matched embedding
        search_results: Raw vector search results
    """
    identity: Identity
    score: float
    distance: float
    matched_embedding_idx: int = 0
    search_results: List[SearchResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "identity": self.identity.to_dict(),
            "score": self.score,
            "distance": self.distance,
            "matched_embedding_idx": self.matched_embedding_idx,
        }


class IdentityMatcher:
    """
    Identity matching engine using vector similarity search.
    
    Combines vector database search with identity store lookups
    to match face embeddings to known identities.
    
    Example:
        >>> matcher = IdentityMatcher(vector_store, identity_store)
        >>> results = matcher.match(query_embedding, top_k=5)
        >>> for result in results:
        ...     print(f"Match: {result.identity.name} ({result.score:.2%})")
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        identity_store: IdentityStore,
        similarity_threshold: float = 0.6,
        top_k: int = 10,
    ):
        """
        Initialize the matcher.
        
        Args:
            vector_store: Vector database for embedding search
            identity_store: Identity storage
            similarity_threshold: Minimum similarity for a match
            top_k: Number of candidates to retrieve
        """
        self.vector_store = vector_store
        self.identity_store = identity_store
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
    
    def match(
        self,
        embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_source: Optional[str] = None,
    ) -> List[MatchResult]:
        """
        Match an embedding against known identities.
        
        Args:
            embedding: Query face embedding
            top_k: Number of results (overrides default)
            threshold: Similarity threshold (overrides default)
            filter_source: Filter by identity source
        
        Returns:
            List of MatchResult objects, sorted by score descending
        """
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        # Search vector store
        filter_metadata = {"source": filter_source} if filter_source else None
        search_results = self.vector_store.search(
            embedding,
            top_k=top_k * 2,  # Get extra for filtering
            threshold=threshold,
            filter_metadata=filter_metadata,
        )
        
        if not search_results:
            return []
        
        # Group by identity
        identity_scores: Dict[str, List[SearchResult]] = {}
        for result in search_results:
            identity_id = result.metadata.get("identity_id")
            if identity_id:
                if identity_id not in identity_scores:
                    identity_scores[identity_id] = []
                identity_scores[identity_id].append(result)
        
        # Build match results
        matches = []
        for identity_id, results in identity_scores.items():
            identity = self.identity_store.get(identity_id)
            if identity is None:
                continue
            
            # Use best score for this identity
            best_result = max(results, key=lambda x: x.score)
            
            matches.append(MatchResult(
                identity=identity,
                score=best_result.score,
                distance=best_result.distance,
                search_results=results,
            ))
        
        # Sort by score descending
        matches.sort(key=lambda x: x.score, reverse=True)
        
        return matches[:top_k]
    
    def match_batch(
        self,
        embeddings: List[np.ndarray],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[List[MatchResult]]:
        """
        Match multiple embeddings.
        
        Args:
            embeddings: List of query embeddings
            top_k: Number of results per query
            threshold: Similarity threshold
        
        Returns:
            List of match results for each query
        """
        return [self.match(emb, top_k, threshold) for emb in embeddings]
    
    def enroll(
        self,
        identity: Identity,
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> str:
        """
        Enroll an identity with embeddings.
        
        Args:
            identity: Identity to enroll
            embeddings: Optional additional embeddings
        
        Returns:
            Identity ID
        """
        # Add embeddings to identity
        if embeddings:
            for emb in embeddings:
                identity.add_embedding(emb)
        
        # Add to identity store
        self.identity_store.add(identity)
        
        # Add embeddings to vector store
        for i, embedding in enumerate(identity.embeddings):
            self.vector_store.add(
                embedding=embedding,
                metadata={
                    "identity_id": identity.id,
                    "embedding_idx": i,
                    "source": identity.source,
                    "name": identity.name,
                },
                id=f"{identity.id}_{i}",
            )
        
        return identity.id
    
    def unenroll(self, identity_id: str) -> bool:
        """
        Remove an identity and its embeddings.
        
        Args:
            identity_id: Identity to remove
        
        Returns:
            True if removed, False if not found
        """
        identity = self.identity_store.get(identity_id)
        if identity is None:
            return False
        
        # Remove embeddings from vector store
        for i in range(identity.num_embeddings):
            self.vector_store.delete(f"{identity_id}_{i}")
        
        # Remove from identity store
        self.identity_store.delete(identity_id)
        
        return True
    
    def update_embeddings(
        self,
        identity_id: str,
        embeddings: List[np.ndarray],
        append: bool = True,
    ) -> bool:
        """
        Update embeddings for an identity.
        
        Args:
            identity_id: Identity to update
            embeddings: New embeddings
            append: If True, append to existing; if False, replace
        
        Returns:
            True if updated
        """
        identity = self.identity_store.get(identity_id)
        if identity is None:
            return False
        
        if not append:
            # Remove existing embeddings from vector store
            for i in range(identity.num_embeddings):
                self.vector_store.delete(f"{identity_id}_{i}")
            identity.embeddings = []
        
        # Add new embeddings
        start_idx = identity.num_embeddings
        for i, embedding in enumerate(embeddings):
            identity.add_embedding(embedding)
            self.vector_store.add(
                embedding=embedding,
                metadata={
                    "identity_id": identity.id,
                    "embedding_idx": start_idx + i,
                    "source": identity.source,
                    "name": identity.name,
                },
                id=f"{identity_id}_{start_idx + i}",
            )
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matcher statistics."""
        return {
            "total_identities": self.identity_store.count(),
            "total_embeddings": self.vector_store.count(),
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
        }
    
    def verify(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Verify if two embeddings belong to the same person.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Verification result with score and decision
        """
        # Compute similarity
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        similarity = float(np.dot(e1, e2))
        
        is_match = similarity >= self.similarity_threshold
        
        return {
            "similarity": similarity,
            "distance": 1.0 - similarity,
            "is_match": is_match,
            "threshold": self.similarity_threshold,
            "confidence": abs(similarity - self.similarity_threshold) / self.similarity_threshold,
        }
