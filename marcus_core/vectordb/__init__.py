"""
Vector Database Module
======================

Vector stores for efficient similarity search using FAISS and HNSWlib.
"""

from marcus_core.vectordb.base import VectorStore, SearchResult
from marcus_core.vectordb.faiss_store import FAISSStore

__all__ = [
    "VectorStore",
    "SearchResult",
    "FAISSStore",
]
