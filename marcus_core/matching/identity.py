"""
Identity Data Structures
=========================

Data models for identity representation and storage.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class Identity:
    """
    Represents a known identity with associated embeddings and metadata.
    
    Attributes:
        id: Unique identifier (UUID)
        name: Display name
        embeddings: List of face embeddings for this identity
        metadata: Additional profile information
        source: Data source (linkedin, manual, etc.)
        consent_status: Consent tracking (granted, pending, revoked)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    embeddings: List[np.ndarray] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    consent_status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Optional profile fields (stored in metadata but with shortcuts)
    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title")
    
    @property
    def company(self) -> Optional[str]:
        return self.metadata.get("company")
    
    @property
    def location(self) -> Optional[str]:
        return self.metadata.get("location")
    
    @property
    def profile_url(self) -> Optional[str]:
        return self.metadata.get("profile_url")
    
    @property
    def image_urls(self) -> List[str]:
        return self.metadata.get("image_urls", [])
    
    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)
    
    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add a new embedding for this identity."""
        self.embeddings.append(embedding)
        self.updated_at = datetime.now()
    
    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """Get the mean of all embeddings (centroid)."""
        if not self.embeddings:
            return None
        stacked = np.vstack(self.embeddings)
        mean = np.mean(stacked, axis=0)
        # Normalize
        return mean / (np.linalg.norm(mean) + 1e-10)
    
    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "source": self.source,
            "consent_status": self.consent_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "num_embeddings": self.num_embeddings,
        }
        if include_embeddings:
            data["embeddings"] = [e.tolist() for e in self.embeddings]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Identity":
        """Create from dictionary."""
        embeddings = []
        if "embeddings" in data:
            embeddings = [np.array(e, dtype=np.float32) for e in data["embeddings"]]
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name"),
            embeddings=embeddings,
            metadata=data.get("metadata", {}),
            source=data.get("source", "manual"),
            consent_status=data.get("consent_status", "pending"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


class IdentityStore:
    """
    Storage and management of identities.
    
    Provides CRUD operations and persistence for identity records.
    
    Example:
        >>> store = IdentityStore(persist_path="./data/identities")
        >>> identity = Identity(name="John Doe")
        >>> identity.add_embedding(embedding)
        >>> store.add(identity)
        >>> store.save()
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize identity store.
        
        Args:
            persist_path: Directory for persistent storage
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self._identities: Dict[str, Identity] = {}
        
        # Load existing data
        if self.persist_path and self.persist_path.exists():
            self.load()
    
    def add(self, identity: Identity) -> str:
        """
        Add or update an identity.
        
        Args:
            identity: Identity to add
        
        Returns:
            Identity ID
        """
        self._identities[identity.id] = identity
        return identity.id
    
    def get(self, id: str) -> Optional[Identity]:
        """Get an identity by ID."""
        return self._identities.get(id)
    
    def get_by_name(self, name: str) -> List[Identity]:
        """Get identities by name (partial match)."""
        name_lower = name.lower()
        return [
            identity for identity in self._identities.values()
            if identity.name and name_lower in identity.name.lower()
        ]
    
    def delete(self, id: str) -> bool:
        """Delete an identity."""
        if id in self._identities:
            del self._identities[id]
            return True
        return False
    
    def list_all(self) -> List[Identity]:
        """List all identities."""
        return list(self._identities.values())
    
    def count(self) -> int:
        """Get number of identities."""
        return len(self._identities)
    
    def search(
        self,
        query: Optional[str] = None,
        source: Optional[str] = None,
        consent_status: Optional[str] = None,
    ) -> List[Identity]:
        """
        Search identities with filters.
        
        Args:
            query: Search query for name/metadata
            source: Filter by source
            consent_status: Filter by consent status
        
        Returns:
            List of matching identities
        """
        results = list(self._identities.values())
        
        if source:
            results = [i for i in results if i.source == source]
        
        if consent_status:
            results = [i for i in results if i.consent_status == consent_status]
        
        if query:
            query_lower = query.lower()
            filtered = []
            for identity in results:
                # Search in name
                if identity.name and query_lower in identity.name.lower():
                    filtered.append(identity)
                    continue
                # Search in metadata
                for value in identity.metadata.values():
                    if isinstance(value, str) and query_lower in value.lower():
                        filtered.append(identity)
                        break
            results = filtered
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """Save identities to disk."""
        path = Path(path) if path else self.persist_path
        if path is None:
            raise ValueError("No save path specified")
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each identity
        for identity in self._identities.values():
            identity_path = path / f"{identity.id}.json"
            with open(identity_path, "w") as f:
                json.dump(identity.to_dict(include_embeddings=True), f, indent=2)
        
        # Save index
        index = {
            "identities": list(self._identities.keys()),
            "count": len(self._identities),
            "saved_at": datetime.now().isoformat(),
        }
        with open(path / "index.json", "w") as f:
            json.dump(index, f, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load identities from disk."""
        path = Path(path) if path else self.persist_path
        if path is None or not path.exists():
            return
        
        # Load from individual files
        for file_path in path.glob("*.json"):
            if file_path.name == "index.json":
                continue
            
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                identity = Identity.from_dict(data)
                self._identities[identity.id] = identity
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    
    def export(self, format: str = "json") -> str:
        """Export all identities to string."""
        if format == "json":
            data = {
                "identities": [i.to_dict(include_embeddings=False) for i in self._identities.values()],
                "exported_at": datetime.now().isoformat(),
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear(self) -> None:
        """Clear all identities."""
        self._identities.clear()
    
    def __len__(self) -> int:
        return len(self._identities)
    
    def __iter__(self):
        return iter(self._identities.values())
