"""
Test Identity Matching Module
=============================
"""

import pytest
import numpy as np
from datetime import datetime

from marcus_core.matching.identity import Identity, IdentityStore


class TestIdentity:
    """Tests for Identity dataclass."""
    
    def test_create_identity(self):
        """Test creating an identity."""
        identity = Identity(name="John Doe")
        
        assert identity.name == "John Doe"
        assert identity.id is not None
        assert len(identity.id) == 36  # UUID format
        assert identity.embeddings == []
        assert identity.metadata == {}
    
    def test_identity_with_embeddings(self):
        """Test identity with embeddings."""
        embedding = np.random.randn(512).astype(np.float32)
        identity = Identity(
            name="Jane Doe",
            embeddings=[embedding],
        )
        
        assert len(identity.embeddings) == 1
        assert identity.embeddings[0].shape == (512,)
    
    def test_add_embedding(self):
        """Test adding embedding to identity."""
        identity = Identity(name="Test")
        embedding = np.random.randn(512).astype(np.float32)
        
        identity.add_embedding(embedding)
        
        assert len(identity.embeddings) == 1
    
    def test_average_embedding(self):
        """Test computing average embedding."""
        identity = Identity(name="Test")
        
        emb1 = np.ones(512, dtype=np.float32)
        emb2 = np.ones(512, dtype=np.float32) * 3
        
        identity.add_embedding(emb1)
        identity.add_embedding(emb2)
        
        avg = identity.average_embedding
        
        assert avg is not None
        np.testing.assert_array_almost_equal(avg, np.ones(512) * 2)
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        identity = Identity(
            name="Test User",
            metadata={"company": "Acme"},
            source="manual",
        )
        
        d = identity.to_dict()
        
        assert d["name"] == "Test User"
        assert d["source"] == "manual"
        assert d["metadata"]["company"] == "Acme"
        assert "id" in d
        assert "created_at" in d
    
    def test_from_dict(self):
        """Test dictionary deserialization."""
        data = {
            "id": "test-id-123",
            "name": "Test User",
            "source": "dataset",
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00",
            "embeddings": [],
        }
        
        identity = Identity.from_dict(data)
        
        assert identity.id == "test-id-123"
        assert identity.name == "Test User"
        assert identity.source == "dataset"


class TestIdentityStore:
    """Tests for IdentityStore."""
    
    def test_add_get_identity(self):
        """Test adding and retrieving identity."""
        store = IdentityStore()
        
        identity = Identity(name="Test User")
        store.add(identity)
        
        retrieved = store.get(identity.id)
        
        assert retrieved is not None
        assert retrieved.name == "Test User"
        assert retrieved.id == identity.id
    
    def test_get_nonexistent(self):
        """Test getting non-existent identity."""
        store = IdentityStore()
        
        result = store.get("nonexistent-id")
        
        assert result is None
    
    def test_delete_identity(self):
        """Test deleting identity."""
        store = IdentityStore()
        
        identity = Identity(name="To Delete")
        store.add(identity)
        
        assert store.delete(identity.id) is True
        assert store.get(identity.id) is None
    
    def test_delete_nonexistent(self):
        """Test deleting non-existent identity."""
        store = IdentityStore()
        
        assert store.delete("nonexistent") is False
    
    def test_search_by_name(self):
        """Test searching by name."""
        store = IdentityStore()
        
        store.add(Identity(name="John Doe"))
        store.add(Identity(name="Jane Doe"))
        store.add(Identity(name="Bob Smith"))
        
        results = store.search(query="Doe")
        
        assert len(results) == 2
        assert all("Doe" in r.name for r in results)
    
    def test_search_by_source(self):
        """Test searching by source."""
        store = IdentityStore()
        
        store.add(Identity(name="User1", source="manual"))
        store.add(Identity(name="User2", source="dataset"))
        store.add(Identity(name="User3", source="manual"))
        
        results = store.search(source="manual")
        
        assert len(results) == 2
        assert all(r.source == "manual" for r in results)
    
    def test_count(self):
        """Test counting identities."""
        store = IdentityStore()
        
        assert store.count() == 0
        
        store.add(Identity(name="User1"))
        store.add(Identity(name="User2"))
        
        assert store.count() == 2
    
    def test_list_all(self):
        """Test listing all identities."""
        store = IdentityStore()
        
        store.add(Identity(name="User1"))
        store.add(Identity(name="User2"))
        store.add(Identity(name="User3"))
        
        all_identities = store.list_all()
        
        assert len(all_identities) == 3
