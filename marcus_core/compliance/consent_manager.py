"""
Consent Manager
================

Consent tracking and verification for UK GDPR compliance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class ConsentStatus(str, Enum):
    """Consent status values."""
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ConsentPurpose(str, Enum):
    """Purposes for which consent may be granted."""
    BIOMETRIC_PROCESSING = "biometric_processing"
    IDENTITY_MATCHING = "identity_matching"
    DATA_STORAGE = "data_storage"
    OSINT_INGESTION = "osint_ingestion"
    TRAINING = "training"
    EXPORT = "export"


@dataclass
class ConsentRecord:
    """
    Record of consent for data processing.
    
    Attributes:
        id: Unique consent record ID
        identity_id: ID of the identity this consent applies to
        purpose: Purpose of consent
        status: Current consent status
        granted_at: When consent was granted
        expires_at: When consent expires
        revoked_at: When consent was revoked (if applicable)
        granted_by: User who granted consent
        source: Where consent was obtained
        proof: Evidence of consent (e.g., signed form reference)
        metadata: Additional consent details
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    identity_id: str = ""
    purpose: str = ConsentPurpose.BIOMETRIC_PROCESSING.value
    status: str = ConsentStatus.PENDING.value
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    granted_by: Optional[str] = None
    source: str = "manual"
    proof: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED.value:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "identity_id": self.identity_id,
            "purpose": self.purpose,
            "status": self.status,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "granted_by": self.granted_by,
            "source": self.source,
            "proof": self.proof,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentRecord":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            identity_id=data.get("identity_id", ""),
            purpose=data.get("purpose", ConsentPurpose.BIOMETRIC_PROCESSING.value),
            status=data.get("status", ConsentStatus.PENDING.value),
            granted_at=datetime.fromisoformat(data["granted_at"]) if data.get("granted_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            revoked_at=datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None,
            granted_by=data.get("granted_by"),
            source=data.get("source", "manual"),
            proof=data.get("proof"),
            metadata=data.get("metadata", {}),
        )


class ConsentManager:
    """
    Manages consent records for UK GDPR compliance.
    
    Tracks consent status per identity and purpose, enforces
    expiry, and provides verification for data processing.
    
    Example:
        >>> manager = ConsentManager(persist_path="./data/consent")
        >>> manager.grant_consent(identity_id="123", purpose="biometric_processing", granted_by="admin")
        >>> if manager.has_valid_consent("123", "biometric_processing"):
        ...     # Process data
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        default_expiry_days: int = 365,
        require_explicit_consent: bool = True,
    ):
        """
        Initialize consent manager.
        
        Args:
            persist_path: Directory for consent records
            default_expiry_days: Default consent validity period
            require_explicit_consent: Require consent before processing
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self.default_expiry_days = default_expiry_days
        self.require_explicit_consent = require_explicit_consent
        
        # Storage: identity_id -> purpose -> ConsentRecord
        self._consents: Dict[str, Dict[str, ConsentRecord]] = {}
        
        # Load existing records
        if self.persist_path and self.persist_path.exists():
            self.load()
    
    def grant_consent(
        self,
        identity_id: str,
        purpose: str = ConsentPurpose.BIOMETRIC_PROCESSING.value,
        granted_by: str = "system",
        expiry_days: Optional[int] = None,
        source: str = "manual",
        proof: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentRecord:
        """
        Grant consent for an identity.
        
        Args:
            identity_id: Identity to grant consent for
            purpose: Purpose of consent
            granted_by: User granting consent
            expiry_days: Days until expiry (uses default if None)
            source: Source of consent
            proof: Proof reference
            metadata: Additional details
        
        Returns:
            Created consent record
        """
        expiry_days = expiry_days or self.default_expiry_days
        now = datetime.utcnow()
        
        record = ConsentRecord(
            identity_id=identity_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED.value,
            granted_at=now,
            expires_at=now + timedelta(days=expiry_days),
            granted_by=granted_by,
            source=source,
            proof=proof,
            metadata=metadata or {},
        )
        
        # Store
        if identity_id not in self._consents:
            self._consents[identity_id] = {}
        self._consents[identity_id][purpose] = record
        
        # Persist
        if self.persist_path:
            self.save()
        
        return record
    
    def revoke_consent(
        self,
        identity_id: str,
        purpose: Optional[str] = None,
        revoked_by: str = "system",
        reason: str = "",
    ) -> List[ConsentRecord]:
        """
        Revoke consent for an identity.
        
        Args:
            identity_id: Identity to revoke consent for
            purpose: Specific purpose to revoke (all if None)
            revoked_by: User revoking consent
            reason: Reason for revocation
        
        Returns:
            List of revoked consent records
        """
        if identity_id not in self._consents:
            return []
        
        revoked = []
        now = datetime.utcnow()
        
        purposes = [purpose] if purpose else list(self._consents[identity_id].keys())
        
        for p in purposes:
            if p in self._consents[identity_id]:
                record = self._consents[identity_id][p]
                record.status = ConsentStatus.REVOKED.value
                record.revoked_at = now
                record.metadata["revoked_by"] = revoked_by
                record.metadata["revocation_reason"] = reason
                revoked.append(record)
        
        if self.persist_path:
            self.save()
        
        return revoked
    
    def has_valid_consent(
        self,
        identity_id: str,
        purpose: str = ConsentPurpose.BIOMETRIC_PROCESSING.value,
    ) -> bool:
        """
        Check if identity has valid consent for a purpose.
        
        Args:
            identity_id: Identity to check
            purpose: Purpose to check consent for
        
        Returns:
            True if valid consent exists
        """
        if not self.require_explicit_consent:
            return True
        
        if identity_id not in self._consents:
            return False
        
        if purpose not in self._consents[identity_id]:
            return False
        
        record = self._consents[identity_id][purpose]
        return record.is_valid()
    
    def get_consent(
        self,
        identity_id: str,
        purpose: Optional[str] = None,
    ) -> Optional[ConsentRecord] | Dict[str, ConsentRecord]:
        """
        Get consent record(s) for an identity.
        
        Args:
            identity_id: Identity to look up
            purpose: Specific purpose (all if None)
        
        Returns:
            ConsentRecord or dict of records
        """
        if identity_id not in self._consents:
            return None if purpose else {}
        
        if purpose:
            return self._consents[identity_id].get(purpose)
        
        return self._consents[identity_id].copy()
    
    def verify_consent(
        self,
        identity_id: str,
        purposes: List[str],
    ) -> Dict[str, bool]:
        """
        Verify consent for multiple purposes.
        
        Args:
            identity_id: Identity to verify
            purposes: List of purposes to check
        
        Returns:
            Dict mapping purpose to consent status
        """
        return {
            purpose: self.has_valid_consent(identity_id, purpose)
            for purpose in purposes
        }
    
    def get_expiring_soon(
        self,
        days: int = 30,
    ) -> List[ConsentRecord]:
        """
        Get consents expiring within specified days.
        
        Args:
            days: Days to look ahead
        
        Returns:
            List of expiring consent records
        """
        cutoff = datetime.utcnow() + timedelta(days=days)
        expiring = []
        
        for identity_consents in self._consents.values():
            for record in identity_consents.values():
                if record.is_valid() and record.expires_at and record.expires_at < cutoff:
                    expiring.append(record)
        
        return sorted(expiring, key=lambda x: x.expires_at or datetime.max)
    
    def refresh_consent(
        self,
        identity_id: str,
        purpose: str,
        expiry_days: Optional[int] = None,
    ) -> Optional[ConsentRecord]:
        """
        Refresh/extend consent expiry.
        
        Args:
            identity_id: Identity to refresh
            purpose: Purpose to refresh
            expiry_days: New expiry period
        
        Returns:
            Updated consent record or None
        """
        if identity_id not in self._consents:
            return None
        
        if purpose not in self._consents[identity_id]:
            return None
        
        record = self._consents[identity_id][purpose]
        if record.status != ConsentStatus.GRANTED.value:
            return None
        
        expiry_days = expiry_days or self.default_expiry_days
        record.expires_at = datetime.utcnow() + timedelta(days=expiry_days)
        
        if self.persist_path:
            self.save()
        
        return record
    
    def delete_all_for_identity(self, identity_id: str) -> int:
        """
        Delete all consent records for an identity.
        
        Used for right-to-erasure requests.
        
        Args:
            identity_id: Identity to delete records for
        
        Returns:
            Number of records deleted
        """
        if identity_id not in self._consents:
            return 0
        
        count = len(self._consents[identity_id])
        del self._consents[identity_id]
        
        if self.persist_path:
            self.save()
        
        return count
    
    def save(self, path: Optional[str] = None) -> None:
        """Save consent records to disk."""
        path = Path(path) if path else self.persist_path
        if path is None:
            return
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {}
        for identity_id, purposes in self._consents.items():
            data[identity_id] = {
                purpose: record.to_dict()
                for purpose, record in purposes.items()
            }
        
        with open(path / "consents.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load consent records from disk."""
        path = Path(path) if path else self.persist_path
        if path is None:
            return
        
        consent_file = path / "consents.json"
        if not consent_file.exists():
            return
        
        with open(consent_file, "r") as f:
            data = json.load(f)
        
        for identity_id, purposes in data.items():
            self._consents[identity_id] = {
                purpose: ConsentRecord.from_dict(record_data)
                for purpose, record_data in purposes.items()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consent statistics."""
        total = 0
        by_status = {s.value: 0 for s in ConsentStatus}
        by_purpose = {p.value: 0 for p in ConsentPurpose}
        
        for identity_consents in self._consents.values():
            for record in identity_consents.values():
                total += 1
                by_status[record.status] = by_status.get(record.status, 0) + 1
                by_purpose[record.purpose] = by_purpose.get(record.purpose, 0) + 1
        
        return {
            "total_records": total,
            "by_status": by_status,
            "by_purpose": by_purpose,
            "identities_with_consent": len(self._consents),
        }
