"""
Audit Logger
=============

Comprehensive audit logging for UK GDPR compliance.
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Data access
    SEARCH = "search"
    VIEW = "view"
    EXPORT = "export"
    
    # Data modification
    ENROLL = "enroll"
    UPDATE = "update"
    DELETE = "delete"
    
    # Consent
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"
    
    # System
    LOGIN = "login"
    LOGOUT = "logout"
    CONFIG_CHANGE = "config_change"
    
    # OSINT
    OSINT_INGEST = "osint_ingest"
    OSINT_FETCH = "osint_fetch"


@dataclass
class AuditEvent:
    """
    Represents a single audit log event.
    
    Attributes:
        id: Unique event identifier
        timestamp: Event timestamp (UTC)
        action: Type of action performed
        user_id: User who performed the action
        resource_type: Type of resource accessed
        resource_id: ID of the resource
        details: Additional event details
        ip_address: Client IP address
        session_id: Session identifier
        success: Whether the action succeeded
        error_message: Error message if failed
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action: str = ""
    user_id: str = "system"
    resource_type: str = ""
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            action=data.get("action", ""),
            user_id=data.get("user_id", "system"),
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id"),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            session_id=data.get("session_id"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


class AuditLogger:
    """
    Audit logger for UK GDPR compliance.
    
    Logs all data access, modifications, and system events
    with timestamps, user information, and action details.
    
    Example:
        >>> logger = AuditLogger(log_path="./data/audit_logs")
        >>> logger.log_search(user_id="user123", query_hash="abc123", results_count=5)
        >>> logger.log_enrollment(user_id="admin", identity_id="id456", source="linkedin")
    """
    
    def __init__(
        self,
        log_path: str = "./data/audit_logs",
        retention_days: int = 365,
        anonymize: bool = False,
        enabled: bool = True,
    ):
        """
        Initialize audit logger.
        
        Args:
            log_path: Directory for log files
            retention_days: How long to retain logs
            anonymize: Hash sensitive data in logs
            enabled: Whether logging is enabled
        """
        self.log_path = Path(log_path)
        self.retention_days = retention_days
        self.anonymize = anonymize
        self.enabled = enabled
        
        # Create log directory
        if self.enabled:
            self.log_path.mkdir(parents=True, exist_ok=True)
    
    def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.
        
        Args:
            event: Audit event to log
        """
        if not self.enabled:
            return
        
        # Anonymize if required
        if self.anonymize:
            event = self._anonymize_event(event)
        
        # Write to daily log file
        log_file = self._get_log_file(event.timestamp)
        
        with open(log_file, "a") as f:
            f.write(event.to_json() + "\n")
    
    def log_search(
        self,
        user_id: str,
        query_hash: str,
        results_count: int,
        top_match_id: Optional[str] = None,
        top_match_score: Optional[float] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log a search/match operation."""
        self.log(AuditEvent(
            action=AuditAction.SEARCH.value,
            user_id=user_id,
            resource_type="face_search",
            details={
                "query_hash": query_hash,
                "results_count": results_count,
                "top_match_id": top_match_id,
                "top_match_score": top_match_score,
            },
            ip_address=ip_address,
            session_id=session_id,
        ))
    
    def log_enrollment(
        self,
        user_id: str,
        identity_id: str,
        source: str,
        name: Optional[str] = None,
        num_embeddings: int = 1,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log an identity enrollment."""
        self.log(AuditEvent(
            action=AuditAction.ENROLL.value,
            user_id=user_id,
            resource_type="identity",
            resource_id=identity_id,
            details={
                "source": source,
                "name": name if not self.anonymize else self._hash(name),
                "num_embeddings": num_embeddings,
            },
            ip_address=ip_address,
        ))
    
    def log_deletion(
        self,
        user_id: str,
        identity_id: str,
        reason: str = "user_request",
        ip_address: Optional[str] = None,
    ) -> None:
        """Log an identity deletion."""
        self.log(AuditEvent(
            action=AuditAction.DELETE.value,
            user_id=user_id,
            resource_type="identity",
            resource_id=identity_id,
            details={"reason": reason},
            ip_address=ip_address,
        ))
    
    def log_view(
        self,
        user_id: str,
        identity_id: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log viewing identity details."""
        self.log(AuditEvent(
            action=AuditAction.VIEW.value,
            user_id=user_id,
            resource_type="identity",
            resource_id=identity_id,
            ip_address=ip_address,
        ))
    
    def log_export(
        self,
        user_id: str,
        export_type: str,
        record_count: int,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log a data export."""
        self.log(AuditEvent(
            action=AuditAction.EXPORT.value,
            user_id=user_id,
            resource_type="data_export",
            details={
                "export_type": export_type,
                "record_count": record_count,
            },
            ip_address=ip_address,
        ))
    
    def log_consent_change(
        self,
        user_id: str,
        identity_id: str,
        consent_granted: bool,
        purpose: str = "biometric_processing",
        ip_address: Optional[str] = None,
    ) -> None:
        """Log consent grant or revocation."""
        action = AuditAction.CONSENT_GRANTED if consent_granted else AuditAction.CONSENT_REVOKED
        self.log(AuditEvent(
            action=action.value,
            user_id=user_id,
            resource_type="consent",
            resource_id=identity_id,
            details={"purpose": purpose},
            ip_address=ip_address,
        ))
    
    def log_osint_ingest(
        self,
        user_id: str,
        source: str,
        record_count: int,
        source_url: Optional[str] = None,
    ) -> None:
        """Log OSINT data ingestion."""
        self.log(AuditEvent(
            action=AuditAction.OSINT_INGEST.value,
            user_id=user_id,
            resource_type="osint_data",
            details={
                "source": source,
                "record_count": record_count,
                "source_url": source_url if not self.anonymize else self._hash(source_url),
            },
        ))
    
    def _get_log_file(self, timestamp: datetime) -> Path:
        """Get log file path for a given date."""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.log_path / f"audit_{date_str}.jsonl"
    
    def _anonymize_event(self, event: AuditEvent) -> AuditEvent:
        """Anonymize sensitive data in an event."""
        event.user_id = self._hash(event.user_id)
        if event.ip_address:
            event.ip_address = self._hash(event.ip_address)
        return event
    
    def _hash(self, value: Optional[str]) -> Optional[str]:
        """Hash a value for anonymization."""
        if value is None:
            return None
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def query(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Query audit logs.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            action: Filter by action type
            user_id: Filter by user
            resource_id: Filter by resource
            limit: Maximum results
        
        Returns:
            List of matching audit events
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        events = []
        current = start_date
        
        while current <= end_date and len(events) < limit:
            log_file = self._get_log_file(current)
            
            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        if len(events) >= limit:
                            break
                        
                        try:
                            data = json.loads(line.strip())
                            event = AuditEvent.from_dict(data)
                            
                            # Apply filters
                            if action and event.action != action:
                                continue
                            if user_id and event.user_id != user_id:
                                continue
                            if resource_id and event.resource_id != resource_id:
                                continue
                            
                            events.append(event)
                        except json.JSONDecodeError:
                            continue
            
            current += timedelta(days=1)
        
        return events
    
    def export_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> str:
        """
        Export logs for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            format: Export format (json, csv)
        
        Returns:
            Exported data as string
        """
        events = self.query(start_date, end_date, limit=100000)
        
        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format == "csv":
            lines = ["timestamp,action,user_id,resource_type,resource_id,success"]
            for e in events:
                lines.append(
                    f"{e.timestamp.isoformat()},{e.action},{e.user_id},"
                    f"{e.resource_type},{e.resource_id or ''},{e.success}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def purge_old_logs(self) -> int:
        """
        Remove logs older than retention period.
        
        Returns:
            Number of files removed
        """
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        removed = 0
        
        for log_file in self.log_path.glob("audit_*.jsonl"):
            try:
                # Parse date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff:
                    log_file.unlink()
                    removed += 1
            except (ValueError, OSError):
                continue
        
        return removed
