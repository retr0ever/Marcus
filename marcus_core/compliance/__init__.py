"""
Compliance Module
=================

UK GDPR compliance features including audit logging,
consent management, and data retention policies.
"""

from marcus_core.compliance.audit_logger import AuditLogger, AuditEvent
from marcus_core.compliance.consent_manager import ConsentManager, ConsentRecord

__all__ = [
    "AuditLogger",
    "AuditEvent",
    "ConsentManager",
    "ConsentRecord",
]
