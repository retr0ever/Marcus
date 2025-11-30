"""
OSINT Data Collection Module
============================

Adapters for collecting facial data from public sources.
"""

from marcus_core.osint.base import OSINTSource, OSINTResult, RateLimiter
from marcus_core.osint.source_registry import SourceRegistry
from marcus_core.osint.web_image_adapter import WebImageAdapter
from marcus_core.osint.dataset_adapter import DatasetAdapter

__all__ = [
    "OSINTSource",
    "OSINTResult",
    "RateLimiter",
    "SourceRegistry",
    "WebImageAdapter",
    "DatasetAdapter",
]
