"""
OSINT Base Classes
==================

Abstract base classes for OSINT data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator
from enum import Enum
import time
import threading


class SourceType(Enum):
    """Type of OSINT data source."""
    SOCIAL_MEDIA = "social_media"
    IMAGE_SEARCH = "image_search"
    DATASET = "dataset"
    WEB_SCRAPER = "web_scraper"
    API = "api"


class ResultStatus(Enum):
    """Status of an OSINT result."""
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"


@dataclass
class OSINTResult:
    """
    Result from an OSINT source.
    
    Attributes:
        source_name: Name of the source
        source_type: Type of source
        image_url: URL to the image
        image_data: Raw image bytes (if downloaded)
        profile_url: URL to the profile/page
        name: Detected name
        metadata: Additional metadata
        timestamp: When the result was collected
        status: Result status
        error: Error message if any
    """
    source_name: str
    source_type: SourceType
    image_url: Optional[str] = None
    image_data: Optional[bytes] = None
    profile_url: Optional[str] = None
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    status: ResultStatus = ResultStatus.SUCCESS
    error: Optional[str] = None
    
    @property
    def has_image(self) -> bool:
        """Check if image is available."""
        return self.image_url is not None or self.image_data is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "image_url": self.image_url,
            "has_image_data": self.image_data is not None,
            "profile_url": self.profile_url,
            "name": self.name,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "error": self.error,
        }


class RateLimiter:
    """
    Thread-safe rate limiter for API requests.
    
    Uses token bucket algorithm.
    """
    
    def __init__(
        self,
        requests_per_second: float = 1.0,
        burst_size: int = 5,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Sustained request rate
            burst_size: Maximum burst size
        """
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait (seconds)
        
        Returns:
            True if acquired, False if timeout
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + elapsed * self.rate,
                )
                self.last_update = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            
            # Check timeout
            if timeout is not None:
                if time.monotonic() - start_time >= timeout:
                    return False
            
            # Wait for a token
            wait_time = (1 - self.tokens) / self.rate
            time.sleep(min(wait_time, 0.1))
    
    def wait(self) -> None:
        """Wait until a request can be made."""
        self.acquire(timeout=None)


class OSINTSource(ABC):
    """
    Abstract base class for OSINT data sources.
    
    Implementations should handle rate limiting, error recovery,
    and ethical data collection.
    """
    
    def __init__(
        self,
        name: str,
        source_type: SourceType,
        rate_limit: float = 1.0,
        burst_size: int = 5,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OSINT source.
        
        Args:
            name: Source name
            source_type: Type of source
            rate_limit: Requests per second
            burst_size: Maximum burst
            enabled: Whether source is enabled
            config: Additional configuration
        """
        self.name = name
        self.source_type = source_type
        self.enabled = enabled
        self.config = config or {}
        
        self._rate_limiter = RateLimiter(rate_limit, burst_size)
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._request_count = 0
    
    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Search for faces matching a query.
        
        Args:
            query: Search query (name, keywords, etc.)
            max_results: Maximum results to return
        
        Returns:
            List of OSINT results
        """
        pass
    
    @abstractmethod
    def search_by_image(
        self,
        image: bytes,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Reverse image search.
        
        Args:
            image: Image bytes
            max_results: Maximum results
        
        Returns:
            List of OSINT results
        """
        pass
    
    def iter_results(
        self,
        query: str,
        max_results: int = 100,
    ) -> Generator[OSINTResult, None, None]:
        """
        Iterate through results with pagination.
        
        Args:
            query: Search query
            max_results: Maximum total results
        
        Yields:
            OSINT results
        """
        offset = 0
        page_size = min(10, max_results)
        
        while offset < max_results:
            results = self._search_page(query, offset, page_size)
            
            if not results:
                break
            
            for result in results:
                yield result
                offset += 1
                
                if offset >= max_results:
                    break
    
    def _search_page(
        self,
        query: str,
        offset: int,
        limit: int,
    ) -> List[OSINTResult]:
        """
        Search a single page. Override for pagination support.
        
        Default implementation calls search() once.
        """
        if offset > 0:
            return []
        return self.search(query, max_results=limit)
    
    def validate(self) -> bool:
        """Validate source configuration and connectivity."""
        return self.enabled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get source statistics."""
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "enabled": self.enabled,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
        }
    
    def _wait_rate_limit(self) -> None:
        """Wait for rate limit."""
        self._rate_limiter.wait()
    
    def _record_request(self) -> None:
        """Record a request."""
        self._request_count += 1
    
    def _record_error(self, error: str) -> None:
        """Record an error."""
        self._error_count += 1
        self._last_error = error
    
    def _create_error_result(self, error: str) -> OSINTResult:
        """Create error result."""
        self._record_error(error)
        return OSINTResult(
            source_name=self.name,
            source_type=self.source_type,
            status=ResultStatus.ERROR,
            error=error,
        )
    
    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"
