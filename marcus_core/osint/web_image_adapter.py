"""
Web Image Adapter
=================

Adapter for collecting images from web sources.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
import io
import hashlib
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from marcus_core.osint.base import (
    OSINTSource,
    OSINTResult,
    SourceType,
    ResultStatus,
)


logger = logging.getLogger(__name__)


class WebImageAdapter(OSINTSource):
    """
    Adapter for fetching images from web URLs.
    
    This is a generic adapter for downloading images from URLs.
    Can be extended for specific image search engines.
    
    Example:
        >>> adapter = WebImageAdapter()
        >>> results = adapter.fetch_urls([
        ...     "https://example.com/image1.jpg",
        ...     "https://example.com/image2.jpg",
        ... ])
    """
    
    def __init__(
        self,
        name: str = "web_images",
        user_agent: Optional[str] = None,
        timeout: float = 10.0,
        rate_limit: float = 2.0,  # 2 requests per second
        max_image_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_formats: Optional[List[str]] = None,
        download_images: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize web image adapter.
        
        Args:
            name: Source name
            user_agent: HTTP user agent
            timeout: Request timeout
            rate_limit: Requests per second
            max_image_size: Maximum image size in bytes
            allowed_formats: Allowed image formats
            download_images: Whether to download image bytes
            cache_dir: Directory to cache downloaded images
            **kwargs: Additional config
        """
        super().__init__(
            name=name,
            source_type=SourceType.WEB_SCRAPER,
            rate_limit=rate_limit,
            **kwargs,
        )
        
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; MarcusBot/1.0; "
            "+https://github.com/marcus-osint)"
        )
        self.timeout = timeout
        self.max_image_size = max_image_size
        self.allowed_formats = allowed_formats or ["jpg", "jpeg", "png", "webp"]
        self.download_images = download_images
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        self._session: Optional[requests.Session] = None
    
    @property
    def session(self) -> "requests.Session":
        """Get or create requests session."""
        if not HAS_REQUESTS:
            raise ImportError("requests not installed")
        
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self.user_agent,
            })
        
        return self._session
    
    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Search for images. Not implemented for generic web adapter.
        
        Override in subclasses for search engine integration.
        """
        logger.warning(f"search() not implemented for {self.name}")
        return []
    
    def search_by_image(
        self,
        image: bytes,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Reverse image search. Not implemented for generic web adapter.
        
        Override in subclasses for reverse image search integration.
        """
        logger.warning(f"search_by_image() not implemented for {self.name}")
        return []
    
    def fetch_urls(
        self,
        urls: List[str],
        download: Optional[bool] = None,
    ) -> List[OSINTResult]:
        """
        Fetch images from URLs.
        
        Args:
            urls: List of image URLs
            download: Override download_images setting
        
        Returns:
            List of results
        """
        download = download if download is not None else self.download_images
        results = []
        
        for url in urls:
            self._wait_rate_limit()
            self._record_request()
            
            try:
                result = self._fetch_url(url, download)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                results.append(self._create_error_result(str(e)))
        
        return results
    
    def _fetch_url(
        self,
        url: str,
        download: bool,
    ) -> OSINTResult:
        """Fetch a single URL."""
        # Check cache first
        if self.cache_dir and download:
            cached = self._get_cached(url)
            if cached is not None:
                return OSINTResult(
                    source_name=self.name,
                    source_type=self.source_type,
                    image_url=url,
                    image_data=cached,
                    status=ResultStatus.SUCCESS,
                    metadata={"cached": True},
                )
        
        # Validate URL format
        if not self._is_valid_image_url(url):
            return OSINTResult(
                source_name=self.name,
                source_type=self.source_type,
                image_url=url,
                status=ResultStatus.ERROR,
                error="Invalid image URL format",
            )
        
        result = OSINTResult(
            source_name=self.name,
            source_type=self.source_type,
            image_url=url,
        )
        
        if not download:
            return result
        
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                result.status = ResultStatus.ERROR
                result.error = f"Invalid content type: {content_type}"
                return result
            
            # Check size
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_image_size:
                result.status = ResultStatus.ERROR
                result.error = f"Image too large: {content_length} bytes"
                return result
            
            # Download image
            image_data = b""
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk
                if len(image_data) > self.max_image_size:
                    result.status = ResultStatus.ERROR
                    result.error = "Image too large"
                    return result
            
            # Validate image
            if HAS_PIL:
                try:
                    img = Image.open(io.BytesIO(image_data))
                    img.verify()
                    result.metadata["format"] = img.format
                    result.metadata["size"] = img.size if hasattr(img, "size") else None
                except Exception as e:
                    result.status = ResultStatus.ERROR
                    result.error = f"Invalid image: {e}"
                    return result
            
            result.image_data = image_data
            result.status = ResultStatus.SUCCESS
            
            # Cache if enabled
            if self.cache_dir:
                self._cache_image(url, image_data)
        
        except requests.RequestException as e:
            result.status = ResultStatus.ERROR
            result.error = str(e)
            self._record_error(str(e))
        
        return result
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        url_lower = url.lower()
        
        # Check scheme
        if not url_lower.startswith(("http://", "https://")):
            return False
        
        # Check extension
        for fmt in self.allowed_formats:
            if url_lower.endswith(f".{fmt}"):
                return True
        
        # Allow URLs without extension (may be dynamic)
        return True
    
    def _get_cached(self, url: str) -> Optional[bytes]:
        """Get cached image."""
        if not self.cache_dir:
            return None
        
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            return cache_path.read_bytes()
        
        return None
    
    def _cache_image(self, url: str, data: bytes) -> None:
        """Cache image data."""
        if not self.cache_dir:
            return
        
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / cache_key
        cache_path.write_bytes(data)
    
    def validate(self) -> bool:
        """Validate adapter."""
        if not HAS_REQUESTS:
            logger.error("requests not installed")
            return False
        
        return self.enabled
    
    def close(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self) -> "WebImageAdapter":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


class GoogleImagesAdapter(WebImageAdapter):
    """
    Adapter for Google Images search.
    
    Note: This is a placeholder. Actual implementation would require
    Google Custom Search API credentials.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Google Images adapter.
        
        Args:
            api_key: Google API key
            cx: Custom Search Engine ID
        """
        super().__init__(
            name="google_images",
            rate_limit=1.0,
            **kwargs,
        )
        self.source_type = SourceType.IMAGE_SEARCH
        self.api_key = api_key
        self.cx = cx
    
    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Search Google Images.
        
        Requires API key and Custom Search Engine ID.
        """
        if not self.api_key or not self.cx:
            return [self._create_error_result("API key or CX not configured")]
        
        self._wait_rate_limit()
        self._record_request()
        
        try:
            response = self.session.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": self.api_key,
                    "cx": self.cx,
                    "q": query,
                    "searchType": "image",
                    "num": min(max_results, 10),  # API limit
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                result = OSINTResult(
                    source_name=self.name,
                    source_type=self.source_type,
                    image_url=item.get("link"),
                    profile_url=item.get("image", {}).get("contextLink"),
                    name=item.get("title"),
                    metadata={
                        "snippet": item.get("snippet"),
                        "width": item.get("image", {}).get("width"),
                        "height": item.get("image", {}).get("height"),
                    },
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return [self._create_error_result(str(e))]
    
    def validate(self) -> bool:
        """Check if properly configured."""
        return bool(self.api_key and self.cx and self.enabled)
