"""
Source Registry
===============

Registry for managing multiple OSINT sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Type
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from marcus_core.osint.base import OSINTSource, OSINTResult, SourceType


logger = logging.getLogger(__name__)


@dataclass
class AggregatedResults:
    """Results aggregated from multiple sources."""
    results: List[OSINTResult]
    total_sources: int
    successful_sources: int
    failed_sources: List[str]
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful sources."""
        if self.total_sources == 0:
            return 0.0
        return self.successful_sources / self.total_sources


class SourceRegistry:
    """
    Registry for managing multiple OSINT sources.
    
    Supports parallel querying and result aggregation.
    
    Example:
        >>> registry = SourceRegistry()
        >>> registry.register(WebImageAdapter())
        >>> registry.register(DatasetAdapter(path="lfw"))
        >>> 
        >>> # Search all sources
        >>> results = registry.search_all("John Doe", max_results=20)
        >>> for result in results.results:
        ...     print(f"{result.source_name}: {result.name}")
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        timeout: float = 30.0,
    ):
        """
        Initialize registry.
        
        Args:
            max_workers: Maximum parallel workers
            timeout: Query timeout in seconds
        """
        self._sources: Dict[str, OSINTSource] = {}
        self._source_factories: Dict[str, Type[OSINTSource]] = {}
        self.max_workers = max_workers
        self.timeout = timeout
    
    def register(self, source: OSINTSource) -> None:
        """
        Register an OSINT source.
        
        Args:
            source: Source instance to register
        """
        if source.name in self._sources:
            logger.warning(f"Overwriting existing source: {source.name}")
        
        self._sources[source.name] = source
        logger.info(f"Registered source: {source.name} ({source.source_type.value})")
    
    def register_factory(
        self,
        name: str,
        factory: Type[OSINTSource],
    ) -> None:
        """
        Register a source factory for lazy instantiation.
        
        Args:
            name: Factory name
            factory: Source class
        """
        self._source_factories[name] = factory
    
    def unregister(self, name: str) -> bool:
        """
        Remove a source from the registry.
        
        Args:
            name: Source name
        
        Returns:
            True if removed
        """
        if name in self._sources:
            del self._sources[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[OSINTSource]:
        """Get source by name."""
        return self._sources.get(name)
    
    def list_sources(
        self,
        source_type: Optional[SourceType] = None,
        enabled_only: bool = True,
    ) -> List[OSINTSource]:
        """
        List registered sources.
        
        Args:
            source_type: Filter by type
            enabled_only: Only enabled sources
        
        Returns:
            List of sources
        """
        sources = list(self._sources.values())
        
        if source_type is not None:
            sources = [s for s in sources if s.source_type == source_type]
        
        if enabled_only:
            sources = [s for s in sources if s.enabled]
        
        return sources
    
    def search_all(
        self,
        query: str,
        max_results: int = 20,
        source_type: Optional[SourceType] = None,
        sources: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> AggregatedResults:
        """
        Search across all registered sources.
        
        Args:
            query: Search query
            max_results: Max results per source
            source_type: Filter by type
            sources: Specific source names to query
            parallel: Execute in parallel
        
        Returns:
            Aggregated results from all sources
        """
        target_sources = self._get_target_sources(source_type, sources)
        
        if parallel:
            return self._search_parallel(target_sources, query, max_results)
        else:
            return self._search_sequential(target_sources, query, max_results)
    
    def search_by_image_all(
        self,
        image: bytes,
        max_results: int = 20,
        sources: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> AggregatedResults:
        """
        Reverse image search across all sources.
        
        Args:
            image: Image bytes
            max_results: Max results per source
            sources: Specific sources to query
            parallel: Execute in parallel
        
        Returns:
            Aggregated results
        """
        target_sources = self._get_target_sources(
            source_type=SourceType.IMAGE_SEARCH,
            sources=sources,
        )
        
        all_results = []
        failed = []
        successful = 0
        
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._search_image_source,
                        source,
                        image,
                        max_results,
                    ): source
                    for source in target_sources
                }
                
                for future in as_completed(futures, timeout=self.timeout):
                    source = futures[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        successful += 1
                    except Exception as e:
                        logger.error(f"Source {source.name} failed: {e}")
                        failed.append(source.name)
        else:
            for source in target_sources:
                try:
                    results = self._search_image_source(source, image, max_results)
                    all_results.extend(results)
                    successful += 1
                except Exception as e:
                    logger.error(f"Source {source.name} failed: {e}")
                    failed.append(source.name)
        
        return AggregatedResults(
            results=all_results,
            total_sources=len(target_sources),
            successful_sources=successful,
            failed_sources=failed,
        )
    
    def _get_target_sources(
        self,
        source_type: Optional[SourceType],
        sources: Optional[List[str]],
    ) -> List[OSINTSource]:
        """Get sources to query."""
        if sources is not None:
            return [
                self._sources[name]
                for name in sources
                if name in self._sources and self._sources[name].enabled
            ]
        
        return self.list_sources(source_type=source_type, enabled_only=True)
    
    def _search_parallel(
        self,
        sources: List[OSINTSource],
        query: str,
        max_results: int,
    ) -> AggregatedResults:
        """Execute parallel search."""
        all_results = []
        failed = []
        successful = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._search_source,
                    source,
                    query,
                    max_results,
                ): source
                for source in sources
            }
            
            for future in as_completed(futures, timeout=self.timeout):
                source = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    successful += 1
                except Exception as e:
                    logger.error(f"Source {source.name} failed: {e}")
                    failed.append(source.name)
        
        return AggregatedResults(
            results=all_results,
            total_sources=len(sources),
            successful_sources=successful,
            failed_sources=failed,
        )
    
    def _search_sequential(
        self,
        sources: List[OSINTSource],
        query: str,
        max_results: int,
    ) -> AggregatedResults:
        """Execute sequential search."""
        all_results = []
        failed = []
        successful = 0
        
        for source in sources:
            try:
                results = self._search_source(source, query, max_results)
                all_results.extend(results)
                successful += 1
            except Exception as e:
                logger.error(f"Source {source.name} failed: {e}")
                failed.append(source.name)
        
        return AggregatedResults(
            results=all_results,
            total_sources=len(sources),
            successful_sources=successful,
            failed_sources=failed,
        )
    
    def _search_source(
        self,
        source: OSINTSource,
        query: str,
        max_results: int,
    ) -> List[OSINTResult]:
        """Search a single source."""
        return source.search(query, max_results=max_results)
    
    def _search_image_source(
        self,
        source: OSINTSource,
        image: bytes,
        max_results: int,
    ) -> List[OSINTResult]:
        """Search a single source by image."""
        return source.search_by_image(image, max_results=max_results)
    
    def validate_all(self) -> Dict[str, bool]:
        """Validate all sources."""
        return {
            name: source.validate()
            for name, source in self._sources.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all sources."""
        return {
            "total_sources": len(self._sources),
            "enabled_sources": len(self.list_sources(enabled_only=True)),
            "sources": {
                name: source.get_statistics()
                for name, source in self._sources.items()
            },
        }
    
    def __len__(self) -> int:
        return len(self._sources)
    
    def __repr__(self) -> str:
        return f"SourceRegistry(sources={len(self._sources)})"
