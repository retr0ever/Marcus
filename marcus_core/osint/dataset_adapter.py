"""
Dataset Adapter
===============

Adapter for loading faces from standard datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterator, Tuple
from pathlib import Path
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    path: Path
    num_identities: int
    num_images: int
    format: str
    metadata: Dict[str, Any]


class DatasetAdapter(OSINTSource):
    """
    Adapter for loading faces from local datasets.
    
    Supports common dataset formats:
    - LFW (Labeled Faces in the Wild)
    - VGGFace2
    - CelebA
    - Custom directory structure (identity/image.jpg)
    
    Example:
        >>> adapter = DatasetAdapter(
        ...     dataset_path="/data/lfw",
        ...     dataset_format="lfw",
        ... )
        >>> 
        >>> # Search for person
        >>> results = adapter.search("George_W_Bush")
        >>> 
        >>> # Iterate through all identities
        >>> for identity, images in adapter.iter_identities():
        ...     print(f"{identity}: {len(images)} images")
    """
    
    SUPPORTED_FORMATS = ["lfw", "vggface2", "celeba", "directory"]
    
    def __init__(
        self,
        dataset_path: str,
        dataset_format: str = "directory",
        name: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        max_images_per_identity: int = 100,
        min_images_per_identity: int = 1,
        **kwargs,
    ):
        """
        Initialize dataset adapter.
        
        Args:
            dataset_path: Path to dataset directory
            dataset_format: Dataset format
            name: Source name
            extensions: Allowed image extensions
            max_images_per_identity: Max images to load per identity
            min_images_per_identity: Min images required per identity
        """
        name = name or f"dataset_{dataset_format}"
        
        super().__init__(
            name=name,
            source_type=SourceType.DATASET,
            rate_limit=100.0,  # Local access, high rate
            **kwargs,
        )
        
        self.dataset_path = Path(dataset_path)
        self.dataset_format = dataset_format
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp"]
        self.max_images_per_identity = max_images_per_identity
        self.min_images_per_identity = min_images_per_identity
        
        # Cache
        self._identity_cache: Optional[Dict[str, List[Path]]] = None
        self._dataset_info: Optional[DatasetInfo] = None
    
    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Search for identities matching query.
        
        Args:
            query: Identity name or partial match
            max_results: Maximum results
        
        Returns:
            List of results
        """
        self._ensure_loaded()
        
        query_lower = query.lower().replace(" ", "_")
        results = []
        
        for identity, images in self._identity_cache.items():
            identity_lower = identity.lower()
            
            # Match by name
            if query_lower in identity_lower or identity_lower in query_lower:
                for img_path in images[:max_results]:
                    result = self._create_result(identity, img_path)
                    results.append(result)
                    
                    if len(results) >= max_results:
                        return results
        
        return results
    
    def search_by_image(
        self,
        image: bytes,
        max_results: int = 10,
    ) -> List[OSINTResult]:
        """
        Search by image. Not implemented for dataset adapter.
        
        This would require embedding extraction and search.
        """
        return []
    
    def get_identity(self, identity_name: str) -> List[OSINTResult]:
        """
        Get all images for an identity.
        
        Args:
            identity_name: Identity name
        
        Returns:
            List of results
        """
        self._ensure_loaded()
        
        if identity_name not in self._identity_cache:
            return []
        
        return [
            self._create_result(identity_name, img_path)
            for img_path in self._identity_cache[identity_name]
        ]
    
    def iter_identities(
        self,
        min_images: Optional[int] = None,
    ) -> Iterator[Tuple[str, List[Path]]]:
        """
        Iterate through identities.
        
        Args:
            min_images: Minimum images per identity
        
        Yields:
            Tuple of (identity_name, image_paths)
        """
        self._ensure_loaded()
        
        min_images = min_images or self.min_images_per_identity
        
        for identity, images in self._identity_cache.items():
            if len(images) >= min_images:
                yield identity, images
    
    def load_image(self, path: Path) -> Optional["np.ndarray"]:
        """
        Load image as numpy array.
        
        Args:
            path: Image path
        
        Returns:
            RGB image array or None
        """
        if not HAS_PIL or not HAS_NUMPY:
            return None
        
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img)
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        self._ensure_loaded()
        return self._dataset_info
    
    def _ensure_loaded(self) -> None:
        """Ensure dataset is loaded."""
        if self._identity_cache is not None:
            return
        
        if self.dataset_format == "lfw":
            self._load_lfw()
        elif self.dataset_format == "vggface2":
            self._load_vggface2()
        elif self.dataset_format == "celeba":
            self._load_celeba()
        else:
            self._load_directory()
        
        # Create dataset info
        self._dataset_info = DatasetInfo(
            name=self.name,
            path=self.dataset_path,
            num_identities=len(self._identity_cache),
            num_images=sum(len(v) for v in self._identity_cache.values()),
            format=self.dataset_format,
            metadata={
                "extensions": self.extensions,
                "max_per_identity": self.max_images_per_identity,
            },
        )
        
        logger.info(
            f"Loaded dataset {self.name}: "
            f"{self._dataset_info.num_identities} identities, "
            f"{self._dataset_info.num_images} images"
        )
    
    def _load_directory(self) -> None:
        """
        Load from directory structure: identity/image.jpg
        """
        self._identity_cache = {}
        
        if not self.dataset_path.exists():
            logger.warning(f"Dataset path not found: {self.dataset_path}")
            return
        
        for identity_dir in self.dataset_path.iterdir():
            if not identity_dir.is_dir():
                continue
            
            identity_name = identity_dir.name
            images = self._find_images(identity_dir)
            
            if len(images) >= self.min_images_per_identity:
                self._identity_cache[identity_name] = images[:self.max_images_per_identity]
    
    def _load_lfw(self) -> None:
        """
        Load LFW dataset.
        
        Structure: lfw/Person_Name/Person_Name_0001.jpg
        """
        self._load_directory()
    
    def _load_vggface2(self) -> None:
        """
        Load VGGFace2 dataset.
        
        Structure: vggface2/n000001/0001.jpg
        """
        self._identity_cache = {}
        
        if not self.dataset_path.exists():
            return
        
        for identity_dir in self.dataset_path.iterdir():
            if not identity_dir.is_dir():
                continue
            
            identity_id = identity_dir.name
            images = self._find_images(identity_dir)
            
            if len(images) >= self.min_images_per_identity:
                self._identity_cache[identity_id] = images[:self.max_images_per_identity]
    
    def _load_celeba(self) -> None:
        """
        Load CelebA dataset.
        
        Structure: celeba/img_align_celeba/*.jpg with identity_CelebA.txt
        """
        self._identity_cache = {}
        
        # CelebA has identity mapping file
        identity_file = self.dataset_path / "identity_CelebA.txt"
        img_dir = self.dataset_path / "img_align_celeba"
        
        if not identity_file.exists() or not img_dir.exists():
            logger.warning("CelebA identity file or image directory not found")
            self._load_directory()  # Fallback
            return
        
        # Parse identity file
        identity_map: Dict[str, List[Path]] = {}
        
        try:
            with open(identity_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_name, identity_id = parts
                        img_path = img_dir / img_name
                        
                        if img_path.exists():
                            if identity_id not in identity_map:
                                identity_map[identity_id] = []
                            
                            if len(identity_map[identity_id]) < self.max_images_per_identity:
                                identity_map[identity_id].append(img_path)
        except Exception as e:
            logger.error(f"Failed to parse CelebA identity file: {e}")
            return
        
        # Filter by min images
        self._identity_cache = {
            k: v for k, v in identity_map.items()
            if len(v) >= self.min_images_per_identity
        }
    
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all images in directory."""
        images = []
        
        for ext in self.extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def _create_result(
        self,
        identity: str,
        image_path: Path,
    ) -> OSINTResult:
        """Create OSINTResult from identity and image path."""
        # Read image bytes
        image_data = None
        try:
            image_data = image_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read image {image_path}: {e}")
        
        return OSINTResult(
            source_name=self.name,
            source_type=self.source_type,
            image_url=str(image_path),
            image_data=image_data,
            name=identity.replace("_", " "),
            metadata={
                "path": str(image_path),
                "identity": identity,
                "dataset_format": self.dataset_format,
            },
            status=ResultStatus.SUCCESS if image_data else ResultStatus.PARTIAL,
        )
    
    def validate(self) -> bool:
        """Validate dataset exists."""
        return self.dataset_path.exists() and self.enabled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = super().get_statistics()
        
        if self._dataset_info:
            stats.update({
                "num_identities": self._dataset_info.num_identities,
                "num_images": self._dataset_info.num_images,
                "dataset_format": self._dataset_info.format,
            })
        
        return stats
