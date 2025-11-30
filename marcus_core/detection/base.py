"""
Base Face Detector Interface
============================

Abstract base class defining the interface for all face detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class FaceDetection:
    """
    Represents a single detected face.
    
    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixels
        confidence: Detection confidence score [0, 1]
        landmarks: Optional 5-point facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
                   Shape: (5, 2) with (x, y) coordinates
        aligned_face: Optional aligned and cropped face image (RGB, uint8)
        embedding: Optional face embedding vector (set after embedding extraction)
        face_id: Optional identifier for tracking
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None  # Shape: (5, 2)
    aligned_face: Optional[np.ndarray] = None  # RGB image
    embedding: Optional[np.ndarray] = None  # Embedding vector
    face_id: Optional[str] = None
    
    @property
    def x1(self) -> int:
        return self.bbox[0]
    
    @property
    def y1(self) -> int:
        return self.bbox[1]
    
    @property
    def x2(self) -> int:
        return self.bbox[2]
    
    @property
    def y2(self) -> int:
        return self.bbox[3]
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding numpy arrays)."""
        return {
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "has_landmarks": self.landmarks is not None,
            "has_aligned_face": self.aligned_face is not None,
            "has_embedding": self.embedding is not None,
            "face_id": self.face_id,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


class FaceDetector(ABC):
    """
    Abstract base class for face detection models.
    
    All face detectors must implement the detect() method.
    Optionally, batch processing can be optimized via detect_batch().
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        min_face_size: int = 20,
        max_faces: int = 100,
        device: str = "auto",
    ):
        """
        Initialize the face detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            min_face_size: Minimum face size in pixels
            max_faces: Maximum number of faces to detect
            device: Compute device (auto, cuda, mps, cpu)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self._device = device
        self._resolved_device: Optional[str] = None
    
    @property
    def device(self) -> str:
        """Get the resolved compute device."""
        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()
        return self._resolved_device
    
    def _resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self._device != "auto":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (RGB format, uint8, shape HxWxC)
            return_landmarks: Whether to return facial landmarks
        
        Returns:
            List of FaceDetection objects for each detected face
        """
        pass
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        return_landmarks: bool = True,
    ) -> List[List[FaceDetection]]:
        """
        Detect faces in multiple images.
        
        Default implementation processes images sequentially.
        Override for batch-optimized processing.
        
        Args:
            images: List of input images (RGB format)
            return_landmarks: Whether to return facial landmarks
        
        Returns:
            List of detection lists, one per input image
        """
        return [self.detect(img, return_landmarks) for img in images]
    
    def filter_detections(
        self,
        detections: List[FaceDetection],
    ) -> List[FaceDetection]:
        """
        Filter detections based on size and limit.
        
        Args:
            detections: List of face detections
        
        Returns:
            Filtered list of detections
        """
        # Filter by minimum size
        filtered = [
            d for d in detections
            if d.width >= self.min_face_size and d.height >= self.min_face_size
        ]
        
        # Sort by confidence and limit
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        return filtered[:self.max_faces]
    
    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the model with a dummy inference.
        
        Call this before timed inference to ensure model is loaded.
        """
        pass
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"confidence={self.confidence_threshold}, "
            f"nms={self.nms_threshold}, "
            f"device={self.device})"
        )
