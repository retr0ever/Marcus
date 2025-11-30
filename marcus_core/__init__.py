"""
MARCUS Core - Facial Analysis System
=====================================

A modular, OSINT-enabled facial analysis library with:
- YOLO-based real-time face detection
- Deep facial embeddings (ArcFace, CosFace)
- Vector database (FAISS/HNSWlib) for similarity search
- Trainable pipeline with metric learning
- UK GDPR compliance logging

Example:
    >>> from marcus_core import FacialPipeline
    >>> pipeline = FacialPipeline.from_config("configs/default.yaml")
    >>> results = pipeline.search(image)
"""

__version__ = "1.0.0"
__author__ = "MARCUS Project"

from marcus_core.config import (
    SystemConfig,
    DetectionConfig,
    EmbeddingConfig,
    VectorDBConfig,
    MatchingConfig,
    TrainingConfig,
    OSINTConfig,
    ComplianceConfig,
    load_config,
)

from marcus_core.detection import FaceDetector, FaceDetection, YOLOFaceDetector
from marcus_core.embedding import EmbeddingExtractor, ArcFaceExtractor
from marcus_core.vectordb import VectorStore, FAISSStore
from marcus_core.matching import IdentityMatcher, Identity, MatchResult
from marcus_core.compliance import AuditLogger, ConsentManager
from marcus_core.pipeline import FacialPipeline

__all__ = [
    # Version
    "__version__",
    # Config
    "SystemConfig",
    "DetectionConfig", 
    "EmbeddingConfig",
    "VectorDBConfig",
    "MatchingConfig",
    "TrainingConfig",
    "OSINTConfig",
    "ComplianceConfig",
    "load_config",
    # Detection
    "FaceDetector",
    "FaceDetection",
    "YOLOFaceDetector",
    # Embedding
    "EmbeddingExtractor",
    "ArcFaceExtractor",
    # Vector DB
    "VectorStore",
    "FAISSStore",
    # Matching
    "IdentityMatcher",
    "Identity",
    "MatchResult",
    # Compliance
    "AuditLogger",
    "ConsentManager",
    # Pipeline
    "FacialPipeline",
]
