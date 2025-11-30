"""
Embedding Module
================

Face embedding extraction using deep learning models (ArcFace, CosFace, etc.)
"""

from marcus_core.embedding.base import EmbeddingExtractor
from marcus_core.embedding.arcface import ArcFaceExtractor
from marcus_core.embedding.model_zoo import ModelZoo, download_model, get_model_path

__all__ = [
    "EmbeddingExtractor",
    "ArcFaceExtractor",
    "ModelZoo",
    "download_model",
    "get_model_path",
]
