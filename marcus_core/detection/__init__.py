"""
Detection Module
================

Face detection using YOLO and other models with face alignment preprocessing.
"""

from marcus_core.detection.base import FaceDetector, FaceDetection
from marcus_core.detection.yolo_detector import YOLOFaceDetector
from marcus_core.detection.preprocessing import FaceAligner, align_face

__all__ = [
    "FaceDetector",
    "FaceDetection",
    "YOLOFaceDetector",
    "FaceAligner",
    "align_face",
]
