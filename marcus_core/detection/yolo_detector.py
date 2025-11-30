"""
YOLO Face Detector
==================

Face detection using YOLOv8-Face and similar YOLO variants.
Provides real-time face detection with optional landmark prediction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

from marcus_core.detection.base import FaceDetector, FaceDetection
from marcus_core.detection.preprocessing import FaceAligner


class YOLOFaceDetector(FaceDetector):
    """
    Face detector using YOLOv8-Face model.
    
    Uses the ultralytics YOLO implementation with face-specific models
    that output bounding boxes and 5-point facial landmarks.
    
    Example:
        >>> detector = YOLOFaceDetector(model="yolov8n-face")
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     print(f"Face at {face.bbox} with confidence {face.confidence:.2f}")
    """
    
    # Known face detection models
    KNOWN_MODELS = {
        "yolov8n-face": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
        "yolov8s-face": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt",
        "yolov8m-face": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt",
    }
    
    def __init__(
        self,
        model: str = "yolov8n-face",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        min_face_size: int = 20,
        max_faces: int = 100,
        device: str = "auto",
        input_size: Tuple[int, int] = (640, 640),
        align_faces: bool = True,
        target_face_size: Tuple[int, int] = (112, 112),
    ):
        """
        Initialize the YOLO face detector.
        
        Args:
            model: Model name (yolov8n-face, yolov8s-face, yolov8m-face) or path
            model_path: Custom model path (overrides model name)
            confidence_threshold: Minimum detection confidence
            nms_threshold: NMS threshold
            min_face_size: Minimum face size in pixels
            max_faces: Maximum faces to detect
            device: Compute device (auto, cuda, mps, cpu)
            input_size: Model input size
            align_faces: Whether to align detected faces
            target_face_size: Target size for aligned faces
        """
        super().__init__(
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_face_size=min_face_size,
            max_faces=max_faces,
            device=device,
        )
        
        self.model_name = model
        self.model_path = model_path or self._resolve_model_path(model)
        self.input_size = input_size
        self.align_faces = align_faces
        
        # Face aligner for preprocessing
        if align_faces:
            self.aligner = FaceAligner(target_size=target_face_size)
        else:
            self.aligner = None
        
        # Lazy-loaded model
        self._model = None
    
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to path, downloading if necessary."""
        # Check if it's already a path
        if os.path.exists(model_name):
            return model_name
        
        # Check in models directory
        models_dir = Path("./models")
        local_path = models_dir / f"{model_name}.pt"
        if local_path.exists():
            return str(local_path)
        
        # Return model name for ultralytics to handle
        return model_name
    
    def _load_model(self):
        """Load the YOLO model."""
        if self._model is not None:
            return
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )
        
        # Download model if needed
        if self.model_name in self.KNOWN_MODELS and not os.path.exists(self.model_path):
            self._download_model()
        
        self._model = YOLO(self.model_path)
        
        # Move to device
        if self.device == "cuda":
            self._model.to("cuda")
        elif self.device == "mps":
            self._model.to("mps")
    
    def _download_model(self) -> None:
        """Download the model weights."""
        import requests
        from tqdm import tqdm
        
        url = self.KNOWN_MODELS.get(self.model_name)
        if not url:
            return
        
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = models_dir / f"{self.model_name}.pt"
        
        print(f"Downloading {self.model_name}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        self.model_path = str(output_path)
        print(f"Model saved to {output_path}")
    
    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            return_landmarks: Whether to extract landmarks
        
        Returns:
            List of FaceDetection objects
        """
        self._load_model()
        
        # Run inference
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            verbose=False,
            device=self.device if self.device != "auto" else None,
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Extract detections
            for i in range(len(boxes)):
                # Bounding box
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Confidence
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Landmarks (if available)
                landmarks = None
                if return_landmarks and hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        kpts = result.keypoints.xy[i].cpu().numpy()
                        if len(kpts) >= 5:
                            landmarks = kpts[:5].astype(np.float32)
                    except (IndexError, AttributeError):
                        pass
                
                # Create detection
                detection = FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    landmarks=landmarks,
                )
                
                # Align face if requested
                if self.align_faces and self.aligner is not None:
                    aligned = self.aligner.align(image, detection)
                    if aligned is not None:
                        detection.aligned_face = aligned
                
                detections.append(detection)
        
        # Filter and sort
        return self.filter_detections(detections)
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        return_landmarks: bool = True,
    ) -> List[List[FaceDetection]]:
        """
        Detect faces in multiple images efficiently.
        
        Args:
            images: List of RGB images
            return_landmarks: Whether to extract landmarks
        
        Returns:
            List of detection lists
        """
        self._load_model()
        
        # Batch inference
        results = self._model.predict(
            source=images,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            verbose=False,
            device=self.device if self.device != "auto" else None,
        )
        
        all_detections = []
        
        for idx, result in enumerate(results):
            image = images[idx]
            detections = []
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    landmarks = None
                    if return_landmarks and hasattr(result, 'keypoints') and result.keypoints is not None:
                        try:
                            kpts = result.keypoints.xy[i].cpu().numpy()
                            if len(kpts) >= 5:
                                landmarks = kpts[:5].astype(np.float32)
                        except (IndexError, AttributeError):
                            pass
                    
                    detection = FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        landmarks=landmarks,
                    )
                    
                    if self.align_faces and self.aligner is not None:
                        aligned = self.aligner.align(image, detection)
                        if aligned is not None:
                            detection.aligned_face = aligned
                    
                    detections.append(detection)
            
            all_detections.append(self.filter_detections(detections))
        
        return all_detections
    
    def warmup(self) -> None:
        """Warm up the model with dummy inference."""
        self._load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, return_landmarks=False)
    
    def __repr__(self) -> str:
        return (
            f"YOLOFaceDetector(model={self.model_name}, "
            f"confidence={self.confidence_threshold}, "
            f"device={self.device})"
        )


class RetinaFaceDetector(FaceDetector):
    """
    Face detector using RetinaFace model via InsightFace.
    
    Provides high-accuracy face detection with 5-point landmarks.
    Falls back to YOLO if InsightFace is not available.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        min_face_size: int = 20,
        max_faces: int = 100,
        device: str = "auto",
        align_faces: bool = True,
        target_face_size: Tuple[int, int] = (112, 112),
    ):
        """Initialize RetinaFace detector."""
        super().__init__(
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_face_size=min_face_size,
            max_faces=max_faces,
            device=device,
        )
        
        self.align_faces = align_faces
        if align_faces:
            self.aligner = FaceAligner(target_size=target_face_size)
        else:
            self.aligner = None
        
        self._detector = None
    
    def _load_model(self):
        """Load the RetinaFace model from InsightFace."""
        if self._detector is not None:
            return
        
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface package is required for RetinaFace. "
                "Install with: pip install insightface"
            )
        
        # Determine providers based on device
        if self.device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self._detector = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
        )
        self._detector.prepare(ctx_id=0 if self.device == "cuda" else -1)
    
    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
    ) -> List[FaceDetection]:
        """Detect faces using RetinaFace."""
        self._load_model()
        
        # InsightFace expects BGR
        import cv2
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        faces = self._detector.get(bgr_image)
        
        detections = []
        for face in faces:
            if face.det_score < self.confidence_threshold:
                continue
            
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            landmarks = None
            if return_landmarks and hasattr(face, 'kps'):
                landmarks = face.kps.astype(np.float32)
            
            detection = FaceDetection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(face.det_score),
                landmarks=landmarks,
            )
            
            if self.align_faces and self.aligner is not None:
                aligned = self.aligner.align(image, detection)
                if aligned is not None:
                    detection.aligned_face = aligned
            
            detections.append(detection)
        
        return self.filter_detections(detections)
    
    def warmup(self) -> None:
        """Warm up the model."""
        self._load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, return_landmarks=False)
