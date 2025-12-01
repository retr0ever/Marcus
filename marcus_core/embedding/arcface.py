"""
ArcFace Embedding Extractor
============================

Face embedding extraction using ArcFace models via InsightFace or ONNX.
"""

from __future__ import annotations

from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import cv2

from marcus_core.embedding.base import EmbeddingExtractor


class ArcFaceExtractor(EmbeddingExtractor):
    """
    ArcFace face embedding extractor.
    
    Uses InsightFace library for high-accuracy face embeddings.
    Supports multiple backbones (ResNet100, ResNet50, MobileFaceNet).
    
    Example:
        >>> extractor = ArcFaceExtractor(backbone="r100")
        >>> embedding = extractor.extract(aligned_face)
        >>> print(embedding.shape)  # (512,)
    """
    
    BACKBONES = {
        "r100": "buffalo_l",
        "r50": "buffalo_s",
        "mobilefacenet": "buffalo_sc",
        "mbf": "buffalo_sc",
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = "r100",
        embedding_dim: int = 512,
        normalize: bool = True,
        device: str = "auto",
        fp16: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize ArcFace extractor.
        
        Args:
            model_path: Custom ONNX model path (optional)
            backbone: Model backbone (r100, r50, mobilefacenet)
            embedding_dim: Expected embedding dimension
            normalize: L2 normalize embeddings
            device: Compute device (auto, cuda, mps, cpu)
            fp16: Use FP16 inference
            batch_size: Batch size for batch extraction
        """
        super().__init__(
            model_path=model_path,
            embedding_dim=embedding_dim,
            normalize=normalize,
            device=device,
            fp16=fp16,
        )
        
        self.backbone = backbone
        self.batch_size = batch_size
        self.input_size = (112, 112)
        
        # Lazy-loaded components
        self._model = None
        self._session = None
        self._use_insightface = model_path is None
    
    def _load_model(self):
        """Load the embedding model."""
        if self._model is not None or self._session is not None:
            return
        
        if self._use_insightface:
            self._load_insightface_model()
        else:
            self._load_onnx_model()
    
    def _load_insightface_model(self):
        """Load model using InsightFace library."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface package is required. "
                "Install with: pip install insightface onnxruntime"
            )
        
        # Determine model name from backbone
        model_name = self.BACKBONES.get(self.backbone, "buffalo_l")
        
        # Determine providers
        if self.device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == "mps":
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Initialize FaceAnalysis (includes detection + recognition)
        self._model = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=['detection', 'recognition'],  # Need detection for face analysis
        )
        self._model.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
        
        # Extract the recognition model
        if hasattr(self._model, 'models') and 'recognition' in self._model.models:
            self._rec_model = self._model.models['recognition']
        else:
            # Fallback: use the full model
            self._rec_model = None
    
    def _load_onnx_model(self):
        """Load ONNX model directly."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX models. "
                "Install with: pip install onnxruntime or onnxruntime-gpu"
            )
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Providers
        if self.device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
    
    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the model.
        
        Args:
            face_image: RGB face image (H, W, C), uint8
        
        Returns:
            Preprocessed tensor (1, C, H, W), float32
        """
        # Resize if needed
        if face_image.shape[:2] != self.input_size:
            face_image = cv2.resize(face_image, self.input_size)
        
        # Convert to float and normalize
        img = face_image.astype(np.float32)
        
        # Standard normalization for InsightFace models
        img = (img - 127.5) / 127.5
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single face image.
        
        Args:
            face_image: Aligned face image (RGB, uint8)
        
        Returns:
            Embedding vector (512,)
        """
        self._load_model()
        
        if self._use_insightface and self._model is not None:
            return self._extract_insightface(face_image)
        else:
            return self._extract_onnx(face_image)
    
    def _extract_insightface(self, face_image: np.ndarray) -> np.ndarray:
        """Extract using InsightFace model."""
        # Convert RGB to BGR for InsightFace
        bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if bgr.shape[:2] != self.input_size:
            bgr = cv2.resize(bgr, self.input_size)
        
        # Get embedding using recognition model
        if self._rec_model is not None:
            embedding = self._rec_model.get_feat(bgr)
        else:
            # Fallback: run detection and get embedding
            faces = self._model.get(bgr)
            if len(faces) > 0:
                embedding = faces[0].embedding
            else:
                # No face detected, compute from image anyway
                preprocessed = self._preprocess(face_image)
                embedding = self._run_inference(preprocessed)
        
        embedding = embedding.flatten()
        return self._normalize_embedding(embedding)
    
    def _extract_onnx(self, face_image: np.ndarray) -> np.ndarray:
        """Extract using ONNX runtime."""
        preprocessed = self._preprocess(face_image)
        embedding = self._run_inference(preprocessed)
        return self._normalize_embedding(embedding.flatten())
    
    def _run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        if self._session is None:
            raise RuntimeError("ONNX session not initialized")
        
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: input_tensor.astype(np.float32)}
        )
        return outputs[0]
    
    def extract_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_images: List of aligned face images
        
        Returns:
            Embeddings array (N, 512)
        """
        self._load_model()
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(face_images), self.batch_size):
            batch = face_images[i:i + self.batch_size]
            
            if self._use_insightface:
                # InsightFace processes one at a time
                batch_embeddings = [self._extract_insightface(img) for img in batch]
            else:
                # ONNX can do batch inference
                preprocessed = np.vstack([self._preprocess(img) for img in batch])
                batch_embeddings = self._run_inference(preprocessed)
                batch_embeddings = self._normalize_embedding(batch_embeddings)
            
            embeddings.extend(batch_embeddings if isinstance(batch_embeddings, list) 
                             else [batch_embeddings[j] for j in range(len(batch_embeddings))])
        
        return np.array(embeddings)
    
    def warmup(self) -> None:
        """Warm up the model with dummy inference."""
        self._load_model()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        self.extract(dummy)
    
    def __repr__(self) -> str:
        return (
            f"ArcFaceExtractor(backbone={self.backbone}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )


class CosFaceExtractor(ArcFaceExtractor):
    """
    CosFace face embedding extractor.
    
    Uses the same architecture as ArcFace but with CosFace loss training.
    Currently wraps ArcFace models as they perform similarly.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return (
            f"CosFaceExtractor(backbone={self.backbone}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )
