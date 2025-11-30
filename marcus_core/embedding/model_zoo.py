"""
Model Zoo - Pre-trained Model Registry
=======================================

Registry of pre-trained face recognition models with download support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import hashlib


@dataclass
class ModelInfo:
    """Information about a pre-trained model."""
    name: str
    url: str
    filename: str
    embedding_dim: int
    input_size: tuple
    model_type: str  # onnx, pytorch, insightface
    sha256: Optional[str] = None
    description: str = ""


# Registry of available models
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # ArcFace models from InsightFace
    "arcface_r100": ModelInfo(
        name="ArcFace ResNet100",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        filename="w600k_r50.onnx",
        embedding_dim=512,
        input_size=(112, 112),
        model_type="insightface",
        description="High-accuracy ArcFace model with ResNet100 backbone",
    ),
    "arcface_r50": ModelInfo(
        name="ArcFace ResNet50",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        filename="w600k_r50.onnx",
        embedding_dim=512,
        input_size=(112, 112),
        model_type="insightface",
        description="Balanced ArcFace model with ResNet50 backbone",
    ),
    "arcface_mobilefacenet": ModelInfo(
        name="ArcFace MobileFaceNet",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        filename="w600k_mbf.onnx",
        embedding_dim=512,
        input_size=(112, 112),
        model_type="insightface",
        description="Fast ArcFace model with MobileFaceNet backbone",
    ),
}


class ModelZoo:
    """
    Model registry and download manager.
    
    Example:
        >>> zoo = ModelZoo(models_dir="./models")
        >>> model_path = zoo.get_model("arcface_r100")
        >>> print(zoo.list_models())
    """
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize the model zoo.
        
        Args:
            models_dir: Directory for storing downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all available models."""
        return MODEL_REGISTRY.copy()
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return MODEL_REGISTRY.get(name)
    
    def get_model_path(self, name: str, download: bool = True) -> Optional[Path]:
        """
        Get path to a model, downloading if necessary.
        
        Args:
            name: Model name from registry
            download: Whether to download if not present
        
        Returns:
            Path to model file or None if not available
        """
        info = MODEL_REGISTRY.get(name)
        if info is None:
            return None
        
        model_path = self.models_dir / name / info.filename
        
        if model_path.exists():
            return model_path
        
        if download:
            return self.download_model(name)
        
        return None
    
    def download_model(self, name: str) -> Optional[Path]:
        """
        Download a model from the registry.
        
        Args:
            name: Model name
        
        Returns:
            Path to downloaded model or None if failed
        """
        info = MODEL_REGISTRY.get(name)
        if info is None:
            print(f"Unknown model: {name}")
            return None
        
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / info.filename
        
        if model_path.exists():
            return model_path
        
        print(f"Downloading {info.name}...")
        
        try:
            if info.model_type == "insightface":
                return self._download_insightface_model(name, info)
            else:
                return self._download_direct(info.url, model_path)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            return None
    
    def _download_insightface_model(self, name: str, info: ModelInfo) -> Optional[Path]:
        """Download model using InsightFace library."""
        try:
            from insightface.app import FaceAnalysis
            
            # InsightFace downloads models automatically
            app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(160, 160))
            
            # Find the downloaded model
            import insightface
            models_path = Path(insightface.__file__).parent / "models"
            
            # Return path to recognition model
            for model_name in ['buffalo_l', 'buffalo_s', 'buffalo_sc']:
                rec_path = models_path / model_name / 'w600k_r50.onnx'
                if rec_path.exists():
                    return rec_path
            
            print("Could not locate InsightFace model after download")
            return None
            
        except ImportError:
            print("InsightFace not installed. Install with: pip install insightface")
            return None
    
    def _download_direct(self, url: str, output_path: Path) -> Optional[Path]:
        """Download a file directly from URL."""
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return output_path
    
    def verify_model(self, name: str) -> bool:
        """Verify model integrity using SHA256 checksum."""
        info = MODEL_REGISTRY.get(name)
        if info is None or info.sha256 is None:
            return True  # No checksum to verify
        
        model_path = self.models_dir / name / info.filename
        if not model_path.exists():
            return False
        
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest() == info.sha256


# Convenience functions
_default_zoo: Optional[ModelZoo] = None


def get_model_zoo(models_dir: str = "./models") -> ModelZoo:
    """Get the default model zoo instance."""
    global _default_zoo
    if _default_zoo is None:
        _default_zoo = ModelZoo(models_dir)
    return _default_zoo


def download_model(name: str, models_dir: str = "./models") -> Optional[Path]:
    """Download a model by name."""
    zoo = get_model_zoo(models_dir)
    return zoo.download_model(name)


def get_model_path(name: str, models_dir: str = "./models") -> Optional[Path]:
    """Get path to a model, downloading if needed."""
    zoo = get_model_zoo(models_dir)
    return zoo.get_model_path(name)
