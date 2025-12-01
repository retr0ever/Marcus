"""
Configuration Management Module
===============================

Centralized configuration for the MARCUS facial analysis system.
Supports YAML files, environment variables, and programmatic configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# =============================================================================
# Enumerations
# =============================================================================

class DeviceType(str, Enum):
    """Supported compute devices."""
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class EmbeddingModelType(str, Enum):
    """Supported embedding model architectures."""
    ARCFACE = "arcface"
    COSFACE = "cosface"
    FACENET = "facenet"
    ADAFACE = "adaface"


class VectorDBType(str, Enum):
    """Supported vector database backends."""
    FAISS = "faiss"
    HNSWLIB = "hnswlib"


class FAISSIndexType(str, Enum):
    """FAISS index types."""
    FLAT = "Flat"
    IVF_FLAT = "IVF_Flat"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"


class MetricType(str, Enum):
    """Distance metrics for similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    INNER_PRODUCT = "inner_product"


class LossType(str, Enum):
    """Metric learning loss functions."""
    ARCFACE = "arcface"
    COSFACE = "cosface"
    TRIPLET = "triplet"
    CONTRASTIVE = "contrastive"


# =============================================================================
# Configuration Classes (Pydantic Models)
# =============================================================================

class DetectionConfig(BaseModel):
    """Face detection configuration."""
    
    model: str = Field(
        default="yolov8n-face",
        description="Detection model name or path"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Custom model path (overrides model name)"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Minimum confidence for face detection"
    )
    nms_threshold: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="Non-maximum suppression threshold"
    )
    min_face_size: int = Field(
        default=20,
        ge=1,
        description="Minimum face size in pixels"
    )
    max_faces: int = Field(
        default=100,
        ge=1,
        description="Maximum number of faces to detect per image"
    )
    device: str = Field(
        default="auto",
        description="Compute device (auto, cuda, mps, cpu)"
    )
    input_size: tuple[int, int] = Field(
        default=(640, 640),
        description="Model input size (width, height)"
    )


class AlignmentConfig(BaseModel):
    """Face alignment and preprocessing configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable face alignment"
    )
    target_size: tuple[int, int] = Field(
        default=(112, 112),
        description="Output face size (width, height)"
    )
    margin: float = Field(
        default=0.2,
        ge=0.0,
        description="Margin around face for cropping"
    )
    use_landmarks: bool = Field(
        default=True,
        description="Use landmark-based alignment"
    )


class EmbeddingConfig(BaseModel):
    """Embedding extraction configuration."""
    
    model: str = Field(
        default="arcface",
        description="Embedding model type (arcface, cosface, facenet)"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Custom model path"
    )
    backbone: str = Field(
        default="r100",
        description="Model backbone (r50, r100, mobilefacenet)"
    )
    embedding_dim: int = Field(
        default=512,
        ge=64,
        description="Embedding vector dimension"
    )
    normalize: bool = Field(
        default=True,
        description="L2 normalize embeddings"
    )
    device: str = Field(
        default="auto",
        description="Compute device"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for inference"
    )
    fp16: bool = Field(
        default=True,
        description="Use FP16 inference for speed"
    )


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    
    type: str = Field(
        default="faiss",
        description="Vector database type (faiss, hnswlib)"
    )
    index_type: str = Field(
        default="Flat",
        description="Index type for FAISS (Flat, IVF_Flat, IVF_PQ, HNSW)"
    )
    metric: str = Field(
        default="cosine",
        description="Distance metric (cosine, euclidean, inner_product)"
    )
    dimension: int = Field(
        default=512,
        ge=64,
        description="Vector dimension"
    )
    nlist: int = Field(
        default=100,
        ge=1,
        description="Number of clusters for IVF index"
    )
    nprobe: int = Field(
        default=10,
        ge=1,
        description="Number of clusters to search"
    )
    ef_construction: int = Field(
        default=200,
        ge=10,
        description="HNSW construction parameter"
    )
    ef_search: int = Field(
        default=50,
        ge=10,
        description="HNSW search parameter"
    )
    persist_path: str = Field(
        default="./data/vector_db",
        description="Path for persistent storage"
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save on changes"
    )


class MatchingConfig(BaseModel):
    """Identity matching configuration."""
    
    similarity_threshold: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="Minimum similarity for a match"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of candidates to retrieve"
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of results after reranking"
    )
    use_clustering: bool = Field(
        default=True,
        description="Use clustering for identity grouping"
    )
    cluster_eps: float = Field(
        default=0.3,
        ge=0.0,
        description="DBSCAN epsilon for clustering"
    )
    cluster_min_samples: int = Field(
        default=2,
        ge=1,
        description="DBSCAN minimum samples"
    )


class TrainingConfig(BaseModel):
    """Training and fine-tuning configuration."""
    
    loss_type: str = Field(
        default="arcface",
        description="Loss function (arcface, cosface, triplet)"
    )
    margin: float = Field(
        default=0.5,
        ge=0.0,
        description="Margin for angular losses"
    )
    scale: float = Field(
        default=64.0,
        ge=1.0,
        description="Scale factor for angular losses"
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate"
    )
    weight_decay: float = Field(
        default=5e-4,
        ge=0.0,
        description="Weight decay"
    )
    epochs: int = Field(
        default=10,
        ge=1,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Training batch size"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="DataLoader workers"
    )
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Checkpoint save directory"
    )
    # Continual learning
    use_experience_replay: bool = Field(
        default=True,
        description="Enable experience replay"
    )
    replay_buffer_size: int = Field(
        default=10000,
        ge=100,
        description="Size of replay buffer"
    )
    ewc_lambda: float = Field(
        default=1000.0,
        ge=0.0,
        description="EWC regularization strength"
    )
    # Hard negative mining
    use_hard_negative_mining: bool = Field(
        default=True,
        description="Enable hard negative mining"
    )
    hard_negative_ratio: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Ratio of hard negatives in batch"
    )


class OSINTConfig(BaseModel):
    """OSINT data ingestion configuration."""
    
    enabled_sources: List[str] = Field(
        default=["linkedin", "public_images", "authorized_datasets"],
        description="Enabled OSINT sources"
    )
    rate_limit_rpm: int = Field(
        default=30,
        ge=1,
        description="Rate limit (requests per minute)"
    )
    download_timeout: int = Field(
        default=30,
        ge=5,
        description="Download timeout in seconds"
    )
    max_image_size_mb: float = Field(
        default=10.0,
        ge=0.1,
        description="Maximum image size in MB"
    )
    allowed_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp"],
        description="Allowed image formats"
    )
    cache_dir: str = Field(
        default="./data/osint_cache",
        description="Cache directory for downloaded data"
    )
    require_authorization: bool = Field(
        default=True,
        description="Require explicit authorization for ingestion"
    )


class ComplianceConfig(BaseModel):
    """UK GDPR compliance configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable compliance logging"
    )
    log_all_access: bool = Field(
        default=True,
        description="Log all data access"
    )
    audit_log_path: str = Field(
        default="./data/audit_logs",
        description="Audit log directory"
    )
    log_retention_days: int = Field(
        default=365,
        ge=1,
        description="Log retention period in days"
    )
    require_consent: bool = Field(
        default=True,
        description="Require consent for processing"
    )
    consent_expiry_days: int = Field(
        default=365,
        ge=1,
        description="Consent validity period"
    )
    data_retention_days: int = Field(
        default=730,
        ge=1,
        description="Data retention period"
    )
    right_to_erasure: bool = Field(
        default=True,
        description="Enable right to erasure"
    )
    encryption_enabled: bool = Field(
        default=False,
        description="Enable data encryption"
    )
    encryption_key_env: str = Field(
        default="MARCUS_ENCRYPTION_KEY",
        description="Environment variable for encryption key"
    )


class APIConfig(BaseModel):
    """API server configuration."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        ge=1, le=65535,
        description="Server port"
    )
    workers: int = Field(
        default=4,
        ge=1,
        description="Number of workers"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    api_key_enabled: bool = Field(
        default=True,
        description="Enable API key authentication"
    )
    api_key_env: str = Field(
        default="MARCUS_API_KEY",
        description="Environment variable for API key"
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_rpm: int = Field(
        default=100,
        ge=1,
        description="Rate limit per minute"
    )
    max_upload_size_mb: float = Field(
        default=50.0,
        ge=1.0,
        description="Maximum upload size in MB"
    )


# =============================================================================
# Master System Configuration
# =============================================================================

class SystemConfig(BaseModel):
    """Master system configuration combining all modules."""
    
    # Project metadata
    project_name: str = Field(default="MARCUS")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    
    # Device
    device: str = Field(
        default="auto",
        description="Global compute device (auto, cuda, mps, cpu)"
    )
    
    # Paths
    data_dir: str = Field(default="./data")
    models_dir: str = Field(default="./models")
    logs_dir: str = Field(default="./logs")
    
    # Sub-configurations
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    osint: OSINTConfig = Field(default_factory=OSINTConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to sync device settings."""
        if self.device != "auto":
            # Propagate device setting to sub-configs
            if self.detection.device == "auto":
                self.detection.device = self.device
            if self.embedding.device == "auto":
                self.embedding.device = self.device
    
    def get_resolved_device(self) -> str:
        """Get the actual device to use (resolving 'auto')."""
        if self.device == "auto":
            return detect_device()
        return self.device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "SystemConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load configuration from environment variables."""
        config_path = os.environ.get("MARCUS_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            return cls.from_yaml(config_path)
        return cls()


# =============================================================================
# Utility Functions
# =============================================================================

def detect_device() -> str:
    """Detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_config(path: Optional[str | Path] = None) -> SystemConfig:
    """
    Load system configuration from file or environment.
    
    Args:
        path: Path to YAML configuration file.
              If None, checks MARCUS_CONFIG_PATH env var, then uses defaults.
    
    Returns:
        SystemConfig instance
    
    Example:
        >>> config = load_config("configs/default.yaml")
        >>> config = load_config()  # Uses env var or defaults
    """
    if path is not None:
        return SystemConfig.from_yaml(path)
    
    env_path = os.environ.get("MARCUS_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return SystemConfig.from_yaml(env_path)
    
    return SystemConfig()


def create_default_config(path: str | Path = "configs/default.yaml") -> SystemConfig:
    """Create and save a default configuration file."""
    config = SystemConfig()
    config.save_yaml(path)
    return config


# =============================================================================
# Default Configuration Instance
# =============================================================================

# Lazy-loaded default config
_default_config: Optional[SystemConfig] = None


def get_default_config() -> SystemConfig:
    """Get the default configuration (lazy-loaded singleton)."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config
