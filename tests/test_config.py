"""
Test Configuration Module
=========================
"""

import pytest
import tempfile
from pathlib import Path

from marcus_core.config import (
    SystemConfig,
    DetectionConfig,
    EmbeddingConfig,
    VectorDBConfig,
    MatchingConfig,
    load_config,
)


class TestDetectionConfig:
    """Tests for DetectionConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DetectionConfig()
        
        assert config.model == "yolov8n-face"
        assert config.confidence_threshold == 0.5
        assert config.nms_threshold == 0.4
        assert config.device == "auto"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DetectionConfig(
            model="yolov8s-face",
            confidence_threshold=0.7,
            max_faces=5,
        )
        
        assert config.model == "yolov8s-face"
        assert config.confidence_threshold == 0.7
        assert config.max_faces == 5
    
    def test_validation(self):
        """Test validation constraints."""
        # Confidence threshold must be 0-1
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=1.5)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        assert config.backbone == "r100"
        assert config.embedding_dim == 512
        assert config.normalize is True
    
    def test_model_options(self):
        """Test different backbone options."""
        for backbone in ["r100", "r50", "mobilefacenet"]:
            config = EmbeddingConfig(backbone=backbone)
            assert config.backbone == backbone


class TestVectorDBConfig:
    """Tests for VectorDBConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VectorDBConfig()
        
        assert config.dimension == 512
        assert config.metric == "cosine"
        assert config.index_type == "Flat"
    
    def test_index_types(self):
        """Test different index types."""
        for index_type in ["Flat", "IVF_Flat", "IVF_PQ", "HNSW"]:
            config = VectorDBConfig(index_type=index_type)
            assert config.index_type == index_type


class TestMatchingConfig:
    """Tests for MatchingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MatchingConfig()
        
        assert config.similarity_threshold == 0.6
        assert config.top_k == 10
        assert config.algorithm == "cosine"


class TestSystemConfig:
    """Tests for SystemConfig."""
    
    def test_default_values(self):
        """Test default system configuration."""
        config = SystemConfig()
        
        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.detection is not None
        assert config.embedding is not None
        assert config.vectordb is not None
    
    def test_nested_configs(self):
        """Test that nested configs are properly initialized."""
        config = SystemConfig(
            detection=DetectionConfig(model="yolov8s-face"),
            embedding=EmbeddingConfig(backbone="r50"),
        )
        
        assert config.detection.model == "yolov8s-face"
        assert config.embedding.backbone == "r50"
    
    def test_yaml_save_load(self):
        """Test YAML serialization."""
        config = SystemConfig(
            data_dir="custom_data",
            detection=DetectionConfig(confidence_threshold=0.7),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            
            # Save
            config.save(str(path))
            assert path.exists()
            
            # Load
            loaded = load_config(str(path))
            assert loaded.data_dir == "custom_data"
            assert loaded.detection.confidence_threshold == 0.7


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_missing_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    def test_load_valid_yaml(self):
        """Test loading valid YAML configuration."""
        yaml_content = """
data_dir: test_data
models_dir: test_models
detection:
  model: yolov8s-face
  confidence_threshold: 0.8
embedding:
  backbone: r50
matching:
  similarity_threshold: 0.7
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            path.write_text(yaml_content)
            
            config = load_config(str(path))
            
            assert config.data_dir == "test_data"
            assert config.detection.model == "yolov8s-face"
            assert config.detection.confidence_threshold == 0.8
            assert config.embedding.backbone == "r50"
            assert config.matching.similarity_threshold == 0.7
