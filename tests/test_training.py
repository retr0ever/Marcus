"""
Test Threshold Optimizer
========================
"""

import pytest
import numpy as np

from marcus_core.training.threshold_optimizer import ThresholdOptimizer, ThresholdMetrics


class TestThresholdOptimizer:
    """Tests for ThresholdOptimizer."""
    
    def test_add_samples(self):
        """Test adding samples."""
        optimizer = ThresholdOptimizer()
        
        optimizer.add_sample(0.8, is_same=True)
        optimizer.add_sample(0.3, is_same=False)
        
        assert len(optimizer._positive_scores) == 1
        assert len(optimizer._negative_scores) == 1
    
    def test_add_batch(self):
        """Test adding batch of samples."""
        optimizer = ThresholdOptimizer()
        
        scores = [0.9, 0.7, 0.4, 0.2]
        labels = [True, True, False, False]
        
        optimizer.add_batch(scores, labels)
        
        assert len(optimizer._positive_scores) == 2
        assert len(optimizer._negative_scores) == 2
    
    def test_compute_metrics(self):
        """Test computing metrics at threshold."""
        optimizer = ThresholdOptimizer()
        
        # Add clear separation
        for _ in range(50):
            optimizer.add_sample(0.9 + np.random.randn() * 0.05, is_same=True)
            optimizer.add_sample(0.3 + np.random.randn() * 0.05, is_same=False)
        
        metrics = optimizer.compute_metrics(threshold=0.6)
        
        assert metrics.accuracy > 0.9
        assert metrics.precision > 0.9
        assert metrics.recall > 0.9
    
    def test_find_eer_threshold(self):
        """Test finding EER threshold."""
        optimizer = ThresholdOptimizer()
        
        # Add samples with overlap
        for _ in range(100):
            optimizer.add_sample(0.7 + np.random.randn() * 0.15, is_same=True)
            optimizer.add_sample(0.4 + np.random.randn() * 0.15, is_same=False)
        
        eer_threshold = optimizer.find_eer_threshold()
        
        # EER threshold should be around 0.55 for this distribution
        assert 0.4 < eer_threshold < 0.7
    
    def test_find_threshold_at_far(self):
        """Test finding threshold at target FAR."""
        optimizer = ThresholdOptimizer()
        
        for _ in range(100):
            optimizer.add_sample(0.8 + np.random.randn() * 0.1, is_same=True)
            optimizer.add_sample(0.3 + np.random.randn() * 0.1, is_same=False)
        
        threshold = optimizer.find_threshold_at_far(target_far=0.01)
        
        # Should be a high threshold to achieve low FAR
        assert threshold > 0.4
    
    def test_compute_auc(self):
        """Test computing AUC."""
        optimizer = ThresholdOptimizer()
        
        # Perfect separation
        for _ in range(50):
            optimizer.add_sample(0.9, is_same=True)
            optimizer.add_sample(0.1, is_same=False)
        
        auc = optimizer.compute_auc()
        
        # Should be close to 1.0 for perfect separation
        assert auc > 0.99
    
    def test_get_summary(self):
        """Test getting summary."""
        optimizer = ThresholdOptimizer()
        
        for _ in range(100):
            optimizer.add_sample(0.8 + np.random.randn() * 0.1, is_same=True)
            optimizer.add_sample(0.3 + np.random.randn() * 0.1, is_same=False)
        
        summary = optimizer.get_summary()
        
        assert summary["num_positive_pairs"] == 100
        assert summary["num_negative_pairs"] == 100
        assert "auc" in summary
        assert "eer" in summary
        assert "best_f1" in summary
    
    def test_reset(self):
        """Test resetting optimizer."""
        optimizer = ThresholdOptimizer()
        
        optimizer.add_sample(0.8, is_same=True)
        optimizer.add_sample(0.3, is_same=False)
        
        optimizer.reset()
        
        assert len(optimizer._positive_scores) == 0
        assert len(optimizer._negative_scores) == 0


class TestThresholdMetrics:
    """Tests for ThresholdMetrics dataclass."""
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = ThresholdMetrics(
            threshold=0.5,
            accuracy=0.9,
            precision=0.85,
            recall=0.95,
            f1_score=0.9,
            false_positive_rate=0.1,
            false_negative_rate=0.05,
            true_positive_rate=0.95,
            true_negative_rate=0.9,
        )
        
        d = metrics.to_dict()
        
        assert d["threshold"] == 0.5
        assert d["accuracy"] == 0.9
        assert d["f1_score"] == 0.9
        assert d["fpr"] == 0.1
