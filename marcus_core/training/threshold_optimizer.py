"""
Threshold Optimizer
===================

Optimize decision thresholds for face verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold."""
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    true_positive_rate: float
    true_negative_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "fpr": self.false_positive_rate,
            "fnr": self.false_negative_rate,
            "tpr": self.true_positive_rate,
            "tnr": self.true_negative_rate,
        }


class ThresholdOptimizer:
    """
    Optimize face verification thresholds.
    
    Finds optimal thresholds for different operating points:
    - Equal Error Rate (EER)
    - Target False Accept Rate (FAR)
    - Target False Reject Rate (FRR)
    - Maximum F1 score
    
    Example:
        >>> optimizer = ThresholdOptimizer()
        >>> 
        >>> # Add verification pairs
        >>> for score, is_same in verification_pairs:
        ...     optimizer.add_sample(score, is_same)
        >>> 
        >>> # Find optimal thresholds
        >>> eer_threshold = optimizer.find_eer_threshold()
        >>> print(f"EER threshold: {eer_threshold:.3f}")
        >>> 
        >>> # Get metrics at threshold
        >>> metrics = optimizer.compute_metrics(threshold=0.5)
        >>> print(f"F1: {metrics.f1_score:.3f}")
    """
    
    def __init__(
        self,
        num_thresholds: int = 1000,
    ):
        """
        Initialize optimizer.
        
        Args:
            num_thresholds: Number of thresholds to evaluate
        """
        self.num_thresholds = num_thresholds
        
        # Store positive and negative scores
        self._positive_scores: List[float] = []  # Same identity
        self._negative_scores: List[float] = []  # Different identity
    
    def add_sample(
        self,
        score: float,
        is_same: bool,
    ) -> None:
        """
        Add a verification sample.
        
        Args:
            score: Similarity score
            is_same: True if same identity
        """
        if is_same:
            self._positive_scores.append(score)
        else:
            self._negative_scores.append(score)
    
    def add_batch(
        self,
        scores: List[float],
        labels: List[bool],
    ) -> None:
        """
        Add a batch of samples.
        
        Args:
            scores: List of similarity scores
            labels: List of same/different labels
        """
        for score, is_same in zip(scores, labels):
            self.add_sample(score, is_same)
    
    def compute_metrics(
        self,
        threshold: float,
    ) -> ThresholdMetrics:
        """
        Compute metrics at a specific threshold.
        
        Args:
            threshold: Decision threshold
        
        Returns:
            Metrics at threshold
        """
        if not self._positive_scores or not self._negative_scores:
            return ThresholdMetrics(
                threshold=threshold,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                true_positive_rate=0.0,
                true_negative_rate=0.0,
            )
        
        pos = np.array(self._positive_scores)
        neg = np.array(self._negative_scores)
        
        # True positives: same identity predicted same
        tp = (pos >= threshold).sum()
        # False negatives: same identity predicted different
        fn = (pos < threshold).sum()
        # True negatives: different identity predicted different
        tn = (neg < threshold).sum()
        # False positives: different identity predicted same
        fp = (neg >= threshold).sum()
        
        total = tp + fn + tn + fp
        
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return ThresholdMetrics(
            threshold=float(threshold),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            false_positive_rate=float(fpr),
            false_negative_rate=float(fnr),
            true_positive_rate=float(tpr),
            true_negative_rate=float(tnr),
        )
    
    def find_eer_threshold(self) -> float:
        """
        Find threshold at Equal Error Rate.
        
        EER is where FAR equals FRR.
        
        Returns:
            Threshold at EER
        """
        if not self._positive_scores or not self._negative_scores:
            return 0.5
        
        pos = np.array(self._positive_scores)
        neg = np.array(self._negative_scores)
        
        # Generate thresholds
        all_scores = np.concatenate([pos, neg])
        thresholds = np.linspace(
            all_scores.min(),
            all_scores.max(),
            self.num_thresholds,
        )
        
        best_threshold = 0.5
        min_diff = float("inf")
        
        for t in thresholds:
            fpr = (neg >= t).sum() / len(neg)
            fnr = (pos < t).sum() / len(pos)
            
            diff = abs(fpr - fnr)
            if diff < min_diff:
                min_diff = diff
                best_threshold = t
        
        return float(best_threshold)
    
    def find_threshold_at_far(
        self,
        target_far: float = 0.01,
    ) -> float:
        """
        Find threshold at target False Accept Rate.
        
        Args:
            target_far: Target FAR (default 1%)
        
        Returns:
            Threshold achieving target FAR
        """
        if not self._negative_scores:
            return 0.5
        
        neg = np.array(self._negative_scores)
        neg_sorted = np.sort(neg)[::-1]  # Descending
        
        # Find threshold where FAR equals target
        idx = int(target_far * len(neg_sorted))
        idx = min(idx, len(neg_sorted) - 1)
        
        return float(neg_sorted[idx])
    
    def find_threshold_at_frr(
        self,
        target_frr: float = 0.01,
    ) -> float:
        """
        Find threshold at target False Reject Rate.
        
        Args:
            target_frr: Target FRR (default 1%)
        
        Returns:
            Threshold achieving target FRR
        """
        if not self._positive_scores:
            return 0.5
        
        pos = np.array(self._positive_scores)
        pos_sorted = np.sort(pos)  # Ascending
        
        # Find threshold where FRR equals target
        idx = int(target_frr * len(pos_sorted))
        idx = min(idx, len(pos_sorted) - 1)
        
        return float(pos_sorted[idx])
    
    def find_best_f1_threshold(self) -> Tuple[float, float]:
        """
        Find threshold with maximum F1 score.
        
        Returns:
            Tuple of (threshold, f1_score)
        """
        if not self._positive_scores or not self._negative_scores:
            return 0.5, 0.0
        
        pos = np.array(self._positive_scores)
        neg = np.array(self._negative_scores)
        
        all_scores = np.concatenate([pos, neg])
        thresholds = np.linspace(
            all_scores.min(),
            all_scores.max(),
            self.num_thresholds,
        )
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        for t in thresholds:
            metrics = self.compute_metrics(t)
            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                best_threshold = t
        
        return float(best_threshold), float(best_f1)
    
    def compute_roc_curve(
        self,
    ) -> Dict[str, List[float]]:
        """
        Compute ROC curve.
        
        Returns:
            Dictionary with fpr, tpr, and thresholds
        """
        if not self._positive_scores or not self._negative_scores:
            return {"fpr": [], "tpr": [], "thresholds": []}
        
        pos = np.array(self._positive_scores)
        neg = np.array(self._negative_scores)
        
        all_scores = np.concatenate([pos, neg])
        thresholds = np.linspace(
            all_scores.min(),
            all_scores.max(),
            self.num_thresholds,
        )
        
        fprs = []
        tprs = []
        
        for t in thresholds:
            fpr = (neg >= t).sum() / len(neg)
            tpr = (pos >= t).sum() / len(pos)
            fprs.append(float(fpr))
            tprs.append(float(tpr))
        
        return {
            "fpr": fprs,
            "tpr": tprs,
            "thresholds": thresholds.tolist(),
        }
    
    def compute_auc(self) -> float:
        """
        Compute Area Under ROC Curve.
        
        Returns:
            AUC value
        """
        roc = self.compute_roc_curve()
        
        if not roc["fpr"] or not roc["tpr"]:
            return 0.0
        
        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])
        
        # Sort by FPR
        sorted_indices = np.argsort(fpr)
        fpr = fpr[sorted_indices]
        tpr = tpr[sorted_indices]
        
        # Trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return float(auc)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of threshold optimization.
        
        Returns:
            Summary dictionary
        """
        eer_threshold = self.find_eer_threshold()
        eer_metrics = self.compute_metrics(eer_threshold)
        
        far_threshold = self.find_threshold_at_far(0.01)
        far_metrics = self.compute_metrics(far_threshold)
        
        f1_threshold, f1_score = self.find_best_f1_threshold()
        
        return {
            "num_positive_pairs": len(self._positive_scores),
            "num_negative_pairs": len(self._negative_scores),
            "auc": self.compute_auc(),
            "eer": {
                "threshold": eer_threshold,
                "error_rate": eer_metrics.false_positive_rate,
            },
            "far_0.01": {
                "threshold": far_threshold,
                "tpr": far_metrics.true_positive_rate,
            },
            "best_f1": {
                "threshold": f1_threshold,
                "f1_score": f1_score,
            },
        }
    
    def reset(self) -> None:
        """Reset all samples."""
        self._positive_scores.clear()
        self._negative_scores.clear()
    
    def __repr__(self) -> str:
        return (
            f"ThresholdOptimizer("
            f"positive={len(self._positive_scores)}, "
            f"negative={len(self._negative_scores)})"
        )
