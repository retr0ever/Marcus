"""
Training Module
===============

Continual learning and fine-tuning for face recognition.
"""

from marcus_core.training.metric_learning import (
    MetricLearner,
    TripletLoss,
    ArcFaceLoss,
    ContrastiveLoss,
)
from marcus_core.training.continual_learning import (
    ContinualLearner,
    HardExampleMiner,
    ExperienceBuffer,
)
from marcus_core.training.threshold_optimizer import (
    ThresholdOptimizer,
    ThresholdMetrics,
)

__all__ = [
    "MetricLearner",
    "TripletLoss",
    "ArcFaceLoss",
    "ContrastiveLoss",
    "ContinualLearner",
    "HardExampleMiner",
    "ExperienceBuffer",
    "ThresholdOptimizer",
    "ThresholdMetrics",
]
