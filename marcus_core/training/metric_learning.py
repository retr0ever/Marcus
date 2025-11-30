"""
Metric Learning
===============

Loss functions and training utilities for face embedding learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import math
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for metric learning."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    margin: float = 0.5
    scale: float = 64.0
    embedding_dim: int = 512
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    fp16: bool = False
    device: str = "cuda"


class TripletLoss:
    """
    Triplet loss for metric learning.
    
    Encourages embeddings of the same identity to be closer than
    embeddings of different identities.
    
    L = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        mining_strategy: str = "semi-hard",  # hard, semi-hard, all
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Triplet margin
            mining_strategy: How to select triplets
        """
        self.margin = margin
        self.mining_strategy = mining_strategy
    
    def __call__(
        self,
        embeddings: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute triplet loss.
        
        Args:
            embeddings: Shape (N, D) embeddings
            labels: Shape (N,) identity labels
        
        Returns:
            Loss tensor
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        # Mine triplets
        if self.mining_strategy == "hard":
            loss = self._hard_triplet_loss(distances, labels)
        elif self.mining_strategy == "semi-hard":
            loss = self._semi_hard_triplet_loss(distances, labels)
        else:
            loss = self._all_triplet_loss(distances, labels)
        
        return loss
    
    def _pairwise_distances(
        self,
        embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute pairwise L2 distances."""
        dot_product = torch.mm(embeddings, embeddings.t())
        squared_norm = torch.diag(dot_product)
        distances = (
            squared_norm.unsqueeze(0)
            - 2.0 * dot_product
            + squared_norm.unsqueeze(1)
        )
        distances = torch.clamp(distances, min=0.0)
        return torch.sqrt(distances + 1e-8)
    
    def _hard_triplet_loss(
        self,
        distances: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Batch hard triplet loss."""
        # Create masks
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        
        # Hardest positive
        pos_distances = distances * pos_mask.float()
        hardest_pos = pos_distances.max(dim=1)[0]
        
        # Hardest negative
        neg_distances = distances + pos_mask.float() * 1e6
        hardest_neg = neg_distances.min(dim=1)[0]
        
        # Triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        
        return loss.mean()
    
    def _semi_hard_triplet_loss(
        self,
        distances: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Semi-hard triplet loss."""
        batch_size = embeddings.size(0)
        
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        
        # For each anchor-positive pair, find semi-hard negatives
        anchor_pos_dist = distances.unsqueeze(2)
        anchor_neg_dist = distances.unsqueeze(1)
        
        # Semi-hard: d(a,n) > d(a,p) AND d(a,n) < d(a,p) + margin
        semi_hard_mask = (
            (anchor_neg_dist > anchor_pos_dist) &
            (anchor_neg_dist < anchor_pos_dist + self.margin) &
            pos_mask.unsqueeze(2) &
            neg_mask.unsqueeze(1)
        )
        
        # Get loss for valid triplets
        triplet_loss = anchor_pos_dist - anchor_neg_dist + self.margin
        triplet_loss = triplet_loss * semi_hard_mask.float()
        
        # Average over valid triplets
        num_valid = semi_hard_mask.sum()
        if num_valid > 0:
            return triplet_loss.sum() / num_valid.float()
        else:
            return self._hard_triplet_loss(distances, labels)
    
    def _all_triplet_loss(
        self,
        distances: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """All-pairs triplet loss."""
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        
        anchor_pos = distances.unsqueeze(2) * pos_mask.unsqueeze(2).float()
        anchor_neg = distances.unsqueeze(1) * neg_mask.unsqueeze(1).float()
        
        triplet_loss = F.relu(anchor_pos - anchor_neg + self.margin)
        
        # Mask invalid triplets
        valid_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        triplet_loss = triplet_loss * valid_mask.float()
        
        num_valid = valid_mask.sum()
        if num_valid > 0:
            return triplet_loss.sum() / num_valid.float()
        else:
            return torch.tensor(0.0, device=distances.device)


class ArcFaceLoss:
    """
    ArcFace (Additive Angular Margin) Loss.
    
    Adds angular margin penalty to the softmax loss for better
    discriminative embeddings.
    
    L = -log(exp(s*cos(θ + m)) / (exp(s*cos(θ + m)) + Σexp(s*cos(θj))))
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        """
        Initialize ArcFace loss.
        
        Args:
            num_classes: Number of identities
            embedding_dim: Embedding dimension
            scale: Scaling factor
            margin: Angular margin in radians
            easy_margin: Use easy margin version
        """
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Pre-compute cos/sin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # Weight matrix (class centers)
        if HAS_TORCH:
            self.weight = nn.Parameter(
                torch.FloatTensor(num_classes, embedding_dim)
            )
            nn.init.xavier_uniform_(self.weight)
    
    def __call__(
        self,
        embeddings: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Shape (N, D) L2-normalized embeddings
            labels: Shape (N,) identity labels
        
        Returns:
            Loss tensor
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        
        # Normalize features and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Compute cos(θ + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-8)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # Create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class ContrastiveLoss:
    """
    Contrastive loss for pairs of embeddings.
    
    L = y * d² + (1-y) * max(0, margin - d)²
    
    Where y=1 for same identity, y=0 for different.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
        """
        self.margin = margin
    
    def __call__(
        self,
        embedding1: "torch.Tensor",
        embedding2: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embeddings (N, D)
            embedding2: Second embeddings (N, D)
            labels: 1 for same, 0 for different (N,)
        
        Returns:
            Loss tensor
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        
        # Compute pairwise distances
        distances = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        pos_loss = labels * distances.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = pos_loss + neg_loss
        
        return loss.mean()


class MetricLearner:
    """
    Trainer for metric learning on face embeddings.
    
    Supports multiple loss functions and training strategies.
    
    Example:
        >>> learner = MetricLearner(
        ...     backbone=resnet_model,
        ...     loss_type="arcface",
        ...     num_classes=10000,
        ... )
        >>> 
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         loss = learner.train_step(batch)
    """
    
    def __init__(
        self,
        backbone: "nn.Module",
        loss_type: str = "triplet",  # triplet, arcface, contrastive
        config: Optional[TrainingConfig] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Initialize metric learner.
        
        Args:
            backbone: Feature extraction backbone
            loss_type: Type of loss function
            config: Training configuration
            num_classes: Number of classes (required for arcface)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        
        self.backbone = backbone
        self.loss_type = loss_type
        self.config = config or TrainingConfig()
        
        # Create loss function
        if loss_type == "triplet":
            self.loss_fn = TripletLoss(margin=self.config.margin)
        elif loss_type == "arcface":
            if num_classes is None:
                raise ValueError("num_classes required for ArcFace")
            self.loss_fn = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=self.config.embedding_dim,
                scale=self.config.scale,
                margin=self.config.margin,
            )
        elif loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss(margin=self.config.margin)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Create optimizer
        params = list(backbone.parameters())
        if hasattr(self.loss_fn, "weight"):
            params.append(self.loss_fn.weight)
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
        )
        
        # AMP for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Move to device
        self.device = torch.device(self.config.device)
        self.backbone.to(self.device)
        
        if hasattr(self.loss_fn, "weight"):
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def train_step(
        self,
        images: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> float:
        """
        Execute one training step.
        
        Args:
            images: Batch of images (N, C, H, W)
            labels: Identity labels (N,)
        
        Returns:
            Loss value
        """
        self.backbone.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            with torch.cuda.amp.autocast():
                embeddings = self.backbone(images)
                loss = self.loss_fn(embeddings, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            embeddings = self.backbone(images)
            loss = self.loss_fn(embeddings, labels)
            
            loss.backward()
            self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def train_epoch(
        self,
        dataloader: "torch.utils.data.DataLoader",
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Epoch metrics
        """
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            loss = self.train_step(images, labels)
            total_loss += loss
            num_batches += 1
        
        self.scheduler.step()
        self.epoch += 1
        
        return {
            "loss": total_loss / num_batches,
            "epoch": self.epoch,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        images: "torch.Tensor",
    ) -> "np.ndarray":
        """
        Extract embeddings for evaluation.
        
        Args:
            images: Batch of images
        
        Returns:
            Embeddings as numpy array
        """
        self.backbone.eval()
        images = images.to(self.device)
        embeddings = self.backbone(images)
        return embeddings.cpu().numpy()
    
    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "backbone_state": self.backbone.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.__dict__,
        }
        
        if hasattr(self.loss_fn, "weight"):
            checkpoint["loss_weight"] = self.loss_fn.weight.data
        
        if self.scaler:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        path: str,
    ) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.backbone.load_state_dict(checkpoint["backbone_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        if "loss_weight" in checkpoint and hasattr(self.loss_fn, "weight"):
            self.loss_fn.weight.data = checkpoint["loss_weight"]
        
        if self.scaler and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
