"""
Continual Learning
==================

Incremental learning and hard example mining for face recognition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import logging
import random

try:
    import torch
    import torch.nn as nn
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
class HardExample:
    """A hard example for replay."""
    embedding: "np.ndarray"
    label: str
    loss: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceBuffer:
    """
    Buffer for storing experiences for replay.
    
    Uses reservoir sampling to maintain a representative sample
    of past experiences.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        sampling_strategy: str = "reservoir",  # reservoir, fifo, priority
    ):
        """
        Initialize experience buffer.
        
        Args:
            capacity: Maximum buffer size
            sampling_strategy: How to sample/evict experiences
        """
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        
        self._buffer: List[HardExample] = []
        self._count = 0  # Total examples seen (for reservoir sampling)
    
    def add(self, example: HardExample) -> None:
        """
        Add an example to the buffer.
        
        Args:
            example: Hard example to add
        """
        if len(self._buffer) < self.capacity:
            self._buffer.append(example)
        elif self.sampling_strategy == "reservoir":
            # Reservoir sampling
            idx = random.randint(0, self._count)
            if idx < self.capacity:
                self._buffer[idx] = example
        elif self.sampling_strategy == "fifo":
            # Replace oldest
            self._buffer.pop(0)
            self._buffer.append(example)
        elif self.sampling_strategy == "priority":
            # Replace lowest loss
            min_idx = min(range(len(self._buffer)), key=lambda i: self._buffer[i].loss)
            if example.loss > self._buffer[min_idx].loss:
                self._buffer[min_idx] = example
        
        self._count += 1
    
    def add_batch(
        self,
        embeddings: "np.ndarray",
        labels: List[str],
        losses: List[float],
    ) -> None:
        """Add a batch of examples."""
        for emb, label, loss in zip(embeddings, labels, losses):
            example = HardExample(embedding=emb, label=label, loss=loss)
            self.add(example)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple["np.ndarray", List[str]]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of examples to sample
        
        Returns:
            Tuple of (embeddings, labels)
        """
        if len(self._buffer) == 0:
            return np.array([]), []
        
        batch_size = min(batch_size, len(self._buffer))
        
        if self.sampling_strategy == "priority":
            # Sample proportional to loss
            losses = [e.loss for e in self._buffer]
            probs = np.array(losses) / sum(losses)
            indices = np.random.choice(
                len(self._buffer),
                size=batch_size,
                replace=False,
                p=probs,
            )
        else:
            indices = random.sample(range(len(self._buffer)), batch_size)
        
        embeddings = np.stack([self._buffer[i].embedding for i in indices])
        labels = [self._buffer[i].label for i in indices]
        
        return embeddings, labels
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self._buffer:
            return {"size": 0, "capacity": self.capacity}
        
        losses = [e.loss for e in self._buffer]
        labels = [e.label for e in self._buffer]
        
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "total_seen": self._count,
            "mean_loss": float(np.mean(losses)),
            "max_loss": float(np.max(losses)),
            "unique_labels": len(set(labels)),
        }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._count = 0
    
    def __len__(self) -> int:
        return len(self._buffer)


class HardExampleMiner:
    """
    Mines hard positive and negative examples for training.
    
    Hard examples are pairs where the model is uncertain or wrong:
    - Hard positives: Same identity but far apart in embedding space
    - Hard negatives: Different identities but close in embedding space
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        num_hard: int = 10,
        mining_type: str = "both",  # hard, semi-hard, both
    ):
        """
        Initialize hard example miner.
        
        Args:
            margin: Margin for semi-hard mining
            num_hard: Number of hard examples to select
            mining_type: Type of hard examples to mine
        """
        self.margin = margin
        self.num_hard = num_hard
        self.mining_type = mining_type
    
    def mine(
        self,
        embeddings: "np.ndarray",
        labels: List[str],
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Mine hard pairs from a batch.
        
        Args:
            embeddings: Shape (N, D) embeddings
            labels: List of identity labels
        
        Returns:
            Dictionary with hard positive and negative pairs
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy not installed")
        
        n = len(embeddings)
        
        # Compute pairwise distances
        distances = self._compute_distances(embeddings)
        
        # Create label matrices
        labels_array = np.array(labels)
        same_identity = labels_array[:, None] == labels_array[None, :]
        
        hard_positives = []
        hard_negatives = []
        
        for i in range(n):
            # Hard positives: same identity, large distance
            pos_mask = same_identity[i] & (np.arange(n) != i)
            if pos_mask.any():
                pos_dists = distances[i] * pos_mask
                pos_dists[~pos_mask] = -np.inf
                
                hard_pos_idx = np.argmax(pos_dists)
                if pos_dists[hard_pos_idx] > 0:
                    hard_positives.append((i, hard_pos_idx))
            
            # Hard negatives: different identity, small distance
            neg_mask = ~same_identity[i]
            if neg_mask.any():
                neg_dists = distances[i].copy()
                neg_dists[~neg_mask] = np.inf
                
                if self.mining_type in ["semi-hard", "both"]:
                    # Semi-hard: farther than positive but within margin
                    if pos_mask.any():
                        pos_dist = distances[i][pos_mask].max()
                        semi_hard_mask = (
                            neg_mask &
                            (distances[i] > pos_dist) &
                            (distances[i] < pos_dist + self.margin)
                        )
                        if semi_hard_mask.any():
                            semi_hard_dists = distances[i].copy()
                            semi_hard_dists[~semi_hard_mask] = np.inf
                            hard_neg_idx = np.argmin(semi_hard_dists)
                            hard_negatives.append((i, hard_neg_idx))
                            continue
                
                # Fallback to hardest negative
                hard_neg_idx = np.argmin(neg_dists)
                if neg_dists[hard_neg_idx] < np.inf:
                    hard_negatives.append((i, hard_neg_idx))
        
        # Limit to num_hard
        if len(hard_positives) > self.num_hard:
            # Sort by distance (largest first for positives)
            hard_positives.sort(
                key=lambda x: distances[x[0], x[1]],
                reverse=True,
            )
            hard_positives = hard_positives[:self.num_hard]
        
        if len(hard_negatives) > self.num_hard:
            # Sort by distance (smallest first for negatives)
            hard_negatives.sort(
                key=lambda x: distances[x[0], x[1]],
            )
            hard_negatives = hard_negatives[:self.num_hard]
        
        return {
            "hard_positives": hard_positives,
            "hard_negatives": hard_negatives,
        }
    
    def _compute_distances(self, embeddings: "np.ndarray") -> "np.ndarray":
        """Compute pairwise L2 distances."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        
        # Cosine distance = 1 - cosine similarity
        similarity = embeddings_norm @ embeddings_norm.T
        distances = 1 - similarity
        
        return distances


class ContinualLearner:
    """
    Continual learning wrapper for face recognition models.
    
    Prevents catastrophic forgetting while learning new identities.
    
    Strategies:
    - Experience Replay: Replay past examples during training
    - EWC: Elastic Weight Consolidation
    - Knowledge Distillation: Preserve old model's knowledge
    
    Example:
        >>> learner = ContinualLearner(
        ...     model=face_model,
        ...     strategy="replay",
        ...     buffer_size=5000,
        ... )
        >>> 
        >>> # Add new identity
        >>> learner.learn_identity(new_embeddings, "John Doe")
        >>> 
        >>> # Update with new data
        >>> learner.update(new_data, new_labels)
    """
    
    def __init__(
        self,
        model: Any,
        strategy: str = "replay",  # replay, ewc, distillation
        buffer_size: int = 5000,
        replay_ratio: float = 0.5,
        ewc_lambda: float = 1000,
    ):
        """
        Initialize continual learner.
        
        Args:
            model: Face recognition model
            strategy: Continual learning strategy
            buffer_size: Size of experience buffer
            replay_ratio: Ratio of replay to new examples
            ewc_lambda: EWC regularization strength
        """
        self.model = model
        self.strategy = strategy
        self.replay_ratio = replay_ratio
        self.ewc_lambda = ewc_lambda
        
        # Experience buffer
        self.buffer = ExperienceBuffer(
            capacity=buffer_size,
            sampling_strategy="priority",
        )
        
        # Hard example miner
        self.miner = HardExampleMiner()
        
        # EWC Fisher information
        self._fisher_info: Optional[Dict[str, "torch.Tensor"]] = None
        self._optimal_params: Optional[Dict[str, "torch.Tensor"]] = None
        
        # Knowledge distillation
        self._old_model: Optional[Any] = None
        
        # Training statistics
        self._identities_learned: Dict[str, int] = {}
        self._total_updates = 0
    
    def learn_identity(
        self,
        embeddings: "np.ndarray",
        identity_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Learn a new identity.
        
        Args:
            embeddings: Embeddings for the identity
            identity_id: Identity ID
            metadata: Additional metadata
        """
        # Add to buffer for replay
        for emb in embeddings:
            example = HardExample(
                embedding=emb,
                label=identity_id,
                loss=1.0,  # High initial loss
                metadata=metadata or {},
            )
            self.buffer.add(example)
        
        self._identities_learned[identity_id] = len(embeddings)
        
        logger.info(
            f"Added identity {identity_id} with {len(embeddings)} embeddings"
        )
    
    def update(
        self,
        embeddings: "np.ndarray",
        labels: List[str],
        losses: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Update with new data.
        
        Args:
            embeddings: New embeddings
            labels: Identity labels
            losses: Per-example losses
        
        Returns:
            Update metrics
        """
        if losses is None:
            losses = [1.0] * len(embeddings)
        
        # Add to buffer
        self.buffer.add_batch(embeddings, labels, losses)
        
        # Mine hard examples
        hard_examples = self.miner.mine(embeddings, labels)
        
        # Get replay batch if using replay
        replay_embeddings = None
        replay_labels = None
        
        if self.strategy == "replay" and len(self.buffer) > 0:
            replay_size = int(len(embeddings) * self.replay_ratio)
            replay_embeddings, replay_labels = self.buffer.sample(replay_size)
        
        self._total_updates += 1
        
        return {
            "hard_positives": len(hard_examples["hard_positives"]),
            "hard_negatives": len(hard_examples["hard_negatives"]),
            "replay_samples": len(replay_labels) if replay_labels else 0,
            "buffer_size": len(self.buffer),
        }
    
    def compute_ewc_loss(
        self,
        current_params: Dict[str, "torch.Tensor"],
    ) -> "torch.Tensor":
        """
        Compute EWC regularization loss.
        
        Args:
            current_params: Current model parameters
        
        Returns:
            EWC loss tensor
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        
        if self._fisher_info is None or self._optimal_params is None:
            return torch.tensor(0.0)
        
        ewc_loss = 0.0
        
        for name, param in current_params.items():
            if name in self._fisher_info:
                fisher = self._fisher_info[name]
                optimal = self._optimal_params[name]
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def update_fisher_info(
        self,
        model: "nn.Module",
        dataloader: Any,
    ) -> None:
        """
        Update Fisher information matrix.
        
        Called after learning a task to preserve important weights.
        """
        if not HAS_TORCH:
            return
        
        self._optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        self._fisher_info = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }
        
        model.eval()
        
        for batch in dataloader:
            model.zero_grad()
            outputs = model(batch)
            
            # Use log probability as loss
            loss = outputs.pow(2).mean()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self._fisher_info[name] += param.grad.pow(2)
        
        # Normalize
        for name in self._fisher_info:
            self._fisher_info[name] /= len(dataloader)
    
    def save_old_model(self, model: Any) -> None:
        """Save copy of model for knowledge distillation."""
        import copy
        self._old_model = copy.deepcopy(model)
        self._old_model.eval()
    
    def compute_distillation_loss(
        self,
        new_embeddings: "torch.Tensor",
        images: "torch.Tensor",
        temperature: float = 2.0,
    ) -> "torch.Tensor":
        """
        Compute knowledge distillation loss.
        
        Args:
            new_embeddings: Embeddings from new model
            images: Input images
            temperature: Distillation temperature
        
        Returns:
            Distillation loss
        """
        if not HAS_TORCH or self._old_model is None:
            return torch.tensor(0.0)
        
        with torch.no_grad():
            old_embeddings = self._old_model(images)
        
        # Cosine similarity loss
        cos_sim = torch.nn.functional.cosine_similarity(
            new_embeddings,
            old_embeddings,
            dim=1,
        )
        
        distill_loss = (1 - cos_sim).mean()
        
        return distill_loss
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get continual learning statistics."""
        return {
            "strategy": self.strategy,
            "identities_learned": len(self._identities_learned),
            "total_updates": self._total_updates,
            "buffer_stats": self.buffer.get_statistics(),
        }
    
    def reset(self) -> None:
        """Reset the learner state."""
        self.buffer.clear()
        self._fisher_info = None
        self._optimal_params = None
        self._old_model = None
        self._identities_learned.clear()
        self._total_updates = 0
