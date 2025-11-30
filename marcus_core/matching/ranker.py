"""
Result Ranker
==============

Re-ranking and scoring algorithms for match results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np

from marcus_core.matching.matcher import MatchResult


@dataclass
class RankingWeights:
    """Weights for different ranking factors."""
    face_similarity: float = 0.8
    recency: float = 0.1
    source_quality: float = 0.1
    
    def normalize(self) -> "RankingWeights":
        """Normalize weights to sum to 1."""
        total = self.face_similarity + self.recency + self.source_quality
        return RankingWeights(
            face_similarity=self.face_similarity / total,
            recency=self.recency / total,
            source_quality=self.source_quality / total,
        )


class ResultRanker:
    """
    Re-ranks match results using multiple factors.
    
    Combines face similarity with other signals like recency,
    source quality, and metadata matches for better ranking.
    
    Example:
        >>> ranker = ResultRanker(weights=RankingWeights(face_similarity=0.7, recency=0.2))
        >>> ranked = ranker.rerank(results, top_k=5)
    """
    
    # Source quality scores (higher = better)
    SOURCE_QUALITY = {
        "verified": 1.0,
        "linkedin": 0.9,
        "company_directory": 0.85,
        "authorized_dataset": 0.8,
        "public_images": 0.6,
        "manual": 0.7,
        "unknown": 0.5,
    }
    
    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        recency_decay_days: float = 365.0,
    ):
        """
        Initialize the ranker.
        
        Args:
            weights: Ranking factor weights
            recency_decay_days: Half-life for recency decay
        """
        self.weights = (weights or RankingWeights()).normalize()
        self.recency_decay_days = recency_decay_days
    
    def rerank(
        self,
        results: List[MatchResult],
        top_k: Optional[int] = None,
        boost_functions: Optional[List[Callable[[MatchResult], float]]] = None,
    ) -> List[MatchResult]:
        """
        Re-rank match results.
        
        Args:
            results: Original match results
            top_k: Number of results to return
            boost_functions: Optional custom boost functions
        
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        scored_results = []
        
        for result in results:
            # Base face similarity score
            face_score = result.score
            
            # Recency score
            recency_score = self._compute_recency_score(result)
            
            # Source quality score
            source_score = self._compute_source_score(result)
            
            # Combined score
            combined_score = (
                self.weights.face_similarity * face_score +
                self.weights.recency * recency_score +
                self.weights.source_quality * source_score
            )
            
            # Apply custom boosts
            if boost_functions:
                for boost_fn in boost_functions:
                    combined_score *= boost_fn(result)
            
            scored_results.append((combined_score, result))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Update scores in results
        ranked = []
        for combined_score, result in scored_results:
            result.score = combined_score
            ranked.append(result)
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked
    
    def _compute_recency_score(self, result: MatchResult) -> float:
        """Compute recency-based score."""
        updated_at = result.identity.updated_at
        days_old = (datetime.now() - updated_at).days
        
        # Exponential decay
        decay = np.exp(-days_old / self.recency_decay_days)
        return float(decay)
    
    def _compute_source_score(self, result: MatchResult) -> float:
        """Compute source quality score."""
        source = result.identity.source
        return self.SOURCE_QUALITY.get(source, 0.5)
    
    def deduplicate(
        self,
        results: List[MatchResult],
        similarity_threshold: float = 0.95,
    ) -> List[MatchResult]:
        """
        Remove duplicate identities from results.
        
        Keeps the highest-scoring entry for each unique identity.
        
        Args:
            results: Match results
            similarity_threshold: Threshold for considering duplicates
        
        Returns:
            Deduplicated results
        """
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            if result.identity.id not in seen_ids:
                seen_ids.add(result.identity.id)
                deduplicated.append(result)
        
        return deduplicated
    
    def filter_by_threshold(
        self,
        results: List[MatchResult],
        min_score: float = 0.6,
        max_results: Optional[int] = None,
    ) -> List[MatchResult]:
        """
        Filter results by score threshold.
        
        Args:
            results: Match results
            min_score: Minimum score to include
            max_results: Maximum number of results
        
        Returns:
            Filtered results
        """
        filtered = [r for r in results if r.score >= min_score]
        
        if max_results:
            filtered = filtered[:max_results]
        
        return filtered
    
    def group_by_confidence(
        self,
        results: List[MatchResult],
    ) -> Dict[str, List[MatchResult]]:
        """
        Group results by confidence level.
        
        Returns:
            Dict with keys: high, medium, low
        """
        groups = {
            "high": [],    # >= 0.8
            "medium": [],  # 0.6 - 0.8
            "low": [],     # < 0.6
        }
        
        for result in results:
            if result.score >= 0.8:
                groups["high"].append(result)
            elif result.score >= 0.6:
                groups["medium"].append(result)
            else:
                groups["low"].append(result)
        
        return groups


class EnsembleRanker(ResultRanker):
    """
    Ensemble ranker combining multiple ranking strategies.
    """
    
    def __init__(
        self,
        rankers: List[ResultRanker],
        ranker_weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble ranker.
        
        Args:
            rankers: List of rankers to combine
            ranker_weights: Weights for each ranker
        """
        super().__init__()
        self.rankers = rankers
        
        if ranker_weights is None:
            ranker_weights = [1.0 / len(rankers)] * len(rankers)
        
        # Normalize weights
        total = sum(ranker_weights)
        self.ranker_weights = [w / total for w in ranker_weights]
    
    def rerank(
        self,
        results: List[MatchResult],
        top_k: Optional[int] = None,
        boost_functions: Optional[List[Callable[[MatchResult], float]]] = None,
    ) -> List[MatchResult]:
        """Re-rank using ensemble of rankers."""
        if not results:
            return []
        
        # Collect scores from each ranker
        all_scores: Dict[str, List[float]] = {r.identity.id: [] for r in results}
        
        for ranker, weight in zip(self.rankers, self.ranker_weights):
            ranked = ranker.rerank(results.copy())
            for result in ranked:
                all_scores[result.identity.id].append(result.score * weight)
        
        # Combine scores
        combined_scores = {
            id: sum(scores) for id, scores in all_scores.items()
        }
        
        # Update and sort
        for result in results:
            result.score = combined_scores.get(result.identity.id, 0.0)
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
