"""
Identity Matching Module
========================

Identity management, matching, and result ranking.
"""

from marcus_core.matching.identity import Identity, IdentityStore
from marcus_core.matching.matcher import IdentityMatcher, MatchResult
from marcus_core.matching.ranker import ResultRanker

__all__ = [
    "Identity",
    "IdentityStore",
    "IdentityMatcher",
    "MatchResult",
    "ResultRanker",
]
