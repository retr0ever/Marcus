"""
Facial Analysis Pipeline
=========================

End-to-end pipeline for face detection, embedding, matching, and search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import hashlib
import numpy as np

from marcus_core.config import SystemConfig, load_config
from marcus_core.detection import FaceDetector, YOLOFaceDetector, FaceDetection
from marcus_core.embedding import EmbeddingExtractor, ArcFaceExtractor
from marcus_core.vectordb import VectorStore, FAISSStore
from marcus_core.matching import Identity, IdentityStore, IdentityMatcher, MatchResult, ResultRanker
from marcus_core.compliance import AuditLogger, ConsentManager


@dataclass
class FaceResult:
    """
    Result for a single detected face.
    
    Attributes:
        detection: Face detection data
        embedding: Face embedding vector
        matches: List of identity matches
        quality_score: Face quality assessment
    """
    detection: FaceDetection
    embedding: Optional[np.ndarray] = None
    matches: List[MatchResult] = field(default_factory=list)
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": list(self.detection.bbox),
            "confidence": self.detection.confidence,
            "has_embedding": self.embedding is not None,
            "num_matches": len(self.matches),
            "top_match": self.matches[0].to_dict() if self.matches else None,
            "quality_score": self.quality_score,
        }


class FacialPipeline:
    """
    Complete facial analysis pipeline.
    
    Combines face detection, embedding extraction, vector search,
    and identity matching with compliance logging.
    
    Example:
        >>> pipeline = FacialPipeline.from_config("configs/default.yaml")
        >>> 
        >>> # Search for identity
        >>> results = pipeline.search(image)
        >>> for face in results:
        ...     for match in face.matches:
        ...         print(f"Match: {match.identity.name} ({match.score:.2%})")
        >>> 
        >>> # Enroll new identity
        >>> identity_id = pipeline.enroll(
        ...     image=photo,
        ...     name="John Doe",
        ...     metadata={"company": "Acme Inc"}
        ... )
    """
    
    def __init__(
        self,
        config: SystemConfig,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[EmbeddingExtractor] = None,
        vector_store: Optional[VectorStore] = None,
        identity_store: Optional[IdentityStore] = None,
        audit_logger: Optional[AuditLogger] = None,
        consent_manager: Optional[ConsentManager] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: System configuration
            detector: Face detector (created from config if None)
            embedder: Embedding extractor (created from config if None)
            vector_store: Vector database (created from config if None)
            identity_store: Identity storage (created from config if None)
            audit_logger: Audit logger (created from config if None)
            consent_manager: Consent manager (created from config if None)
        """
        self.config = config
        
        # Initialize components
        self.detector = detector or self._create_detector()
        self.embedder = embedder or self._create_embedder()
        self.vector_store = vector_store or self._create_vector_store()
        self.identity_store = identity_store or self._create_identity_store()
        
        # Matcher combines vector store and identity store
        self.matcher = IdentityMatcher(
            vector_store=self.vector_store,
            identity_store=self.identity_store,
            similarity_threshold=config.matching.similarity_threshold,
            top_k=config.matching.top_k,
        )
        
        # Ranker for result re-ranking
        self.ranker = ResultRanker()
        
        # Compliance components
        self.audit_logger = audit_logger or self._create_audit_logger()
        self.consent_manager = consent_manager or self._create_consent_manager()
        
        # Current user context
        self._current_user = "system"
        self._current_session = None
    
    @classmethod
    def from_config(cls, config_path: str) -> "FacialPipeline":
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            Configured FacialPipeline instance
        """
        config = load_config(config_path)
        return cls(config)
    
    def _create_detector(self) -> FaceDetector:
        """Create face detector from config."""
        return YOLOFaceDetector(
            model=self.config.detection.model,
            model_path=self.config.detection.model_path,
            confidence_threshold=self.config.detection.confidence_threshold,
            nms_threshold=self.config.detection.nms_threshold,
            min_face_size=self.config.detection.min_face_size,
            max_faces=self.config.detection.max_faces,
            device=self.config.detection.device,
            input_size=self.config.detection.input_size,
            align_faces=self.config.alignment.enabled,
            target_face_size=self.config.alignment.target_size,
        )
    
    def _create_embedder(self) -> EmbeddingExtractor:
        """Create embedding extractor from config."""
        return ArcFaceExtractor(
            model_path=self.config.embedding.model_path,
            backbone=self.config.embedding.backbone,
            embedding_dim=self.config.embedding.embedding_dim,
            normalize=self.config.embedding.normalize,
            device=self.config.embedding.device,
            fp16=self.config.embedding.fp16,
            batch_size=self.config.embedding.batch_size,
        )
    
    def _create_vector_store(self) -> VectorStore:
        """Create vector store from config."""
        store = FAISSStore(
            dimension=self.config.vectordb.dimension,
            metric=self.config.vectordb.metric,
            index_type=self.config.vectordb.index_type,
            nlist=self.config.vectordb.nlist,
            nprobe=self.config.vectordb.nprobe,
            persist_path=self.config.vectordb.persist_path,
            auto_save=self.config.vectordb.auto_save,
        )
        
        # Load existing data
        if Path(self.config.vectordb.persist_path).exists():
            store.load()
        
        return store
    
    def _create_identity_store(self) -> IdentityStore:
        """Create identity store from config."""
        return IdentityStore(
            persist_path=str(Path(self.config.data_dir) / "identities")
        )
    
    def _create_audit_logger(self) -> AuditLogger:
        """Create audit logger from config."""
        return AuditLogger(
            log_path=self.config.compliance.audit_log_path,
            retention_days=self.config.compliance.log_retention_days,
            enabled=self.config.compliance.enabled,
        )
    
    def _create_consent_manager(self) -> ConsentManager:
        """Create consent manager from config."""
        return ConsentManager(
            persist_path=str(Path(self.config.data_dir) / "consent"),
            default_expiry_days=self.config.compliance.consent_expiry_days,
            require_explicit_consent=self.config.compliance.require_consent,
        )
    
    def set_user_context(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Set current user context for audit logging."""
        self._current_user = user_id
        self._current_session = session_id
    
    def detect(
        self,
        image: np.ndarray,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            List of face detections
        """
        return self.detector.detect(image)
    
    def detect_and_embed(
        self,
        image: np.ndarray,
    ) -> List[FaceResult]:
        """
        Detect faces and extract embeddings.
        
        Args:
            image: RGB image
        
        Returns:
            List of FaceResult with detections and embeddings
        """
        # Detect faces
        detections = self.detector.detect(image)
        
        results = []
        for detection in detections:
            result = FaceResult(detection=detection)
            
            # Extract embedding if aligned face is available
            if detection.aligned_face is not None:
                result.embedding = self.embedder.extract(detection.aligned_face)
            
            results.append(result)
        
        return results
    
    def search(
        self,
        image: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ) -> List[FaceResult]:
        """
        Search for identities matching faces in an image.
        
        Args:
            image: RGB image
            top_k: Number of matches per face
            threshold: Minimum similarity threshold
            rerank: Apply result re-ranking
        
        Returns:
            List of FaceResult with matches
        """
        top_k = top_k or self.config.matching.top_k
        threshold = threshold or self.config.matching.similarity_threshold
        
        # Detect and embed
        face_results = self.detect_and_embed(image)
        
        # Search for each face
        for result in face_results:
            if result.embedding is not None:
                matches = self.matcher.match(
                    result.embedding,
                    top_k=top_k,
                    threshold=threshold,
                )
                
                if rerank and matches:
                    matches = self.ranker.rerank(matches, top_k=top_k)
                
                result.matches = matches
        
        # Log search
        if self.audit_logger and face_results:
            query_hash = self._hash_image(image)
            top_match = None
            top_score = None
            
            for r in face_results:
                if r.matches:
                    if top_match is None or r.matches[0].score > top_score:
                        top_match = r.matches[0].identity.id
                        top_score = r.matches[0].score
            
            self.audit_logger.log_search(
                user_id=self._current_user,
                query_hash=query_hash,
                results_count=sum(len(r.matches) for r in face_results),
                top_match_id=top_match,
                top_match_score=top_score,
                session_id=self._current_session,
            )
        
        return face_results
    
    def enroll(
        self,
        image: np.ndarray,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        require_consent: bool = True,
    ) -> Optional[str]:
        """
        Enroll a new identity from an image.
        
        Args:
            image: RGB image with face
            name: Identity name
            metadata: Additional metadata
            source: Data source
            require_consent: Check/create consent
        
        Returns:
            Identity ID or None if failed
        """
        # Detect and embed
        results = self.detect_and_embed(image)
        
        if not results:
            return None
        
        # Use the first/largest face
        face_result = max(results, key=lambda x: x.detection.area)
        
        if face_result.embedding is None:
            return None
        
        # Create identity
        identity = Identity(
            name=name,
            metadata=metadata or {},
            source=source,
        )
        
        # Grant consent if required
        if require_consent:
            self.consent_manager.grant_consent(
                identity_id=identity.id,
                granted_by=self._current_user,
                source=source,
            )
        
        # Enroll
        identity_id = self.matcher.enroll(
            identity=identity,
            embeddings=[face_result.embedding],
        )
        
        # Log enrollment
        if self.audit_logger:
            self.audit_logger.log_enrollment(
                user_id=self._current_user,
                identity_id=identity_id,
                source=source,
                name=name,
                num_embeddings=1,
            )
        
        # Save stores
        self.identity_store.save()
        self.vector_store.save()
        
        return identity_id
    
    def enroll_multiple(
        self,
        images: List[np.ndarray],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "manual",
    ) -> Optional[str]:
        """
        Enroll identity with multiple reference images.
        
        Args:
            images: List of RGB images
            name: Identity name
            metadata: Additional metadata
            source: Data source
        
        Returns:
            Identity ID or None if failed
        """
        embeddings = []
        
        for image in images:
            results = self.detect_and_embed(image)
            if results and results[0].embedding is not None:
                embeddings.append(results[0].embedding)
        
        if not embeddings:
            return None
        
        # Create identity
        identity = Identity(
            name=name,
            metadata=metadata or {},
            source=source,
        )
        
        # Grant consent
        self.consent_manager.grant_consent(
            identity_id=identity.id,
            granted_by=self._current_user,
            source=source,
        )
        
        # Enroll with all embeddings
        identity_id = self.matcher.enroll(
            identity=identity,
            embeddings=embeddings,
        )
        
        # Log
        if self.audit_logger:
            self.audit_logger.log_enrollment(
                user_id=self._current_user,
                identity_id=identity_id,
                source=source,
                name=name,
                num_embeddings=len(embeddings),
            )
        
        self.identity_store.save()
        self.vector_store.save()
        
        return identity_id
    
    def delete_identity(
        self,
        identity_id: str,
        reason: str = "user_request",
    ) -> bool:
        """
        Delete an identity and all associated data.
        
        Implements right-to-erasure for GDPR compliance.
        
        Args:
            identity_id: Identity to delete
            reason: Reason for deletion
        
        Returns:
            True if deleted
        """
        # Revoke consent
        self.consent_manager.revoke_consent(
            identity_id=identity_id,
            revoked_by=self._current_user,
            reason=reason,
        )
        
        # Delete from matcher (removes from vector store and identity store)
        success = self.matcher.unenroll(identity_id)
        
        if success:
            # Delete consent records
            self.consent_manager.delete_all_for_identity(identity_id)
            
            # Log deletion
            if self.audit_logger:
                self.audit_logger.log_deletion(
                    user_id=self._current_user,
                    identity_id=identity_id,
                    reason=reason,
                )
            
            # Save stores
            self.identity_store.save()
            self.vector_store.save()
        
        return success
    
    def verify(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Verify if two images show the same person.
        
        Args:
            image1: First image
            image2: Second image
        
        Returns:
            Verification result with similarity and decision
        """
        # Get embeddings
        results1 = self.detect_and_embed(image1)
        results2 = self.detect_and_embed(image2)
        
        if not results1 or not results2:
            return {
                "error": "Face not detected in one or both images",
                "is_match": False,
            }
        
        emb1 = results1[0].embedding
        emb2 = results2[0].embedding
        
        if emb1 is None or emb2 is None:
            return {
                "error": "Failed to extract embeddings",
                "is_match": False,
            }
        
        return self.matcher.verify(emb1, emb2)
    
    def get_identity(self, identity_id: str) -> Optional[Identity]:
        """Get identity by ID."""
        identity = self.identity_store.get(identity_id)
        
        if identity and self.audit_logger:
            self.audit_logger.log_view(
                user_id=self._current_user,
                identity_id=identity_id,
            )
        
        return identity
    
    def list_identities(
        self,
        query: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Identity]:
        """List identities with optional filtering."""
        return self.identity_store.search(
            query=query,
            source=source,
        )[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_identities": self.identity_store.count(),
            "total_embeddings": self.vector_store.count(),
            "consent_stats": self.consent_manager.get_statistics(),
            "matcher_stats": self.matcher.get_statistics(),
            "config": {
                "similarity_threshold": self.config.matching.similarity_threshold,
                "top_k": self.config.matching.top_k,
            },
        }
    
    def warmup(self) -> None:
        """Warm up all models."""
        self.detector.warmup()
        self.embedder.warmup()
    
    def _hash_image(self, image: np.ndarray) -> str:
        """Create hash of image for audit logging."""
        return hashlib.sha256(image.tobytes()).hexdigest()[:16]
    
    def save(self) -> None:
        """Save all persistent data."""
        self.vector_store.save()
        self.identity_store.save()
        self.consent_manager.save()
    
    def __repr__(self) -> str:
        return (
            f"FacialPipeline("
            f"identities={self.identity_store.count()}, "
            f"embeddings={self.vector_store.count()})"
        )
