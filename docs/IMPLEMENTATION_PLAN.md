# MARCUS Facial Analysis System - Implementation Plan

## Overview

Build a modular, OSINT-enabled facial analysis system with:
- **YOLO-based real-time face detection**
- **Deep facial embeddings** (ArcFace, CosFace, Triplet Loss)
- **Vector database** (FAISS/HNSWlib) for fast similarity search
- **Trainable pipeline** with metric learning and continual learning
- **Multi-source OSINT ingestion** (LinkedIn, public images, authorized datasets)
- **UK GDPR compliance** with full audit logging
- **Streamlit test UI** for webcam/photo matching

---

## Project Structure

```
marcus-codebase/
├── marcus_core/                    # ← REUSABLE LIBRARY (pip installable)
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   │
│   ├── detection/                  # Face Detection Module
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract detector interface
│   │   ├── yolo_detector.py        # YOLOv8-Face implementation
│   │   └── preprocessing.py        # Face alignment & normalization
│   │
│   ├── embedding/                  # Embedding Extraction Module
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract embedding interface
│   │   ├── arcface.py              # ArcFace model wrapper
│   │   ├── cosface.py              # CosFace model wrapper
│   │   └── model_zoo.py            # Pre-trained model registry
│   │
│   ├── vectordb/                   # Vector Database Module
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract vector store interface
│   │   ├── faiss_store.py          # FAISS implementation
│   │   └── hnswlib_store.py        # HNSWlib implementation
│   │
│   ├── matching/                   # Identity Matching Module
│   │   ├── __init__.py
│   │   ├── matcher.py              # Core matching logic
│   │   ├── ranker.py               # Result ranking algorithms
│   │   └── identity.py             # Identity data structures
│   │
│   ├── training/                   # Training & Fine-tuning Module
│   │   ├── __init__.py
│   │   ├── metric_learning.py      # ArcFace, CosFace, Triplet losses
│   │   ├── continual_learning.py   # Incremental learning strategies
│   │   ├── clustering_feedback.py  # Embedding refinement via clustering
│   │   └── threshold_optimizer.py  # Optimal threshold tuning
│   │
│   ├── osint/                      # OSINT Ingestion Module
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract source interface
│   │   ├── source_registry.py      # Source management
│   │   ├── linkedin_adapter.py     # LinkedIn public data adapter
│   │   ├── web_image_adapter.py    # Public web images adapter
│   │   └── dataset_adapter.py      # Authorized dataset adapter
│   │
│   ├── compliance/                 # UK GDPR Compliance Module
│   │   ├── __init__.py
│   │   ├── audit_logger.py         # Access and operation logging
│   │   ├── consent_manager.py      # Consent tracking & verification
│   │   ├── data_retention.py       # Retention policy enforcement
│   │   └── erasure_handler.py      # Right-to-erasure implementation
│   │
│   ├── pipeline/                   # Unified Pipeline
│   │   ├── __init__.py
│   │   └── facial_pipeline.py      # End-to-end processing pipeline
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── image_utils.py          # Image loading, conversion
│       ├── geometry.py             # Face geometry calculations
│       └── device.py               # Device detection (CUDA/MPS/CPU)
│
├── marcus_ui/                      # ← TEST APPLICATION
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit application
│   ├── pages/
│   │   ├── 1_live_detection.py     # Webcam live detection
│   │   ├── 2_photo_search.py       # Upload photo to search
│   │   ├── 3_enroll_identity.py    # Add new identities
│   │   ├── 4_manage_database.py    # View/manage enrolled faces
│   │   └── 5_training.py           # Fine-tuning interface
│   └── components/
│       ├── webcam.py               # Webcam capture component
│       ├── face_card.py            # Match result display card
│       └── metrics.py              # Performance metrics display
│
├── tests/                          # Test Suite
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_embedding.py
│   ├── test_vectordb.py
│   ├── test_matching.py
│   └── test_compliance.py
│
├── configs/                        # Configuration Files
│   ├── default.yaml                # Default system configuration
│   ├── production.yaml             # Production settings
│   └── gdpr_policies.yaml          # GDPR policy definitions
│
├── models/                         # Model Weights (gitignored)
│   └── .gitkeep
│
├── data/                           # Data Storage (gitignored)
│   ├── vector_db/                  # FAISS indices
│   ├── identities/                 # Identity metadata
│   ├── osint_cache/                # Cached OSINT data
│   └── audit_logs/                 # Compliance logs
│
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── .env.example                    # Environment template
├── .gitignore
├── README.md                       # User documentation
└── IMPLEMENTATION_PLAN.md          # This file
```

---

## Step-by-Step Implementation

### Phase 1: Project Foundation

#### Step 1.1: Initialize Package Structure
Create all directories and `__init__.py` files as shown above.

```bash
# Create directory structure
mkdir -p marcus_core/{detection,embedding,vectordb,matching,training,osint,compliance,pipeline,utils}
mkdir -p marcus_ui/{pages,components}
mkdir -p tests configs models data/{vector_db,identities,osint_cache,audit_logs}
```

#### Step 1.2: Create `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "marcus-core"
version = "1.0.0"
description = "OSINT-enabled facial analysis system"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "faiss-cpu>=1.7.4",
    "hnswlib>=0.7.0",
    "insightface>=0.7.3",
    "onnxruntime-gpu>=1.16.0",
    "pillow>=10.0.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",
    "scikit-learn>=1.3.0",
]

[project.optional-dependencies]
ui = ["streamlit>=1.28.0", "streamlit-webrtc>=0.45.0"]
dev = ["pytest>=7.0.0", "black", "ruff", "mypy"]
```

#### Step 1.3: Create `requirements.txt`
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
faiss-cpu>=1.7.4
hnswlib>=0.7.0
insightface>=0.7.3
onnxruntime-gpu>=1.16.0
pillow>=10.0.0
pyyaml>=6.0
pydantic>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.28.0
streamlit-webrtc>=0.45.0
aiohttp>=3.9.0
requests>=2.31.0
python-dotenv>=1.0.0
```

#### Step 1.4: Create Core Configuration (`marcus_core/config.py`)
Define dataclasses for:
- `DetectionConfig`: model path, confidence threshold, NMS threshold, device
- `EmbeddingConfig`: model type (arcface/cosface), backbone, embedding dim
- `VectorDBConfig`: db type (faiss/hnsw), index type, metric, dimension
- `MatchingConfig`: similarity threshold, top-k, reranking options
- `TrainingConfig`: loss type, learning rate, batch size, epochs
- `OSINTConfig`: enabled sources, rate limits, cache settings
- `ComplianceConfig`: log retention, consent expiry, encryption settings
- `SystemConfig`: master config combining all above

---

### Phase 2: Detection Module

#### Step 2.1: Create Base Detector Interface (`marcus_core/detection/base.py`)
```python
# Abstract base class with methods:
# - detect(image) -> List[FaceDetection]
# - detect_batch(images) -> List[List[FaceDetection]]
# 
# FaceDetection dataclass:
# - bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
# - confidence: float
# - landmarks: Optional[np.ndarray]  # 5-point landmarks
# - aligned_face: Optional[np.ndarray]  # Preprocessed crop
```

#### Step 2.2: Implement YOLO Detector (`marcus_core/detection/yolo_detector.py`)
- Use `ultralytics` YOLOv8-face model
- Support GPU/CPU inference
- Return bounding boxes, confidence scores, and 5-point landmarks
- Batch processing for efficiency

#### Step 2.3: Implement Face Preprocessing (`marcus_core/detection/preprocessing.py`)
- 5-point landmark detection (eyes, nose, mouth corners)
- Affine transformation to align face
- Normalize to 112x112 RGB output
- Handle edge cases (partial faces, extreme angles)

---

### Phase 3: Embedding Module

#### Step 3.1: Create Base Embedding Interface (`marcus_core/embedding/base.py`)
```python
# Abstract base class with methods:
# - extract(face_image) -> np.ndarray  # 512-dim vector
# - extract_batch(face_images) -> np.ndarray
# - get_embedding_dim() -> int
```

#### Step 3.2: Implement Model Zoo (`marcus_core/embedding/model_zoo.py`)
- Registry of pre-trained models with download URLs
- Automatic model downloading on first use
- Support models: ArcFace-R100, ArcFace-R50, CosFace, MobileFaceNet

#### Step 3.3: Implement ArcFace Extractor (`marcus_core/embedding/arcface.py`)
- Load pre-trained ArcFace model (InsightFace or ONNX)
- GPU acceleration with automatic fallback
- FP16 inference for speed
- L2 normalization of output embeddings

#### Step 3.4: Implement CosFace Extractor (`marcus_core/embedding/cosface.py`)
- Similar structure to ArcFace
- Support for CosFace-specific models

---

### Phase 4: Vector Database Module

#### Step 4.1: Create Base VectorDB Interface (`marcus_core/vectordb/base.py`)
```python
# Abstract base class with methods:
# - add(embedding, metadata) -> str  # Returns ID
# - add_batch(embeddings, metadatas) -> List[str]
# - search(query_embedding, top_k) -> List[SearchResult]
# - delete(id) -> bool
# - save(path) / load(path)
# - get_count() -> int
```

#### Step 4.2: Implement FAISS Store (`marcus_core/vectordb/faiss_store.py`)
- Support index types: Flat, IVF_Flat, IVF_PQ, HNSW
- Configurable distance metric (cosine, L2, inner product)
- Automatic index training for IVF
- Persistent storage with metadata sidecar (JSON/SQLite)
- GPU acceleration option

#### Step 4.3: Implement HNSWlib Store (`marcus_core/vectordb/hnswlib_store.py`)
- Alternative to FAISS for smaller datasets
- Good recall with fast queries
- Simpler configuration

---

### Phase 5: Identity Matching Module

#### Step 5.1: Create Identity Data Structures (`marcus_core/matching/identity.py`)
```python
# Identity dataclass:
# - id: str (UUID)
# - name: Optional[str]
# - embeddings: List[np.ndarray]  # Multiple embeddings per person
# - metadata: Dict[str, Any]  # Profile info, source, etc.
# - created_at: datetime
# - updated_at: datetime
# - consent_status: str
```

#### Step 5.2: Implement Core Matcher (`marcus_core/matching/matcher.py`)
- Query vector database for top-k candidates
- Apply similarity threshold filtering
- Support multiple embeddings per identity (average/max pooling)
- Return ranked match results with confidence scores

#### Step 5.3: Implement Result Ranker (`marcus_core/matching/ranker.py`)
- Re-ranking strategies: similarity, recency, source quality
- Ensemble scoring combining face + metadata similarity
- Configurable ranking weights

---

### Phase 6: Training Module

#### Step 6.1: Implement Metric Learning Losses (`marcus_core/training/metric_learning.py`)
```python
# Loss implementations:
# - ArcFaceLoss(margin=0.5, scale=64)
# - CosFaceLoss(margin=0.35, scale=64)
# - TripletLoss(margin=0.3)
# - ContrastiveLoss(margin=1.0)
# 
# Each with forward(embeddings, labels) -> loss
```

#### Step 6.2: Implement Continual Learning (`marcus_core/training/continual_learning.py`)
- Experience replay buffer for old samples
- Elastic Weight Consolidation (EWC) to prevent forgetting
- Knowledge distillation from old model
- Incremental class learning support

#### Step 6.3: Implement Clustering Feedback (`marcus_core/training/clustering_feedback.py`)
- DBSCAN/HDBSCAN clustering of embeddings
- Automatic cluster quality assessment
- Suggest identity merges/splits
- Generate pseudo-labels for unlabeled data

#### Step 6.4: Implement Threshold Optimizer (`marcus_core/training/threshold_optimizer.py`)
- Grid search over threshold values
- Optimize for target FAR/FRR
- Cross-validation on held-out set
- Output optimal threshold with confidence interval

---

### Phase 7: OSINT Ingestion Module

#### Step 7.1: Create Source Registry (`marcus_core/osint/source_registry.py`)
```python
# SourceRegistry class:
# - register_source(name, adapter, config)
# - get_source(name) -> SourceAdapter
# - list_sources() -> List[str]
# - validate_authorization(source) -> bool
```

#### Step 7.2: Create Base Source Adapter (`marcus_core/osint/base.py`)
```python
# Abstract SourceAdapter:
# - fetch_profile(identifier) -> ProfileData
# - search_profiles(query) -> List[ProfileData]
# - download_image(url) -> bytes
# - get_rate_limit() -> RateLimitConfig
```

#### Step 7.3: Implement LinkedIn Adapter (`marcus_core/osint/linkedin_adapter.py`)
- Public profile data extraction (with authorization)
- Profile image downloading
- Rate limiting and retry logic
- Consent verification before ingestion

#### Step 7.4: Implement Web Image Adapter (`marcus_core/osint/web_image_adapter.py`)
- Generic public image fetching
- Image validation (format, size, quality)
- Source URL tracking for compliance

#### Step 7.5: Implement Dataset Adapter (`marcus_core/osint/dataset_adapter.py`)
- Load from authorized local/remote datasets
- Support formats: folder structure, CSV manifest, JSON
- Batch ingestion with progress tracking

---

### Phase 8: Compliance Module

#### Step 8.1: Implement Audit Logger (`marcus_core/compliance/audit_logger.py`)
```python
# AuditLogger class:
# - log_access(user, action, resource, details)
# - log_search(user, query_image_hash, results)
# - log_enrollment(user, identity_id, source)
# - log_deletion(user, identity_id, reason)
# - export_logs(start_date, end_date, format)
# 
# All logs include: timestamp, user_id, action, resource, ip_address, details
```

#### Step 8.2: Implement Consent Manager (`marcus_core/compliance/consent_manager.py`)
- Track consent status per identity/source
- Consent expiry checking
- Consent withdrawal handling
- Consent proof storage

#### Step 8.3: Implement Data Retention (`marcus_core/compliance/data_retention.py`)
- Configurable retention periods per data type
- Automatic purging of expired data
- Retention policy enforcement

#### Step 8.4: Implement Erasure Handler (`marcus_core/compliance/erasure_handler.py`)
- Right-to-erasure request processing
- Complete data deletion (embeddings, images, metadata)
- Cascade deletion across all stores
- Deletion confirmation and audit trail

---

### Phase 9: Unified Pipeline

#### Step 9.1: Implement Facial Pipeline (`marcus_core/pipeline/facial_pipeline.py`)
```python
# FacialPipeline class combining all modules:
# 
# __init__(config: SystemConfig)
#   - Initialize detector, embedder, vectordb, matcher
#   - Set up compliance logging
# 
# detect_and_embed(image) -> List[FaceResult]
#   - Run detection
#   - Align faces
#   - Extract embeddings
#   - Return faces with embeddings
# 
# search(image, top_k=10) -> List[MatchResult]
#   - Detect and embed
#   - Query vector database
#   - Rank and filter results
#   - Log access for compliance
# 
# enroll(image, identity_metadata) -> str
#   - Verify consent
#   - Detect and embed
#   - Add to vector database
#   - Log enrollment
#   - Return identity ID
# 
# delete_identity(identity_id) -> bool
#   - Remove from vector database
#   - Delete associated data
#   - Log deletion
```

---

### Phase 10: Test UI Application

#### Step 10.1: Create Main App (`marcus_ui/app.py`)
```python
# Streamlit multi-page app setup
# - Initialize pipeline on startup
# - Sidebar navigation
# - System status display
```

#### Step 10.2: Live Detection Page (`marcus_ui/pages/1_live_detection.py`)
- Webcam capture using streamlit-webrtc
- Real-time face detection overlay
- Live matching against database
- Display match results with identity cards

#### Step 10.3: Photo Search Page (`marcus_ui/pages/2_photo_search.py`)
- Image upload (file or URL)
- Face detection with bounding boxes
- Search results with similarity scores
- Detailed identity information display

#### Step 10.4: Enroll Identity Page (`marcus_ui/pages/3_enroll_identity.py`)
- Upload reference photos
- Enter identity metadata (name, source, notes)
- Consent acknowledgment checkbox
- Multi-photo enrollment for robustness

#### Step 10.5: Database Management Page (`marcus_ui/pages/4_manage_database.py`)
- List all enrolled identities
- View identity details and embeddings
- Delete identities (with confirmation)
- Export/import database

#### Step 10.6: Training Page (`marcus_ui/pages/5_training.py`)
- Upload training dataset
- Configure training parameters
- Run fine-tuning with progress
- Evaluate model performance

---

### Phase 11: Testing & Documentation

#### Step 11.1: Write Unit Tests
- Test each module independently
- Mock external dependencies
- Aim for >80% coverage

#### Step 11.2: Write Integration Tests
- Test full pipeline end-to-end
- Test UI interactions
- Performance benchmarks

#### Step 11.3: Create User README
- Installation instructions
- Quick start guide
- Configuration reference
- API documentation

---

## Implementation Order (Recommended)

```
Week 1: Foundation
├── Step 1.1-1.4: Project structure and configuration
├── Step 2.1-2.3: Detection module
└── Step 3.1-3.4: Embedding module

Week 2: Core Pipeline
├── Step 4.1-4.3: Vector database module
├── Step 5.1-5.3: Matching module
└── Step 9.1: Unified pipeline

Week 3: Compliance & OSINT
├── Step 8.1-8.4: Compliance module
└── Step 7.1-7.5: OSINT ingestion

Week 4: Training & UI
├── Step 6.1-6.4: Training module
├── Step 10.1-10.6: Streamlit UI
└── Step 11.1-11.3: Testing & docs
```

---

## Key Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Detection | YOLOv8-Face | Best speed/accuracy trade-off, active development |
| Embeddings | ArcFace-R100 | State-of-the-art accuracy, well-supported |
| Vector DB | FAISS (primary) | Mature, fast, GPU support, scales to millions |
| UI Framework | Streamlit | Rapid development, webcam support, Python-native |
| Compliance | Custom logging | Full control over UK GDPR requirements |
| Training | PyTorch | Flexibility, large ecosystem, metric learning support |

---

## Configuration Example (`configs/default.yaml`)

```yaml
system:
  project_name: "MARCUS"
  debug: false
  device: "auto"  # auto, cuda, mps, cpu

detection:
  model: "yolov8n-face"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  min_face_size: 20

embedding:
  model: "arcface"
  backbone: "r100"
  embedding_dim: 512
  normalize: true

vectordb:
  type: "faiss"
  index_type: "IVF_Flat"
  metric: "cosine"
  dimension: 512
  persist_path: "./data/vector_db"

matching:
  similarity_threshold: 0.6
  top_k: 10

training:
  loss: "arcface"
  margin: 0.5
  scale: 64.0
  learning_rate: 0.0001
  batch_size: 32

osint:
  sources:
    - linkedin
    - public_images
  rate_limit_rpm: 30
  cache_dir: "./data/osint_cache"

compliance:
  enabled: true
  log_all_access: true
  log_retention_days: 365
  require_consent: true
  audit_log_path: "./data/audit_logs"
```

---

## Notes for AI Implementation

2. **Log all operations** through the compliance module
3. **Use type hints** throughout for clarity
4. **Handle errors gracefully** with informative messages
5. **Support both sync and async** operations where beneficial
6. **Make modules independently testable** with dependency injection
7. **Use environment variables** for sensitive configuration (API keys, etc.)
8. **Validate all inputs** before processing
9. **Provide progress callbacks** for long-running operations
10. **Document public APIs** with docstrings

---

## Getting Started (After Implementation)

```bash
# Install the package
pip install -e .

# Or with UI dependencies
pip install -e ".[ui]"

# Run the UI
streamlit run marcus_ui/app.py

# Use as library
from marcus_core import FacialPipeline
pipeline = FacialPipeline.from_config("configs/default.yaml")
results = pipeline.search(image)
```
