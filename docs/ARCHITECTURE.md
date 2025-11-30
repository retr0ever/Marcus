# Marcus Architecture

## Overview

Marcus is a modular facial analysis system that detects faces, extracts embeddings, and matches identities against a vector database.

```
Image → Detection → Embedding → Vector Search → Match Results
```

---

## Core Pipeline

| Stage | Technology | Purpose |
|-------|------------|---------|
| Detection | YOLOv8-Face | Locate faces in images (fast, accurate) |
| Fallback | RetinaFace | Secondary detector for edge cases |
| Alignment | 5-point landmarks | Normalise face orientation |
| Embedding | ArcFace (InsightFace) | Extract 512-dimensional identity vectors |
| Storage | FAISS | Vector database with similarity search |
| Matching | Cosine similarity | Compare embeddings and rank results |

---

## Technology Stack

**Core:**
- Python 3.10+
- PyTorch 2.0+ (MPS/CUDA/CPU)
- Pydantic v2 (configuration)

**Detection:**
- Ultralytics (YOLOv8)
- InsightFace (RetinaFace, alignment)

**Embeddings:**
- ArcFace-R100 (highest accuracy)
- ArcFace-R50 (balanced)
- MobileFaceNet (fastest)

**Vector Database:**
- FAISS Flat (exact search)
- FAISS IVF (approximate, scalable)
- FAISS HNSW (graph-based, fast)

**UI:**
- Streamlit
- streamlit-webrtc (webcam)

---

## Module Structure

```
marcus_core/
├── detection/      # Face detection and alignment
├── embedding/      # Feature extraction
├── vectordb/       # FAISS vector storage
├── matching/       # Identity matching logic
├── compliance/     # GDPR audit and consent
├── pipeline/       # Unified FacialPipeline class
├── osint/          # External data adapters
├── training/       # Continual learning
└── utils/          # Device management, image tools
```

---

## How Matching Works

1. **Detect**: Find all faces in the input image
2. **Align**: Normalise each face using landmark points
3. **Embed**: Extract a 512-dim vector per face
4. **Search**: Query FAISS for nearest neighbours
5. **Rank**: Sort by cosine similarity, apply threshold
6. **Return**: Top-K matches with scores and metadata

---

## Scalability

| Scale | Index Type | Identities | Search Time |
|-------|------------|------------|-------------|
| Small | Flat | < 10,000 | < 10ms |
| Medium | IVF_Flat | < 1,000,000 | < 50ms |
| Large | HNSW | 10,000,000+ | < 100ms |

---

## Growth Paths

### Accuracy
- Fine-tune embeddings on domain-specific data
- Enable continual learning with hard example mining
- Add quality filtering (blur, occlusion, pose)

### Scale
- Switch to IVF or HNSW indexing
- Shard database across multiple FAISS indices
- Add Redis/PostgreSQL for metadata

### Features
- OSINT adapters for external sources
- Video stream processing
- Multi-face tracking
- Age/gender/emotion estimation

### Deployment
- Dockerise with GPU support
- REST API with FastAPI
- Kubernetes for horizontal scaling
- Edge deployment with ONNX/TensorRT

---

## Compliance

- UK GDPR compliant by design
- Consent management per identity
- Full audit logging of all operations
- Data retention policies
- Right to erasure support
