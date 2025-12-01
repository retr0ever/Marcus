# Marcus 👁️

**OSINT-enabled facial analysis system with real-time detection, embedding extraction, and identity matching.**

Marcus is a modular face recognition library designed for research and development purposes. It combines state-of-the-art face detection (YOLOv8-Face) with robust embedding extraction (ArcFace) and efficient vector search (FAISS) to enable face identification against known identities.

## Ethical Use Notice

This software is intended for legitimate research and development purposes only. Users must:
- Comply with all applicable laws including UK GDPR
- Obtain proper consent before processing biometric data
- Respect individuals' privacy rights
- Not use for surveillance without lawful authority

## Features

- **Face Detection**: YOLOv8-Face with RetinaFace fallback
- **Embedding Extraction**: ArcFace (R100/R50/MobileFaceNet) with 512-dim vectors
- **Vector Database**: FAISS with Flat/IVF/HNSW index types
- **Identity Matching**: Cosine similarity with result re-ranking
- **Compliance**: UK GDPR with audit logging and consent management
- **Continual Learning**: Hard example mining and experience replay
- **User Interface**: Streamlit-based testing interface

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/retr0ever/Marcus.git
cd Marcus

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core library
pip install -e .

# Install with UI support
pip install -e ".[ui]"

# Install with GPU support
pip install -e ".[gpu]"

# Install all (development)
pip install -e ".[dev]"
```

### Basic Usage

```python
from marcus_core import FacialPipeline, SystemConfig

# Initialize with default config
config = SystemConfig()
pipeline = FacialPipeline(config)

# Or from YAML config
pipeline = FacialPipeline.from_config("configs/default.yaml")

# Enroll an identity
identity_id = pipeline.enroll(
    image=image_array,
    name="John Doe",
    source="manual",
)

# Search for faces
results = pipeline.search(query_image)
for face in results:
    for match in face.matches:
        print(f"Match: {match.identity.name} ({match.score:.2%})")

# Verify two images
verification = pipeline.verify(image1, image2)
print(f"Same person: {verification['is_match']}")
```

### Running the UI

```bash
`streamlit run marcus_ui/app.py
````

## 📁 Project Structure

```
marcus-codebase/
├── marcus_core/           # Core library
│   ├── detection/         # Face detection (YOLO, RetinaFace)
│   ├── embedding/         # Embedding extraction (ArcFace)
│   ├── vectordb/          # Vector database (FAISS)
│   ├── matching/          # Identity matching and ranking
│   ├── compliance/        # GDPR compliance (audit, consent)
│   ├── osint/             # OSINT data collection
│   ├── training/          # Continual learning
│   ├── pipeline/          # Unified pipeline
│   ├── utils/             # Utilities
│   └── config.py          # Configuration
├── marcus_ui/             # Streamlit UI
│   ├── app.py             # Main application
│   └── pages/             # UI pages
├── configs/               # Configuration files
├── data/                  # Data directory (gitignored)
├── models/                # Model weights (gitignored)
└── tests/                 # Test suite
```

## Configuration

Marcus uses YAML configuration files. See `configs/default.yaml` for all options.

Key settings:

```yaml
detection:
  model: "yolov8n-face"
  confidence_threshold: 0.5

embedding:
  backbone: "r100"
  normalize: true

matching:
  similarity_threshold: 0.6
  top_k: 10

compliance:
  enabled: true
  require_consent: true
```

## API Reference

### FacialPipeline

The main entry point for the system.

```python
pipeline = FacialPipeline(config)

# Detection only
detections = pipeline.detect(image)

# Detection + Embedding
face_results = pipeline.detect_and_embed(image)

# Full search
results = pipeline.search(image, top_k=5, threshold=0.6)

# Enrollment
identity_id = pipeline.enroll(image, name="Name", metadata={...})
identity_id = pipeline.enroll_multiple(images, name="Name")

# Verification
result = pipeline.verify(image1, image2)

# Management
pipeline.delete_identity(identity_id, reason="user_request")
identities = pipeline.list_identities(query="search term")
```

### Compliance

```python
from marcus_core.compliance import AuditLogger, ConsentManager

# Audit logging
logger = AuditLogger(log_path="logs/audit")
logger.log_search(user_id="user123", query_hash="...", results_count=5)

# Consent management
consent_mgr = ConsentManager(persist_path="data/consent")
consent_mgr.grant_consent(identity_id="...", granted_by="admin")
is_valid = consent_mgr.verify_consent(identity_id)
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=marcus_core --cov-report=html
```

## Performance

| Component | Speed | Hardware |
|-----------|-------|----------|
| YOLOv8n-Face | ~30ms/image | GPU (RTX 3080) |
| ArcFace R100 | ~10ms/face | GPU |
| FAISS HNSW Search | <1ms/query | CPU (10K vectors) |

## Security & Compliance

Marcus includes built-in compliance features for UK GDPR:

- **Audit Logging**: All searches, enrollments, and deletions are logged
- **Consent Management**: Track and verify consent for each identity
- **Right to Erasure**: Complete deletion of identity data
- **Data Retention**: Configurable retention policies
- **Access Control**: User context tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [InsightFace](https://github.com/deepinsight/insightface) for ArcFace
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io/) for the UI framework

---

<p align="center">
  <a href="https://github.com/retr0ever/Marcus">GitHub Repository</a>
</p>
