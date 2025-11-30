"""
Home Page
=========

Landing page with system overview.
"""

import streamlit as st


def render():
    """Render the home page."""
    
    st.markdown(
        '<h1 class="main-header">👁️ Marcus Face Analysis</h1>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        '<p class="sub-header">'
        'OSINT-enabled facial analysis system with real-time detection, '
        'embedding extraction, and identity matching.'
        '</p>',
        unsafe_allow_html=True,
    )
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📹 Live Detection</h3>
            <p>Real-time face detection using your webcam with 
            instant identity matching.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Photo Search</h3>
            <p>Upload a photo to search for matching identities 
            in the database.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>➕ Enrollment</h3>
            <p>Enroll new identities with one or more reference 
            photos for future matching.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Quick Start")
    
    st.markdown("""
    1. **Initialize the Pipeline**: Go to ⚙️ Settings and configure your pipeline.
    2. **Enroll Identities**: Add known faces using the ➕ Enroll Identity page.
    3. **Search**: Use 📹 Live Detection or 🔍 Photo Search to find matches.
    """)
    
    # System capabilities
    st.subheader("✨ Capabilities")
    
    capabilities = [
        ("Face Detection", "YOLOv8-Face with RetinaFace fallback", "✅"),
        ("Embedding Extraction", "ArcFace R100/R50/MobileFaceNet", "✅"),
        ("Vector Database", "FAISS with HNSW/IVF indexing", "✅"),
        ("Identity Matching", "Cosine similarity with re-ranking", "✅"),
        ("Compliance", "UK GDPR with audit logging", "✅"),
        ("Continual Learning", "Hard example mining and replay", "✅"),
    ]
    
    for name, desc, status in capabilities:
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.write(f"**{name}**")
        with col2:
            st.write(desc)
        with col3:
            st.write(status)
    
    st.markdown("---")
    
    # Performance metrics (if pipeline is available)
    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        st.subheader("📊 Current Statistics")
        
        try:
            stats = st.session_state.pipeline.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Identities",
                    stats.get("total_identities", 0),
                )
            
            with col2:
                st.metric(
                    "Total Embeddings",
                    stats.get("total_embeddings", 0),
                )
            
            with col3:
                consent_stats = stats.get("consent_stats", {})
                st.metric(
                    "Active Consents",
                    consent_stats.get("active_consents", 0),
                )
            
            with col4:
                st.metric(
                    "Similarity Threshold",
                    f"{stats.get('config', {}).get('similarity_threshold', 0.6):.0%}",
                )
        
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    else:
        st.info("💡 Initialize the pipeline in Settings to see statistics.")
