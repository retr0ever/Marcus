"""
Photo Search Page
=================

Search for identities by uploading a photo.
"""

import streamlit as st
import numpy as np
from PIL import Image


def render():
    """Render the photo search page."""
    
    st.title("🔍 Photo Search")
    st.markdown(
        "Upload a photo to search for matching identities in the database."
    )
    
    # Check if pipeline is initialized
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        st.warning("⚠️ Pipeline not initialized. Please go to Settings first.")
        return
    
    pipeline = st.session_state.pipeline
    
    st.markdown("---")
    
    # Search settings
    col1, col2 = st.columns([1, 1])
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.6,
            step=0.05,
            help="Minimum similarity for identity matching",
        )
    
    with col2:
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of matches to return",
        )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a photo containing one or more faces",
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Results")
            
            # Process image
            with st.spinner("Searching..."):
                try:
                    results = pipeline.search(
                        image_np,
                        top_k=top_k,
                        threshold=similarity_threshold,
                    )
                    
                    if not results:
                        st.warning("No faces detected in the image")
                    else:
                        st.success(f"Detected {len(results)} face(s)")
                        
                        for i, face_result in enumerate(results):
                            with st.expander(
                                f"Face {i+1} - {len(face_result.matches)} matches",
                                expanded=True,
                            ):
                                # Detection info
                                det = face_result.detection
                                st.write(f"**Detection confidence:** {det.confidence:.2%}")
                                st.write(f"**Quality score:** {face_result.quality_score:.2%}")
                                
                                # Matches
                                if face_result.matches:
                                    st.markdown("---")
                                    st.markdown("**Matches:**")
                                    
                                    for match in face_result.matches:
                                        score = match.score
                                        identity = match.identity
                                        
                                        # Determine match quality
                                        if score >= 0.8:
                                            css_class = "match-high"
                                            quality = "High"
                                        elif score >= 0.6:
                                            css_class = "match-medium"
                                            quality = "Medium"
                                        else:
                                            css_class = "match-low"
                                            quality = "Low"
                                        
                                        name = identity.name or "Unknown"
                                        source = identity.source or "N/A"
                                        
                                        st.markdown(f"""
                                        <div class="match-card {css_class}">
                                            <strong>{name}</strong><br>
                                            Score: {score:.2%} ({quality})<br>
                                            Source: {source}<br>
                                            ID: <code>{identity.id[:8]}...</code>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No matches found above threshold")
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Batch upload option
    st.markdown("---")
    st.subheader("Batch Search")
    
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple photos for batch searching",
    )
    
    if batch_files:
        if st.button("Search All", type="primary"):
            progress = st.progress(0)
            results_container = st.container()
            
            for i, file in enumerate(batch_files):
                progress.progress((i + 1) / len(batch_files))
                
                try:
                    image = Image.open(file)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image_np = np.array(image)
                    
                    results = pipeline.search(
                        image_np,
                        top_k=top_k,
                        threshold=similarity_threshold,
                    )
                    
                    with results_container:
                        with st.expander(f"📷 {file.name}"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.image(image, width=150)
                            
                            with col2:
                                if results and results[0].matches:
                                    for match in results[0].matches[:3]:
                                        name = match.identity.name or "Unknown"
                                        st.write(
                                            f"• {name}: {match.score:.2%}"
                                        )
                                else:
                                    st.write("No matches found")
                
                except Exception as e:
                    with results_container:
                        st.error(f"Error processing {file.name}: {e}")
            
            progress.empty()
    
    # Tips
    st.markdown("---")
    st.subheader("Tips for Better Results")
    
    st.markdown("""
    - Use clear, well-lit photos
    - Ensure faces are visible and not obstructed
    - Higher resolution images may yield better results
    - Front-facing photos work best
    - Avoid extreme angles or expressions
    """)
