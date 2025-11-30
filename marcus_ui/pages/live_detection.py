"""
Live Detection Page
===================

Real-time face detection and matching using webcam.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io


def render():
    """Render the live detection page."""
    
    st.title("📹 Live Detection")
    st.markdown(
        "Detect and identify faces in real-time using your webcam."
    )
    
    # Check if pipeline is initialized
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        st.warning("⚠️ Pipeline not initialized. Please go to Settings first.")
        return
    
    pipeline = st.session_state.pipeline
    
    st.markdown("---")
    
    # Settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Minimum confidence for face detection",
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.6,
            step=0.05,
            help="Minimum similarity for identity matching",
        )
        
        top_k = st.slider(
            "Top K Matches",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of top matches to display",
        )
    
    with col2:
        st.subheader("Display Options")
        
        show_landmarks = st.checkbox("Show Landmarks", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)
        show_bbox = st.checkbox("Show Bounding Box", value=True)
    
    st.markdown("---")
    
    # Camera input
    st.subheader("Camera Feed")
    
    # Use Streamlit's camera input
    camera_image = st.camera_input(
        "Take a photo",
        help="Click to capture an image for face detection",
    )
    
    if camera_image is not None:
        # Process the captured image
        image = Image.open(camera_image)
        image_np = np.array(image)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)
        
        with col2:
            # Run detection and matching
            with st.spinner("Processing..."):
                try:
                    results = pipeline.search(
                        image_np,
                        top_k=top_k,
                        threshold=similarity_threshold,
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} face(s)")
                        
                        for i, face_result in enumerate(results):
                            st.markdown(f"**Face {i+1}**")
                            
                            # Show detection info
                            det = face_result.detection
                            if show_confidence:
                                st.write(f"Detection confidence: {det.confidence:.2%}")
                            
                            # Show matches
                            if face_result.matches:
                                st.markdown("**Matches:**")
                                for j, match in enumerate(face_result.matches):
                                    score = match.score
                                    
                                    # Color based on score
                                    if score >= 0.8:
                                        color = "🟢"
                                    elif score >= 0.6:
                                        color = "🟡"
                                    else:
                                        color = "🔴"
                                    
                                    name = match.identity.name or "Unknown"
                                    st.write(
                                        f"{color} {j+1}. {name}: {score:.2%}"
                                    )
                            else:
                                st.info("No matches found")
                            
                            st.markdown("---")
                    else:
                        st.warning("No faces detected in the image")
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")
    
    # Instructions
    st.markdown("---")
    st.subheader("Instructions")
    
    st.markdown("""
    1. Allow camera access when prompted
    2. Position your face in the camera view
    3. Click the capture button to take a photo
    4. The system will detect faces and search for matches
    
    **Tips:**
    - Ensure good lighting
    - Face the camera directly
    - Remove obstructions (sunglasses, masks)
    """)
    
    # Note about live streaming
    st.info(
        "💡 For true real-time streaming, consider using streamlit-webrtc. "
        "This implementation uses snapshot-based detection."
    )
