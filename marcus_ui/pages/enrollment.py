"""
Enrollment Page
===============

Enroll new identities into the system.
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime


def render():
    """Render the enrollment page."""
    
    st.title("➕ Enroll Identity")
    st.markdown(
        "Add new identities to the database for future matching."
    )
    
    # Check if pipeline is initialized
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        st.warning("⚠️ Pipeline not initialized. Please go to Settings first.")
        return
    
    pipeline = st.session_state.pipeline
    
    st.markdown("---")
    
    # Enrollment form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Identity Information")
        
        name = st.text_input(
            "Name",
            placeholder="John Doe",
            help="Name of the person to enroll",
        )
        
        source = st.selectbox(
            "Data Source",
            options=["manual", "public", "consented", "dataset"],
            index=0,
            help="Source of the identity data",
        )
        
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Additional notes about this identity...",
            help="Any additional metadata",
        )
        
        # Consent checkbox
        consent = st.checkbox(
            "I confirm I have the right to enroll this person's biometric data",
            help="Required for GDPR compliance",
        )
    
    with col2:
        st.subheader("Reference Photos")
        
        uploaded_files = st.file_uploader(
            "Upload photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload one or more clear photos of the person's face",
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} photo(s)")
            
            # Preview images
            cols = st.columns(min(len(uploaded_files), 4))
            for i, file in enumerate(uploaded_files[:4]):
                with cols[i]:
                    img = Image.open(file)
                    st.image(img, width=100)
    
    st.markdown("---")
    
    # Enrollment button
    if st.button("Enroll Identity", type="primary", disabled=not consent):
        if not name:
            st.error("Please provide a name")
        elif not uploaded_files:
            st.error("Please upload at least one photo")
        else:
            with st.spinner("Enrolling identity..."):
                try:
                    # Load images
                    images = []
                    for file in uploaded_files:
                        file.seek(0)  # Reset file pointer
                        img = Image.open(file)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        images.append(np.array(img))
                    
                    # Build metadata
                    metadata = {
                        "notes": notes,
                        "enrolled_at": datetime.now().isoformat(),
                        "num_photos": len(images),
                    }
                    
                    # Enroll
                    if len(images) == 1:
                        identity_id = pipeline.enroll(
                            image=images[0],
                            name=name,
                            metadata=metadata,
                            source=source,
                        )
                    else:
                        identity_id = pipeline.enroll_multiple(
                            images=images,
                            name=name,
                            metadata=metadata,
                            source=source,
                        )
                    
                    if identity_id:
                        st.success(f"✅ Successfully enrolled: {name}")
                        st.info(f"Identity ID: `{identity_id}`")
                        
                        # Clear form (by rerunning)
                        if st.button("Enroll Another"):
                            st.rerun()
                    else:
                        st.error(
                            "Failed to enroll identity. "
                            "No face detected in the uploaded photos."
                        )
                
                except Exception as e:
                    st.error(f"Error during enrollment: {e}")
    
    st.markdown("---")
    
    # View enrolled identities
    st.subheader("Enrolled Identities")
    
    search_query = st.text_input(
        "Search identities",
        placeholder="Search by name...",
    )
    
    try:
        identities = pipeline.list_identities(query=search_query or None)
        
        if identities:
            st.write(f"Found {len(identities)} identit(ies)")
            
            for identity in identities[:20]:  # Limit display
                with st.expander(f"👤 {identity.name or 'Unknown'}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**ID:** `{identity.id}`")
                        st.write(f"**Source:** {identity.source}")
                        st.write(f"**Enrolled:** {identity.created_at.strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Embeddings:** {len(identity.embeddings)}")
                        
                        if identity.metadata:
                            st.write("**Metadata:**")
                            st.json(identity.metadata)
                    
                    with col2:
                        if st.button(
                            "🗑️ Delete",
                            key=f"delete_{identity.id}",
                            help="Delete this identity",
                        ):
                            if st.session_state.get(f"confirm_delete_{identity.id}"):
                                pipeline.delete_identity(
                                    identity.id,
                                    reason="user_request",
                                )
                                st.success("Identity deleted")
                                st.rerun()
                            else:
                                st.session_state[f"confirm_delete_{identity.id}"] = True
                                st.warning("Click again to confirm deletion")
        else:
            st.info("No identities enrolled yet")
    
    except Exception as e:
        st.error(f"Error loading identities: {e}")
    
    # Instructions
    st.markdown("---")
    st.subheader("Guidelines")
    
    st.markdown("""
    **Photo Requirements:**
    - Clear, front-facing photos
    - Good lighting
    - Face clearly visible
    - Multiple photos from different angles improve matching
    
    **GDPR Compliance:**
    - Only enroll individuals with proper consent
    - Data is stored locally with audit logging
    - Identities can be deleted upon request (right to erasure)
    """)
