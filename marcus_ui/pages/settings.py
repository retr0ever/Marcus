"""
Settings Page
=============

Configure and manage the Marcus system.
"""

import streamlit as st
from pathlib import Path
import yaml


def render():
    """Render the settings page."""
    
    st.title("⚙️ Settings")
    st.markdown(
        "Configure the Marcus face analysis system."
    )
    
    st.markdown("---")
    
    # Pipeline initialization
    st.subheader("🚀 Pipeline Initialization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        config_source = st.radio(
            "Configuration Source",
            options=["Default", "Custom Config File", "Manual"],
            horizontal=True,
        )
        
        if config_source == "Custom Config File":
            config_path = st.text_input(
                "Config File Path",
                placeholder="/path/to/config.yaml",
                help="Path to YAML configuration file",
            )
        else:
            config_path = None
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        
        if st.button("Initialize Pipeline", type="primary"):
            with st.spinner("Initializing pipeline..."):
                try:
                    # Import here to avoid circular imports
                    from marcus_core.config import SystemConfig
                    from marcus_core.pipeline import FacialPipeline
                    
                    if config_path and Path(config_path).exists():
                        pipeline = FacialPipeline.from_config(config_path)
                    else:
                        # Create default config
                        config = SystemConfig()
                        pipeline = FacialPipeline(config)
                    
                    # Store in session state
                    st.session_state.pipeline = pipeline
                    st.session_state.config = config if not config_path else None
                    
                    st.success("✅ Pipeline initialized successfully!")
                    
                    # Show info
                    stats = pipeline.get_statistics()
                    st.json(stats)
                
                except Exception as e:
                    st.error(f"Failed to initialize pipeline: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Manual configuration
    if config_source == "Manual":
        st.subheader("📝 Manual Configuration")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Detection",
            "Embedding",
            "Matching",
            "Compliance",
        ])
        
        with tab1:
            st.markdown("**Detection Settings**")
            
            det_model = st.selectbox(
                "Model",
                options=["yolov8n-face", "yolov8s-face", "retinaface"],
                index=0,
            )
            
            det_confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
            )
            
            det_device = st.selectbox(
                "Device",
                options=["auto", "cuda", "mps", "cpu"],
                index=0,
            )
        
        with tab2:
            st.markdown("**Embedding Settings**")
            
            emb_backbone = st.selectbox(
                "Backbone",
                options=["r100", "r50", "mobilefacenet"],
                index=0,
            )
            
            emb_normalize = st.checkbox("Normalize Embeddings", value=True)
            emb_fp16 = st.checkbox("Use FP16", value=False)
        
        with tab3:
            st.markdown("**Matching Settings**")
            
            match_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.3,
                max_value=0.95,
                value=0.6,
                step=0.05,
            )
            
            match_top_k = st.slider(
                "Top K Results",
                min_value=1,
                max_value=50,
                value=10,
            )
            
            match_algorithm = st.selectbox(
                "Matching Algorithm",
                options=["cosine", "euclidean"],
                index=0,
            )
        
        with tab4:
            st.markdown("**Compliance Settings**")
            
            comp_enabled = st.checkbox("Enable Compliance", value=True)
            comp_require_consent = st.checkbox("Require Consent", value=True)
            comp_retention_days = st.number_input(
                "Log Retention (days)",
                min_value=1,
                max_value=365,
                value=90,
            )
        
        # Save configuration
        if st.button("Save Configuration"):
            config = {
                "detection": {
                    "model": det_model,
                    "confidence_threshold": det_confidence,
                    "device": det_device,
                },
                "embedding": {
                    "backbone": emb_backbone,
                    "normalize": emb_normalize,
                    "fp16": emb_fp16,
                },
                "matching": {
                    "similarity_threshold": match_threshold,
                    "top_k": match_top_k,
                    "algorithm": match_algorithm,
                },
                "compliance": {
                    "enabled": comp_enabled,
                    "require_consent": comp_require_consent,
                    "log_retention_days": comp_retention_days,
                },
            }
            
            st.session_state.manual_config = config
            st.success("Configuration saved!")
            st.json(config)
    
    st.markdown("---")
    
    # Current status
    st.subheader("📊 Current Status")
    
    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        pipeline = st.session_state.pipeline
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", "Active", delta="Running")
        
        with col2:
            stats = pipeline.get_statistics()
            st.metric("Identities", stats.get("total_identities", 0))
        
        with col3:
            st.metric("Embeddings", stats.get("total_embeddings", 0))
        
        # Detailed stats
        with st.expander("View Full Statistics"):
            st.json(stats)
        
        # Actions
        st.markdown("**Actions:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Data"):
                try:
                    pipeline.save()
                    st.success("Data saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
        
        with col2:
            if st.button("🔄 Reload Models"):
                try:
                    pipeline.warmup()
                    st.success("Models reloaded!")
                except Exception as e:
                    st.error(f"Failed to reload: {e}")
        
        with col3:
            if st.button("🗑️ Reset Pipeline"):
                if st.session_state.get("confirm_reset"):
                    st.session_state.pipeline = None
                    del st.session_state["confirm_reset"]
                    st.success("Pipeline reset!")
                    st.rerun()
                else:
                    st.session_state["confirm_reset"] = True
                    st.warning("Click again to confirm reset")
    else:
        st.info("Pipeline not initialized. Use the section above to initialize.")
    
    st.markdown("---")
    
    # Export/Import
    st.subheader("📦 Export / Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Configuration**")
        if st.button("Export Config to YAML"):
            if "manual_config" in st.session_state:
                yaml_str = yaml.dump(st.session_state.manual_config, default_flow_style=False)
                st.download_button(
                    "Download YAML",
                    data=yaml_str,
                    file_name="marcus_config.yaml",
                    mime="text/yaml",
                )
            else:
                st.warning("No configuration to export")
    
    with col2:
        st.markdown("**Import Configuration**")
        uploaded_config = st.file_uploader(
            "Upload YAML config",
            type=["yaml", "yml"],
        )
        
        if uploaded_config:
            try:
                config = yaml.safe_load(uploaded_config)
                st.session_state.manual_config = config
                st.success("Configuration imported!")
                st.json(config)
            except Exception as e:
                st.error(f"Failed to parse config: {e}")
    
    # System info
    st.markdown("---")
    st.subheader("💻 System Information")
    
    try:
        from marcus_core.utils.device import get_device_info
        info = get_device_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**PyTorch Available:** {info.get('pytorch_available', False)}")
            st.write(f"**CUDA Available:** {info.get('cuda_available', False)}")
            st.write(f"**MPS Available:** {info.get('mps_available', False)}")
            st.write(f"**Current Device:** {info.get('current_device', 'N/A')}")
        
        with col2:
            if info.get("gpus"):
                for i, gpu in enumerate(info["gpus"]):
                    st.write(f"**GPU {i}:** {gpu.get('name', 'Unknown')}")
                    st.write(f"  Memory: {gpu.get('total_memory_gb', 0):.1f} GB")
    
    except Exception as e:
        st.error(f"Failed to get system info: {e}")
