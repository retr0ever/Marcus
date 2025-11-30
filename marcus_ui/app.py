"""
Marcus Face Analysis UI
========================

Streamlit application for testing the Marcus face analysis system.

Run with:
    streamlit run marcus_ui/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from marcus_ui.pages import home, live_detection, photo_search, enrollment, settings


# Page configuration
st.set_page_config(
    page_title="Marcus - Face Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .match-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .match-high {
        border-color: #4CAF50;
        background-color: #E8F5E9;
    }
    .match-medium {
        border-color: #FFC107;
        background-color: #FFF8E1;
    }
    .match-low {
        border-color: #F44336;
        background-color: #FFEBEE;
    }
</style>
""", unsafe_allow_html=True)


# Navigation
PAGES = {
    "🏠 Home": home,
    "📹 Live Detection": live_detection,
    "🔍 Photo Search": photo_search,
    "➕ Enroll Identity": enrollment,
    "⚙️ Settings": settings,
}


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("👁️ Marcus")
    st.sidebar.markdown("---")
    
    selection = st.sidebar.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    
    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Check if pipeline is initialized
    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        st.sidebar.success("Pipeline: Ready")
        
        # Show statistics
        try:
            stats = st.session_state.pipeline.get_statistics()
            st.sidebar.metric("Identities", stats.get("total_identities", 0))
            st.sidebar.metric("Embeddings", stats.get("total_embeddings", 0))
        except Exception:
            pass
    else:
        st.sidebar.warning("Pipeline: Not initialized")
        st.sidebar.info("Go to Settings to configure")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>Marcus v0.1.0 | "
        "[GitHub](https://github.com/retr0ever/Marcus)</small>",
        unsafe_allow_html=True,
    )
    
    # Render selected page
    page = PAGES[selection]
    page.render()


if __name__ == "__main__":
    main()
