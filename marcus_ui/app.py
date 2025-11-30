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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: normal;
        color: #333;
        text-align: left;
        margin-bottom: 1.5rem;
        font-family: 'Times New Roman', Times, serif;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: left;
        margin-bottom: 1.5rem;
        font-family: 'Times New Roman', Times, serif;
    }
    .metric-card {
        background-colour: #fafafa;
        border: 1px solid #ddd;
        padding: 1rem;
        text-align: left;
        font-family: 'Times New Roman', Times, serif;
    }
    .match-card {
        border: 1px solid #999;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Times New Roman', Times, serif;
    }
    .match-high {
        border-color: #333;
        background-colour: #f5f5f5;
    }
    .match-medium {
        border-color: #666;
        background-colour: #fafafa;
    }
    .match-low {
        border-color: #999;
        background-colour: #fff;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    .stMarkdown, .stText {
        font-family: 'Times New Roman', Times, serif !important;
    }
</style>
""", unsafe_allow_html=True)


# Navigation
PAGES = {
    "Home": home,
    "Live Detection": live_detection,
    "Photo Search": photo_search,
    "Enroll Identity": enrollment,
    "Settings": settings,
}


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("Marcus")
    st.sidebar.markdown("---")
    
    selection = st.sidebar.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    
    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Check if pipeline is initialised
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
        st.sidebar.warning("Pipeline: Not initialised")
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
