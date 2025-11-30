"""
UI Components
=============

Reusable Streamlit components for Marcus UI.
"""

import streamlit as st
from typing import List, Optional, Dict, Any
import numpy as np


def match_card(
    name: str,
    score: float,
    identity_id: str,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Display a match result card.
    
    Args:
        name: Identity name
        score: Match score (0-1)
        identity_id: Identity ID
        source: Data source
        metadata: Additional metadata
    """
    # Determine color based on score
    if score >= 0.8:
        color = "#4CAF50"
        bg_color = "#E8F5E9"
        quality = "High Confidence"
    elif score >= 0.6:
        color = "#FFC107"
        bg_color = "#FFF8E1"
        quality = "Medium Confidence"
    else:
        color = "#F44336"
        bg_color = "#FFEBEE"
        quality = "Low Confidence"
    
    st.markdown(f"""
    <div style="
        border: 2px solid {color};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: {bg_color};
    ">
        <h4 style="margin: 0; color: {color};">{name}</h4>
        <p style="margin: 0.5rem 0 0 0;">
            <strong>Score:</strong> {score:.1%} ({quality})<br>
            <strong>Source:</strong> {source or 'N/A'}<br>
            <strong>ID:</strong> <code>{identity_id[:12]}...</code>
        </p>
    </div>
    """, unsafe_allow_html=True)


def face_detection_box(
    image: np.ndarray,
    bbox: tuple,
    label: Optional[str] = None,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """
    Draw detection box on image.
    
    Args:
        image: RGB image
        bbox: Bounding box (x1, y1, x2, y2)
        label: Optional label
        confidence: Detection confidence
    
    Returns:
        Image with detection drawn
    """
    try:
        from marcus_core.utils.image_utils import draw_detection
        return draw_detection(
            image=image,
            bbox=bbox,
            label=label,
            confidence=confidence,
        )
    except ImportError:
        return image


def metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
) -> None:
    """
    Display a metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change indicator
        help_text: Help text
    """
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    ">
        <p style="margin: 0; color: #666; font-size: 0.9rem;">{label}</p>
        <h2 style="margin: 0.5rem 0; color: #1E88E5;">{value}</h2>
        {f'<p style="margin: 0; color: #4CAF50;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)
    
    if help_text:
        st.caption(help_text)


def status_indicator(
    status: str,
    message: Optional[str] = None,
) -> None:
    """
    Display a status indicator.
    
    Args:
        status: Status type (success, warning, error, info)
        message: Status message
    """
    colors = {
        "success": ("🟢", "#4CAF50"),
        "warning": ("🟡", "#FFC107"),
        "error": ("🔴", "#F44336"),
        "info": ("🔵", "#2196F3"),
    }
    
    icon, color = colors.get(status, ("⚪", "#9E9E9E"))
    
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-left: 3px solid {color};
    ">
        <span style="font-size: 1.2rem;">{icon}</span>
        <span>{message or status.capitalize()}</span>
    </div>
    """, unsafe_allow_html=True)


def image_grid(
    images: List[np.ndarray],
    captions: Optional[List[str]] = None,
    columns: int = 4,
) -> None:
    """
    Display images in a grid.
    
    Args:
        images: List of images
        captions: Optional captions
        columns: Number of columns
    """
    cols = st.columns(columns)
    
    for i, img in enumerate(images):
        with cols[i % columns]:
            caption = captions[i] if captions and i < len(captions) else None
            st.image(img, caption=caption, use_container_width=True)


def confidence_bar(
    score: float,
    label: Optional[str] = None,
) -> None:
    """
    Display a confidence bar.
    
    Args:
        score: Confidence score (0-1)
        label: Optional label
    """
    # Determine color
    if score >= 0.8:
        color = "#4CAF50"
    elif score >= 0.6:
        color = "#FFC107"
    else:
        color = "#F44336"
    
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        {f'<p style="margin: 0 0 0.25rem 0;">{label}</p>' if label else ''}
        <div style="
            background-color: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        ">
            <div style="
                width: {score * 100}%;
                height: 20px;
                background-color: {color};
                border-radius: 5px;
                text-align: center;
                color: white;
                font-size: 0.8rem;
                line-height: 20px;
            ">
                {score:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
