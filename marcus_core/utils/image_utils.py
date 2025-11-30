"""
Image Utilities
===============

Common image processing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union
import base64
import io
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


logger = logging.getLogger(__name__)


def load_image(
    path: Union[str, Path],
    mode: str = "RGB",
    max_size: Optional[int] = None,
) -> "np.ndarray":
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        mode: Color mode (RGB, BGR, GRAY)
        max_size: Maximum dimension (preserves aspect ratio)
    
    Returns:
        Image as numpy array
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise ImportError("PIL and NumPy required")
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path)
    
    # Convert mode
    if mode == "RGB":
        if img.mode != "RGB":
            img = img.convert("RGB")
    elif mode == "GRAY":
        if img.mode != "L":
            img = img.convert("L")
    elif mode == "BGR":
        if img.mode != "RGB":
            img = img.convert("RGB")
    
    # Resize if needed
    if max_size is not None:
        w, h = img.size
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
    
    # Convert to numpy
    arr = np.array(img)
    
    # Handle BGR
    if mode == "BGR" and len(arr.shape) == 3:
        arr = arr[:, :, ::-1].copy()
    
    return arr


def save_image(
    image: "np.ndarray",
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array (RGB)
        path: Output path
        quality: JPEG quality (1-100)
    """
    if not HAS_PIL:
        raise ImportError("PIL required")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(image)
    
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        img.save(path, quality=quality)
    else:
        img.save(path)


def resize_image(
    image: "np.ndarray",
    size: Tuple[int, int],
    keep_aspect: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> "np.ndarray":
    """
    Resize an image.
    
    Args:
        image: Input image
        size: Target size (width, height)
        keep_aspect: Preserve aspect ratio with padding
        pad_color: Padding color (RGB)
    
    Returns:
        Resized image
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise ImportError("PIL and NumPy required")
    
    img = Image.fromarray(image)
    target_w, target_h = size
    
    if keep_aspect:
        # Calculate scale
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create padded image
        result = Image.new("RGB", (target_w, target_h), pad_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        result.paste(img, (paste_x, paste_y))
        
        return np.array(result)
    else:
        img = img.resize((target_w, target_h), Image.LANCZOS)
        return np.array(img)


def crop_face(
    image: "np.ndarray",
    bbox: Tuple[float, float, float, float],
    margin: float = 0.2,
) -> "np.ndarray":
    """
    Crop a face region from an image.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        margin: Margin to add (fraction of box size)
    
    Returns:
        Cropped face image
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy required")
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add margin
    box_w = x2 - x1
    box_h = y2 - y1
    margin_w = box_w * margin
    margin_h = box_h * margin
    
    x1 = max(0, int(x1 - margin_w))
    y1 = max(0, int(y1 - margin_h))
    x2 = min(w, int(x2 + margin_w))
    y2 = min(h, int(y2 + margin_h))
    
    return image[y1:y2, x1:x2].copy()


def draw_detection(
    image: "np.ndarray",
    bbox: Tuple[float, float, float, float],
    label: Optional[str] = None,
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    landmarks: Optional[List[Tuple[float, float]]] = None,
) -> "np.ndarray":
    """
    Draw face detection on image.
    
    Args:
        image: Input image
        bbox: Bounding box
        label: Text label
        confidence: Detection confidence
        color: Box color (RGB)
        thickness: Line thickness
        landmarks: Facial landmarks to draw
    
    Returns:
        Image with detection drawn
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise ImportError("PIL and NumPy required")
    
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)
    
    x1, y1, x2, y2 = bbox
    
    # Draw box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    
    # Draw label
    if label or confidence is not None:
        text_parts = []
        if label:
            text_parts.append(label)
        if confidence is not None:
            text_parts.append(f"{confidence:.2%}")
        text = " - ".join(text_parts)
        
        # Try to get font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
        
        # Draw text background
        bbox_text = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)
    
    # Draw landmarks
    if landmarks:
        for lm in landmarks:
            x, y = lm
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))
    
    return np.array(img)


def draw_matches(
    image: "np.ndarray",
    matches: List[dict],
    top_k: int = 3,
) -> "np.ndarray":
    """
    Draw match results on image.
    
    Args:
        image: Input image with face
        matches: List of match results
        top_k: Number of matches to show
    
    Returns:
        Image with matches drawn
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise ImportError("PIL and NumPy required")
    
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)
    
    # Try to get font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    
    h, w = image.shape[:2]
    y_offset = 10
    
    for i, match in enumerate(matches[:top_k]):
        name = match.get("name", "Unknown")
        score = match.get("score", 0.0)
        
        text = f"{i+1}. {name}: {score:.2%}"
        
        # Color based on score
        if score >= 0.8:
            color = (0, 200, 0)
        elif score >= 0.6:
            color = (200, 200, 0)
        else:
            color = (200, 0, 0)
        
        draw.text((10, y_offset), text, fill=color, font=font)
        y_offset += 25
    
    return np.array(img)


def encode_image_base64(
    image: "np.ndarray",
    format: str = "JPEG",
    quality: int = 90,
) -> str:
    """
    Encode image as base64 string.
    
    Args:
        image: RGB image array
        format: Image format (JPEG, PNG)
        quality: JPEG quality
    
    Returns:
        Base64 encoded string
    """
    if not HAS_PIL:
        raise ImportError("PIL required")
    
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    
    if format.upper() == "JPEG":
        img.save(buffer, format="JPEG", quality=quality)
    else:
        img.save(buffer, format=format)
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_base64(
    encoded: str,
) -> "np.ndarray":
    """
    Decode base64 string to image.
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        RGB image array
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise ImportError("PIL and NumPy required")
    
    data = base64.b64decode(encoded)
    buffer = io.BytesIO(data)
    img = Image.open(buffer)
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return np.array(img)


def compute_image_hash(
    image: "np.ndarray",
    method: str = "average",
    size: int = 8,
) -> str:
    """
    Compute perceptual hash of an image.
    
    Args:
        image: Input image
        method: Hash method (average, difference, perceptual)
        size: Hash size
    
    Returns:
        Hash string
    """
    if not HAS_PIL or not HAS_NUMPY:
        raise ImportError("PIL and NumPy required")
    
    img = Image.fromarray(image)
    
    if method == "average":
        # Average hash
        img = img.convert("L")
        img = img.resize((size, size), Image.LANCZOS)
        pixels = np.array(img)
        avg = pixels.mean()
        bits = pixels > avg
        return "".join(str(int(b)) for b in bits.flatten())
    
    elif method == "difference":
        # Difference hash
        img = img.convert("L")
        img = img.resize((size + 1, size), Image.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join(str(int(b)) for b in diff.flatten())
    
    else:
        raise ValueError(f"Unknown hash method: {method}")


def blend_images(
    image1: "np.ndarray",
    image2: "np.ndarray",
    alpha: float = 0.5,
) -> "np.ndarray":
    """
    Blend two images together.
    
    Args:
        image1: First image
        image2: Second image
        alpha: Blend factor (0-1)
    
    Returns:
        Blended image
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy required")
    
    return (image1 * alpha + image2 * (1 - alpha)).astype(np.uint8)
