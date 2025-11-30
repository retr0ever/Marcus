"""
Face Preprocessing and Alignment
================================

Facial alignment using landmarks for consistent face representations.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2

from marcus_core.detection.base import FaceDetection


# Standard 5-point landmark template for 112x112 aligned face
# Positions for: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose
    [41.5493, 92.3655],   # Left mouth
    [70.7299, 92.2041],   # Right mouth
], dtype=np.float32)


# Template scaled for different output sizes
def get_template_for_size(target_size: Tuple[int, int]) -> np.ndarray:
    """
    Get landmark template scaled for target output size.
    
    Args:
        target_size: (width, height) of target image
    
    Returns:
        Scaled landmark template
    """
    scale_x = target_size[0] / 112.0
    scale_y = target_size[1] / 112.0
    template = ARCFACE_TEMPLATE.copy()
    template[:, 0] *= scale_x
    template[:, 1] *= scale_y
    return template


class FaceAligner:
    """
    Face alignment using affine transformation based on facial landmarks.
    
    Aligns faces to a standard template for consistent embedding extraction.
    
    Example:
        >>> aligner = FaceAligner(target_size=(112, 112))
        >>> aligned_face = aligner.align(image, face_detection)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (112, 112),
        template: Optional[np.ndarray] = None,
    ):
        """
        Initialize the face aligner.
        
        Args:
            target_size: Output face size (width, height)
            template: Custom landmark template. If None, uses ArcFace template.
        """
        self.target_size = target_size
        self.template = template if template is not None else get_template_for_size(target_size)
    
    def align(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        border_value: Tuple[int, int, int] = (0, 0, 0),
    ) -> Optional[np.ndarray]:
        """
        Align a detected face using landmarks.
        
        If landmarks are not available, falls back to simple cropping and resizing.
        
        Args:
            image: Source image (RGB, uint8)
            detection: Face detection with bbox and optional landmarks
            border_value: Border fill color for affine transform
        
        Returns:
            Aligned face image (RGB, uint8) or None if failed
        """
        if detection.landmarks is not None and len(detection.landmarks) >= 5:
            return self._align_with_landmarks(image, detection.landmarks, border_value)
        else:
            return self._align_without_landmarks(image, detection.bbox)
    
    def _align_with_landmarks(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        border_value: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Align face using 5-point landmarks.
        
        Uses similarity transformation to map detected landmarks to template.
        """
        try:
            # Estimate similarity transform
            src_pts = landmarks[:5].astype(np.float32)
            dst_pts = self.template.astype(np.float32)
            
            # Calculate similarity transform matrix
            transform_matrix = self._estimate_similarity_transform(src_pts, dst_pts)
            
            if transform_matrix is None:
                return self._align_without_landmarks(image, None)
            
            # Apply affine transformation
            aligned = cv2.warpAffine(
                image,
                transform_matrix,
                self.target_size,
                borderValue=border_value,
                flags=cv2.INTER_LINEAR,
            )
            
            return aligned
            
        except Exception as e:
            print(f"Landmark alignment failed: {e}")
            return None
    
    def _estimate_similarity_transform(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Estimate 2D similarity transformation matrix.
        
        Uses least squares to find optimal rotation, scale, and translation.
        """
        num_pts = src_pts.shape[0]
        
        # Build linear system
        # [x'] = [s*cos(θ)  -s*sin(θ)  tx] [x]
        # [y']   [s*sin(θ)   s*cos(θ)  ty] [y]
        #                                   [1]
        
        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)
        
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        
        # Calculate scale and rotation
        src_std = np.std(src_centered)
        dst_std = np.std(dst_centered)
        
        if src_std < 1e-6:
            return None
        
        scale = dst_std / src_std
        
        # SVD for rotation
        H = dst_centered.T @ src_centered
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        
        # Build transformation matrix
        transform = np.zeros((2, 3), dtype=np.float32)
        transform[:2, :2] = scale * R
        transform[:, 2] = dst_mean - scale * R @ src_mean
        
        return transform
    
    def _align_without_landmarks(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        """
        Align face using simple crop and resize (fallback).
        """
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin = 0.2
        
        x1 = max(0, int(x1 - face_w * margin))
        y1 = max(0, int(y1 - face_h * margin))
        x2 = min(w, int(x2 + face_w * margin))
        y2 = min(h, int(y2 + face_h * margin))
        
        # Crop and resize
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        aligned = cv2.resize(face, self.target_size, interpolation=cv2.INTER_LINEAR)
        return aligned


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Convenience function to align a face using landmarks.
    
    Args:
        image: Source image (RGB)
        landmarks: 5-point facial landmarks
        target_size: Output size
    
    Returns:
        Aligned face image
    """
    aligner = FaceAligner(target_size=target_size)
    detection = FaceDetection(
        bbox=(0, 0, image.shape[1], image.shape[0]),
        confidence=1.0,
        landmarks=landmarks,
    )
    aligned = aligner.align(image, detection)
    return aligned if aligned is not None else cv2.resize(image, target_size)


def estimate_pose(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate face pose (yaw, pitch, roll) from 5-point landmarks.
    
    This is a simplified estimation based on landmark positions.
    
    Args:
        landmarks: 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    
    Returns:
        Tuple of (yaw, pitch, roll) in degrees
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Eye center
    eye_center = (left_eye + right_eye) / 2
    
    # Mouth center
    mouth_center = (left_mouth + right_mouth) / 2
    
    # Yaw estimation (left-right turn)
    eye_width = np.linalg.norm(right_eye - left_eye)
    nose_offset = nose[0] - eye_center[0]
    yaw = np.arctan2(nose_offset, eye_width / 2) * 180 / np.pi
    
    # Pitch estimation (up-down tilt)
    vertical_dist = mouth_center[1] - eye_center[1]
    nose_vertical = nose[1] - eye_center[1]
    expected_nose = vertical_dist * 0.4
    pitch = (nose_vertical - expected_nose) / vertical_dist * 30
    
    # Roll estimation (head tilt)
    roll = np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ) * 180 / np.pi
    
    return (float(yaw), float(pitch), float(roll))


def check_face_quality(
    face: np.ndarray,
    min_brightness: float = 40,
    max_brightness: float = 220,
    min_contrast: float = 30,
    blur_threshold: float = 100,
) -> Tuple[bool, dict]:
    """
    Check face image quality for embedding extraction.
    
    Args:
        face: Face image (RGB or grayscale)
        min_brightness: Minimum acceptable brightness
        max_brightness: Maximum acceptable brightness
        min_contrast: Minimum acceptable contrast
        blur_threshold: Laplacian variance threshold for blur
    
    Returns:
        Tuple of (is_acceptable, quality_metrics)
    """
    # Convert to grayscale if needed
    if len(face.shape) == 3:
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    else:
        gray = face
    
    # Brightness
    brightness = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    # Blur (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    # Quality checks
    is_bright_ok = min_brightness <= brightness <= max_brightness
    is_contrast_ok = contrast >= min_contrast
    is_sharp_ok = blur_score >= blur_threshold
    
    is_acceptable = is_bright_ok and is_contrast_ok and is_sharp_ok
    
    metrics = {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "blur_score": float(blur_score),
        "is_bright_ok": is_bright_ok,
        "is_contrast_ok": is_contrast_ok,
        "is_sharp_ok": is_sharp_ok,
        "is_acceptable": is_acceptable,
    }
    
    return is_acceptable, metrics
