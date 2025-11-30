"""
Utility Module
==============

Common utilities for the Marcus face analysis system.
"""

from marcus_core.utils.image_utils import (
    load_image,
    save_image,
    resize_image,
    crop_face,
    draw_detection,
    draw_matches,
    encode_image_base64,
    decode_image_base64,
)
from marcus_core.utils.device import (
    get_device,
    get_device_info,
    set_device,
    is_cuda_available,
    is_mps_available,
)

__all__ = [
    "load_image",
    "save_image",
    "resize_image",
    "crop_face",
    "draw_detection",
    "draw_matches",
    "encode_image_base64",
    "decode_image_base64",
    "get_device",
    "get_device_info",
    "set_device",
    "is_cuda_available",
    "is_mps_available",
]
