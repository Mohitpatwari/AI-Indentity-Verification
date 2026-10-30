"""Shared image decoding/validation utilities."""
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    """
    Decode raw bytes into an OpenCV BGR image array.
    Returns None if decoding fails (corrupt/unsupported file).
    """
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        return None