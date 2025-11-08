"""
Preprocessing utilities for document layout analysis.
Handles image loading, grayscale conversion, and binarization.
"""

from typing import Tuple

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        image: Input image (BGR or already grayscale)
        
    Returns:
        Grayscale image
    """
    # Check if already grayscale
    if len(image.shape) == 2:
        return image
    
    # Convert BGR to grayscale
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unexpected number of channels: {image.shape[2]}")


def otsu_binarize(gray_image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Apply Otsu's adaptive thresholding to binarize image.
    
    Otsu's method automatically determines the optimal threshold value
    by maximizing the between-class variance.
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        Tuple of (binary_image, threshold_value)
        - binary_image: Binarized image (0 or 255)
        - threshold_value: Computed Otsu threshold
    """
    # Apply Otsu's thresholding
    threshold, binary = cv2.threshold(
        gray_image,
        0,  # threshold value (ignored, auto-computed)
        255,  # max value
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary, int(threshold)


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline: load, grayscale, binarize.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (original_image, gray_image, binary_image)
    """
    # Load image
    original = load_image(image_path)
    
    # Convert to grayscale
    gray = to_grayscale(original)
    
    # Binarize with Otsu
    binary, threshold = otsu_binarize(gray)
    
    print(f"Image loaded: {original.shape[:2][::-1]} (W x H)")
    print(f"Otsu threshold: {threshold}")
    
    return original, gray, binary


def get_content_density(binary_image: np.ndarray, roi: Tuple[int, int, int, int] = None) -> float:
    """
    Calculate content density (ratio of dark pixels) in an image or ROI.
    
    Args:
        binary_image: Binary image (0=white, 255=black after inversion)
        roi: Optional (x, y, w, h) region of interest
        
    Returns:
        Content density as float between 0 and 1
    """
    if roi is not None:
        x, y, w, h = roi
        region = binary_image[y:y+h, x:x+w]
    else:
        region = binary_image
    
    if region.size == 0:
        return 0.0
    
    # Invert: dark pixels (text/content) should be counted
    # In binary image, content is typically black (0), background is white (255)
    dark_pixels = np.sum(region < 128)
    total_pixels = region.size
    
    return dark_pixels / total_pixels


def save_image(image: np.ndarray, output_path: str, is_binary: bool = False) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image to save
        output_path: Output file path
        is_binary: If True, force PNG format to avoid compression artifacts
    """
    # For binary images, force PNG to preserve exact pixel values
    if is_binary and not output_path.lower().endswith('.png'):
        output_path = output_path.rsplit('.', 1)[0] + '.png'
    
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Failed to save image to {output_path}")


def visualize_preprocessing(original: np.ndarray, gray: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side visualization of preprocessing steps.
    
    Args:
        original: Original color image
        gray: Grayscale image
        binary: Binary image
        
    Returns:
        Combined visualization image
    """
    # Convert grayscale and binary to BGR for concatenation
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Resize if images are large
    h, w = original.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        gray_bgr = cv2.resize(gray_bgr, (new_w, new_h))
        binary_bgr = cv2.resize(binary_bgr, (new_w, new_h))
    
    # Concatenate horizontally
    combined = np.hstack([original, gray_bgr, binary_bgr])
    
    return combined

