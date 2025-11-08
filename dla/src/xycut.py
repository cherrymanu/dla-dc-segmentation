"""
XY-Cut divide-and-conquer algorithm for document layout segmentation.
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.region import Region


def compute_projection_profile(binary_roi: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute projection profile along an axis.
    
    For binary images where content is black (0) and background is white (255),
    we sum the darkness (255 - pixel_value) along the perpendicular axis.
    
    Args:
        binary_roi: Binary image region (0=black content, 255=white background)
        axis: 0 for vertical projection (sum along rows), 
              1 for horizontal projection (sum along columns)
    
    Returns:
        Projection profile array
    """
    # Invert so content (black=0) becomes high values
    inverted = 255 - binary_roi
    
    # Sum along perpendicular axis
    if axis == 0:
        # Vertical projection: sum along rows (axis=0)
        # Result has length = width
        profile = np.sum(inverted, axis=0)
    else:
        # Horizontal projection: sum along columns (axis=1)
        # Result has length = height
        profile = np.sum(inverted, axis=1)
    
    return profile


def find_valleys(profile: np.ndarray, min_gap: int = 5, 
                 large_gap_threshold: int = 40) -> List[Tuple[int, int]]:
    """
    Find valleys (low-content regions) in projection profile.
    
    Valleys indicate whitespace gaps between content blocks.
    Prioritizes larger gaps (semantic separators) over small line gaps.
    
    Args:
        profile: Projection profile array
        min_gap: Minimum valley width in pixels
        large_gap_threshold: Threshold for "large" gaps (default: 40px, semantic level)
    
    Returns:
        List of (start, end) indices for each valley
    """
    if len(profile) == 0:
        return []
    
    # Normalize profile to [0, 1]
    max_val = profile.max()
    if max_val == 0:
        return []  # Empty region
    
    normalized = profile / max_val
    
    # Simple threshold: mean - std
    mean = normalized.mean()
    std = normalized.std()
    threshold = max(0.1, mean - std)
    
    # Find regions below threshold
    below_threshold = normalized < threshold
    
    # Group consecutive low values into valleys
    all_valleys = []
    in_valley = False
    start = 0
    
    for i, is_low in enumerate(below_threshold):
        if is_low and not in_valley:
            # Start of a valley
            start = i
            in_valley = True
        elif not is_low and in_valley:
            # End of a valley
            end = i
            if end - start >= min_gap:
                all_valleys.append((start, end))
            in_valley = False
    
    # Handle valley at the end
    if in_valley and len(profile) - start >= min_gap:
        all_valleys.append((start, len(profile)))
    
    # Filter: prefer large gaps (semantic separators)
    large_valleys = [(s, e) for s, e in all_valleys if e - s >= large_gap_threshold]
    
    # If we found large gaps, use only those; otherwise use all valleys
    return large_valleys if len(large_valleys) > 0 else all_valleys


def valleys_to_split_indices(valleys: List[Tuple[int, int]], 
                             dimension: int,
                             min_size: int) -> List[Tuple[int, int]]:
    """
    Convert valleys to split indices (regions to keep).
    
    Args:
        valleys: List of (start, end) valley positions
        dimension: Total dimension (width or height)
        min_size: Minimum region size
    
    Returns:
        List of (start, end) for content regions (between valleys)
    """
    if not valleys:
        return [(0, dimension)]
    
    splits = []
    current = 0
    
    for valley_start, valley_end in valleys:
        # Content region before this valley
        if valley_start - current >= min_size:
            splits.append((current, valley_start))
        current = valley_end
    
    # Content region after last valley
    if dimension - current >= min_size:
        splits.append((current, dimension))
    
    # If no valid splits, return whole region
    if not splits:
        return [(0, dimension)]
    
    return splits


def should_stop(roi: np.ndarray, w: int, h: int, min_size: int) -> bool:
    """
    Determine if recursion should stop based on region properties.
    
    Args:
        roi: Binary image region
        w: Width
        h: Height
        min_size: Minimum dimension threshold
    
    Returns:
        True if should stop splitting
    """
    # Check minimum size
    if w < min_size or h < min_size:
        return True
    
    # Check if region is mostly empty (edge density check)
    # For simplicity, just check content density
    content_pixels = np.sum(roi < 128)
    total_pixels = roi.size
    
    if total_pixels == 0:
        return True
    
    content_density = content_pixels / total_pixels
    
    # Stop if very low content (< 1%)
    if content_density < 0.01:
        return True
    
    return False


def choose_split_direction(vertical_splits: List[Tuple[int, int]],
                          horizontal_splits: List[Tuple[int, int]],
                          w: int, h: int, depth: int) -> bool:
    """
    Choose whether to split vertically or horizontally.
    
    Args:
        vertical_splits: List of vertical split regions
        horizontal_splits: List of horizontal split regions
        w: Width
        h: Height
        depth: Current recursion depth
    
    Returns:
        True for vertical split, False for horizontal split
    """
    # Prefer vertical splits for wide regions (columns)
    # Prefer horizontal splits for tall regions (rows)
    
    num_v = len(vertical_splits)
    num_h = len(horizontal_splits)
    
    # If only one direction has splits, use it
    if num_v > 1 and num_h == 1:
        return True
    if num_h > 1 and num_v == 1:
        return False
    
    # If both have splits, prefer based on aspect ratio
    if w > h * 1.5:
        # Wide region: prefer vertical splits (columns)
        return True
    elif h > w * 1.5:
        # Tall region: prefer horizontal splits (rows)
        return False
    
    # Similar aspect ratio: prefer more splits
    if num_v > num_h:
        return True
    else:
        return False


def xycut_recursive(binary: np.ndarray,
                    x: int, y: int, w: int, h: int,
                    min_size: int, depth: int, max_depth: int) -> List[Region]:
    """
    Recursive XY-Cut algorithm.
    
    Args:
        binary: Binary image (full page)
        x, y: Top-left corner of current region
        w, h: Width and height of current region
        min_size: Minimum dimension for splitting
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    
    Returns:
        List of leaf regions
    """
    # Extract ROI
    roi = binary[y:y+h, x:x+w]
    
    # Base cases
    if depth >= max_depth or should_stop(roi, w, h, min_size):
        return [Region(x=x, y=y, w=w, h=h)]
    
    # Compute projection profiles
    v_profile = compute_projection_profile(roi, axis=0)  # Vertical
    h_profile = compute_projection_profile(roi, axis=1)  # Horizontal
    
    # Find valleys
    v_valleys = find_valleys(v_profile, min_gap=5)
    h_valleys = find_valleys(h_profile, min_gap=5)
    
    # Convert valleys to split indices
    v_splits = valleys_to_split_indices(v_valleys, w, min_size)
    h_splits = valleys_to_split_indices(h_valleys, h, min_size)
    
    # Choose split direction
    use_vertical = choose_split_direction(v_splits, h_splits, w, h, depth)
    
    # Perform split
    regions = []
    
    if use_vertical and len(v_splits) > 1:
        # Split vertically (into columns)
        for x_start, x_end in v_splits:
            sub_regions = xycut_recursive(
                binary, 
                x + x_start, y, 
                x_end - x_start, h,
                min_size, depth + 1, max_depth
            )
            regions.extend(sub_regions)
    elif len(h_splits) > 1:
        # Split horizontally (into rows)
        for y_start, y_end in h_splits:
            sub_regions = xycut_recursive(
                binary,
                x, y + y_start,
                w, y_end - y_start,
                min_size, depth + 1, max_depth
            )
            regions.extend(sub_regions)
    else:
        # No good split found, return as leaf
        regions = [Region(x=x, y=y, w=w, h=h)]
    
    return regions


def segment_page_xycut(binary: np.ndarray, 
                       min_region: int = 80, 
                       max_depth: int = 15) -> List[Region]:
    """
    Segment a page using XY-Cut algorithm.
    
    Args:
        binary: Binary image of the page
        min_region: Minimum region dimension in pixels (default: 80)
        max_depth: Maximum recursion depth (default: 15)
    
    Returns:
        List of segmented regions
    """
    h, w = binary.shape
    
    print(f"Starting XY-Cut segmentation:")
    print(f"  Image size: {w} x {h}")
    print(f"  Min region: {min_region}px")
    print(f"  Max depth: {max_depth}")
    
    regions = xycut_recursive(binary, 0, 0, w, h, min_region, 0, max_depth)
    
    print(f"  Segmentation complete: {len(regions)} regions found")
    
    return regions

