"""
Region labeling using simple feature-based classification.
Classifies regions as text, table, figure, or blank.
"""

from typing import Tuple

import cv2
import numpy as np

from src.region import Region


def compute_edge_density(gray_roi: np.ndarray) -> float:
    """
    Compute edge density using Canny edge detection.
    
    Args:
        gray_roi: Grayscale ROI
        
    Returns:
        Edge density (0 to 1)
    """
    if gray_roi.size == 0:
        return 0.0
    
    # Canny edge detection
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Density = fraction of edge pixels
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    
    return edge_pixels / total_pixels if total_pixels > 0 else 0.0


def compute_content_density(binary_roi: np.ndarray) -> float:
    """
    Compute content density (ratio of dark pixels).
    
    Args:
        binary_roi: Binary ROI (0=black content, 255=white background)
        
    Returns:
        Content density (0 to 1)
    """
    if binary_roi.size == 0:
        return 0.0
    
    # Count dark pixels (content)
    dark_pixels = np.sum(binary_roi < 128)
    total_pixels = binary_roi.size
    
    return dark_pixels / total_pixels


def has_grid_structure(gray_roi: np.ndarray) -> Tuple[bool, int, int]:
    """
    Detect grid structure (indicates table).
    
    Looks for horizontal and vertical lines using edge detection
    and Hough line transform.
    
    Args:
        gray_roi: Grayscale ROI
        
    Returns:
        Tuple of (has_grid, num_h_lines, num_v_lines)
    """
    if gray_roi.size == 0 or gray_roi.shape[0] < 50 or gray_roi.shape[1] < 50:
        return False, 0, 0
    
    # Edge detection
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=min(gray_roi.shape) // 3,
                            maxLineGap=10)
    
    if lines is None:
        return False, 0, 0
    
    # Classify lines as horizontal or vertical
    h_lines = 0
    v_lines = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > dy * 3:  # Mostly horizontal (angle ~0 or ~180)
            h_lines += 1
        elif dy > dx * 3:  # Mostly vertical (angle ~90)
            v_lines += 1
    
    # Grid structure requires both horizontal and vertical lines
    has_grid = h_lines >= 2 and v_lines >= 2
    
    return has_grid, h_lines, v_lines


def is_figure_like(gray_roi: np.ndarray, edge_density: float, 
                   content_density: float, aspect_ratio: float,
                   area: int) -> bool:
    """
    Check if region has figure characteristics (plots, diagrams, images).
    
    Figures typically have:
    - Complex visual structure (high edge density)
    - Square-ish or landscape aspect ratio (NOT tall columns)
    - Varied content distribution (not uniform like text)
    - Moderate size (not too small, not entire page)
    """
    # Size constraints - figures have reasonable bounds
    if area < 40000:  # Too small for typical figure
        return False
    
    if area > 800000:  # Too large - likely merged multiple regions
        return False
    
    # Aspect ratio check - figures are square-ish or wide, not tall narrow columns
    if aspect_ratio < 0.5 or aspect_ratio > 4.0:
        return False  # Too tall/narrow or too wide
    
    # High edge density is strong indicator of complex visual content
    if edge_density > 0.12:
        # But check it's not just a dense text block
        if aspect_ratio >= 0.6 and aspect_ratio <= 2.5:
            return True
    
    # Moderate edge density + good size/shape + sufficient content
    if 0.07 <= edge_density <= 0.15:
        if 60000 <= area <= 600000:
            # Good aspect ratio for plots/diagrams
            if 0.6 <= aspect_ratio <= 2.0:
                # Check content isn't too sparse (not mostly white)
                if content_density > 0.03:
                    return True
    
    # Check content distribution variance (figures have varied visual patterns)
    if gray_roi.size > 0 and 50000 <= area <= 500000:
        h, w = gray_roi.shape
        if h > 50 and w > 50:
            # Split into quadrants and check variance
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                gray_roi[:mid_h, :mid_w],
                gray_roi[:mid_h, mid_w:],
                gray_roi[mid_h:, :mid_w],
                gray_roi[mid_h:, mid_w:]
            ]
            stds = [np.std(q) for q in quadrants if q.size > 0]
            if stds and len(stds) == 4:
                variance_of_std = np.var(stds)
                # Figures have varied patterns across quadrants
                if variance_of_std > 800 and edge_density > 0.06:
                    if 0.6 <= aspect_ratio <= 2.0:
                        return True
    
    return False


def is_table_like(has_grid: bool, h_lines: int, v_lines: int,
                  edge_density: float, content_density: float,
                  aspect_ratio: float, area: int) -> bool:
    """
    Check if region has table characteristics.
    
    Tables have:
    - Grid structure with borders
    - Text content arranged in cells
    - Rectangular shape
    - Multiple rows and columns
    """
    # Strong grid structure
    if has_grid and h_lines >= 3 and v_lines >= 2:
        return True
    
    # Moderate edge density from borders, but not as high as figures
    # Tables are more structured and less "busy" visually
    if 0.04 <= edge_density <= 0.10:
        # Good content density (text in cells)
        if 0.02 <= content_density <= 0.15:
            # Reasonable size and shape
            if area > 15000 and 0.3 <= aspect_ratio <= 4.0:
                return True
    
    return False


def label_region(gray: np.ndarray, binary: np.ndarray, region: Region) -> str:
    """
    Classify a region based on its visual features.
    
    Improved classification with better figure/table distinction.
    
    Args:
        gray: Full grayscale image
        binary: Full binary image
        region: Region to classify
        
    Returns:
        Label: 'text', 'table', 'figure', or 'blank'
    """
    # Extract ROIs
    gray_roi = gray[region.y:region.y2, region.x:region.x2]
    binary_roi = binary[region.y:region.y2, region.x:region.x2]
    
    # Compute features
    edge_density = compute_edge_density(gray_roi)
    content_density = compute_content_density(binary_roi)
    aspect_ratio = region.aspect_ratio
    area = region.area
    
    # Classification rules (hierarchical, order matters!)
    
    # 1. Blank region (very low content)
    if content_density < 0.005 and edge_density < 0.01:
        return "blank"
    
    # Very small regions with low content
    if area < 5000 and content_density < 0.03:
        return "blank"
    
    # 2. Figure detection (BEFORE table, as figures can have grid-like elements)
    if is_figure_like(gray_roi, edge_density, content_density, 
                      aspect_ratio, area):
        return "figure"
    
    # 3. Table detection
    has_grid, h_lines, v_lines = has_grid_structure(gray_roi)
    
    if is_table_like(has_grid, h_lines, v_lines, edge_density, 
                     content_density, aspect_ratio, area):
        return "table"
    
    # 4. Text detection
    # Text has lower edge density than figures, moderate content
    if 0.02 <= edge_density <= 0.12 and content_density > 0.01:
        # Text regions are often tall (vertical stacking) or wide (horizontal)
        # but not square like figures
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            return "text"
        
        # Moderate size text regions
        if 5000 <= area <= 150000:
            return "text"
    
    # 5. Default based on content
    if content_density > 0.02:
        # Has substantial content
        if edge_density > 0.08:
            # Complex content -> likely figure
            return "figure"
        else:
            # Simple content -> likely text
            return "text"
    
    # 6. Low content default
    if content_density > 0.005:
        return "text"
    
    # 7. Final fallback
    return "blank"


def label_regions(gray: np.ndarray, binary: np.ndarray, 
                  regions: list) -> list:
    """
    Label all regions in the list.
    
    Args:
        gray: Full grayscale image
        binary: Full binary image
        regions: List of Region objects
        
    Returns:
        List of labeled Region objects (regions are modified in place)
    """
    print(f"\nLabeling {len(regions)} regions...")
    
    label_counts = {"text": 0, "table": 0, "figure": 0, "blank": 0}
    
    for i, region in enumerate(regions):
        label = label_region(gray, binary, region)
        region.label = label
        label_counts[label] += 1
    
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        if count > 0:
            print(f"    {label}: {count}")
    
    return regions

