"""
Region dataclass for document layout segmentation.
Represents a rectangular region with label and spatial information.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Region:
    """Represents a rectangular region in the document."""
    
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    w: int  # Width
    h: int  # Height
    label: str = "unknown"  # text, table, figure, blank, unknown
    
    def __post_init__(self):
        """Ensure non-negative dimensions."""
        self.w = max(0, self.w)
        self.h = max(0, self.h)
    
    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.w
    
    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.h
    
    @property
    def area(self) -> int:
        """Area of the region."""
        return self.w * self.h
    
    @property
    def center(self) -> tuple:
        """Center point (cx, cy)."""
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio. Returns inf if height is 0."""
        return self.w / self.h if self.h > 0 else float('inf')
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point is inside the region."""
        return self.x <= px < self.x2 and self.y <= py < self.y2
    
    def overlaps(self, other: 'Region', padding: int = 0) -> bool:
        """
        Check if this region overlaps with another region.
        
        Args:
            other: Another region
            padding: Expand regions by padding pixels before checking overlap
            
        Returns:
            True if regions overlap
        """
        # Expand both regions by padding
        x1_min = self.x - padding
        x1_max = self.x2 + padding
        y1_min = self.y - padding
        y1_max = self.y2 + padding
        
        x2_min = other.x - padding
        x2_max = other.x2 + padding
        y2_min = other.y - padding
        y2_max = other.y2 + padding
        
        # Check if rectangles overlap
        return not (x1_max <= x2_min or x2_max <= x1_min or
                    y1_max <= y2_min or y2_max <= y1_min)
    
    def is_adjacent(self, other: 'Region', padding: int = 30) -> bool:
        """
        Check if this region is adjacent to another (for merging).
        Uses overlaps with padding to determine adjacency.
        
        Args:
            other: Another region
            padding: Distance threshold for adjacency (default: 30px)
            
        Returns:
            True if regions are adjacent
        """
        return self.overlaps(other, padding)
    
    def iou(self, other: 'Region') -> float:
        """
        Calculate Intersection over Union (IoU) with another region.
        
        Args:
            other: Another region
            
        Returns:
            IoU score between 0 and 1
        """
        # Calculate intersection
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_vertically_aligned(self, other: 'Region', threshold: float = 0.5) -> bool:
        """
        Check if two regions are vertically aligned (same column).
        
        Args:
            other: Another region
            threshold: Minimum overlap ratio to consider aligned
            
        Returns:
            True if regions are in the same column
        """
        # Calculate horizontal overlap
        x_left = max(self.x, other.x)
        x_right = min(self.x2, other.x2)
        
        if x_right <= x_left:
            return False
        
        overlap = x_right - x_left
        min_width = min(self.w, other.w)
        
        return overlap / min_width >= threshold if min_width > 0 else False
    
    def is_horizontally_aligned(self, other: 'Region', threshold: float = 0.5) -> bool:
        """
        Check if two regions are horizontally aligned (same row).
        
        Args:
            other: Another region
            threshold: Minimum overlap ratio to consider aligned
            
        Returns:
            True if regions are in the same row
        """
        # Calculate vertical overlap
        y_top = max(self.y, other.y)
        y_bottom = min(self.y2, other.y2)
        
        if y_bottom <= y_top:
            return False
        
        overlap = y_bottom - y_top
        min_height = min(self.h, other.h)
        
        return overlap / min_height >= threshold if min_height > 0 else False
    
    @staticmethod
    def merge(regions: List['Region']) -> 'Region':
        """
        Merge multiple regions into a single bounding box region.
        
        Args:
            regions: List of regions to merge
            
        Returns:
            New region that encompasses all input regions
        """
        if not regions:
            raise ValueError("Cannot merge empty list of regions")
        
        # Find bounding box
        x_min = min(r.x for r in regions)
        y_min = min(r.y for r in regions)
        x_max = max(r.x2 for r in regions)
        y_max = max(r.y2 for r in regions)
        
        # Use most common label, or first region's label
        labels = [r.label for r in regions]
        most_common_label = max(set(labels), key=labels.count)
        
        return Region(
            x=x_min,
            y=y_min,
            w=x_max - x_min,
            h=y_max - y_min,
            label=most_common_label
        )
    
    def to_dict(self) -> dict:
        """Convert region to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'h': self.h,
            'x2': self.x2,
            'y2': self.y2,
            'area': self.area,
            'label': self.label
        }
    
    def __repr__(self) -> str:
        return f"Region(x={self.x}, y={self.y}, w={self.w}, h={self.h}, label='{self.label}')"

