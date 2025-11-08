"""
Greedy region merging to combine over-segmented regions.
"""

from typing import List

from src.region import Region


def should_merge(r1: Region, r2: Region, padding: int = 30) -> bool:
    """
    Determine if two regions should be merged.
    
    Args:
        r1: First region
        r2: Second region
        padding: Adjacency padding in pixels
        
    Returns:
        True if regions should be merged
    """
    # Must be adjacent
    if not r1.is_adjacent(r2, padding):
        return False
    
    # Same label - always merge
    if r1.label == r2.label:
        return True
    
    # Both text-like (text or table are similar content)
    # This helps merge text regions that were mislabeled
    text_like = {"text", "table"}
    if r1.label in text_like and r2.label in text_like:
        # Only merge if vertically aligned (same column)
        if r1.is_vertically_aligned(r2, threshold=0.5):
            return True
    
    # Figures and tables should stay separate
    # But figure pieces might be mislabeled as tables
    if (r1.label == "figure" and r2.label == "table") or \
       (r1.label == "table" and r2.label == "figure"):
        # Check if they're in the same column and close
        if r1.is_vertically_aligned(r2, threshold=0.7):
            # Merge if very close vertically
            vertical_gap = abs(r1.y - r2.y2) if r1.y > r2.y2 else abs(r2.y - r1.y2)
            if vertical_gap < padding:
                return True
    
    return False


def greedy_merge(regions: List[Region], padding: int = 30) -> List[Region]:
    """
    Iteratively merge adjacent compatible regions.
    
    Args:
        regions: List of regions to merge
        padding: Adjacency padding in pixels
        
    Returns:
        List of merged regions
    """
    print(f"\nGreedy merge (padding={padding}px)...")
    print(f"  Starting with {len(regions)} regions")
    
    iteration = 0
    while True:
        iteration += 1
        merged_any = False
        new_regions = []
        merged_indices = set()
        
        for i, r1 in enumerate(regions):
            if i in merged_indices:
                continue
            
            # Try to find a merge partner
            merge_partner = None
            for j, r2 in enumerate(regions):
                if j <= i or j in merged_indices:
                    continue
                
                if should_merge(r1, r2, padding):
                    merge_partner = j
                    break
            
            if merge_partner is not None:
                # Merge r1 and r2
                r2 = regions[merge_partner]
                merged = Region.merge([r1, r2])
                new_regions.append(merged)
                merged_indices.add(i)
                merged_indices.add(merge_partner)
                merged_any = True
            else:
                # Keep r1 as is
                new_regions.append(r1)
                merged_indices.add(i)
        
        regions = new_regions
        
        if not merged_any:
            break
    
    print(f"  Completed in {iteration} iterations")
    print(f"  Final: {len(regions)} regions")
    
    return regions


def merge_small_regions(regions: List[Region], 
                        max_size: int = 10000,
                        padding: int = 30) -> List[Region]:
    """
    Merge very small adjacent regions in the same column.
    
    This helps clean up tiny over-segmented pieces.
    
    Args:
        regions: List of regions
        max_size: Maximum area for "small" regions
        padding: Adjacency padding
        
    Returns:
        List of regions with small ones merged
    """
    print(f"\nMerging small regions (area < {max_size})...")
    print(f"  Starting with {len(regions)} regions")
    
    merged_any = True
    while merged_any:
        merged_any = False
        new_regions = []
        merged_indices = set()
        
        for i, r1 in enumerate(regions):
            if i in merged_indices:
                continue
            
            # If r1 is small, try to merge with adjacent
            if r1.area < max_size:
                merge_partner = None
                
                for j, r2 in enumerate(regions):
                    if j <= i or j in merged_indices:
                        continue
                    
                    # Must be adjacent and vertically aligned
                    if r1.is_adjacent(r2, padding) and \
                       r1.is_vertically_aligned(r2, threshold=0.5):
                        merge_partner = j
                        break
                
                if merge_partner is not None:
                    r2 = regions[merge_partner]
                    merged = Region.merge([r1, r2])
                    new_regions.append(merged)
                    merged_indices.add(i)
                    merged_indices.add(merge_partner)
                    merged_any = True
                    continue
            
            # Keep r1 as is
            new_regions.append(r1)
            merged_indices.add(i)
        
        regions = new_regions
    
    print(f"  Final: {len(regions)} regions")
    
    return regions


def remove_blank_regions(regions: List[Region]) -> List[Region]:
    """
    Remove regions labeled as blank.
    
    Args:
        regions: List of regions
        
    Returns:
        List of non-blank regions
    """
    non_blank = [r for r in regions if r.label != "blank"]
    
    if len(non_blank) < len(regions):
        print(f"\nRemoved {len(regions) - len(non_blank)} blank regions")
    
    return non_blank

