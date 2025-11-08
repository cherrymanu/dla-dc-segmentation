"""
Main pipeline for document layout segmentation.
Integrates all steps: preprocessing, XY-cut, labeling, and merging.
"""

from typing import List, Tuple

import numpy as np

from src.labeling import label_regions
from src.merge import greedy_merge, merge_small_regions, remove_blank_regions
from src.preprocessing import preprocess_image
from src.region import Region
from src.xycut import segment_page_xycut


def segment_document(image_path: str,
                     min_region: int = 80,
                     max_depth: int = 15,
                     do_merge: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Region]]:
    """
    Complete document layout segmentation pipeline.
    
    Args:
        image_path: Path to document image
        min_region: Minimum region size for XY-cut
        max_depth: Maximum recursion depth for XY-cut
        do_merge: Whether to perform greedy merging
        
    Returns:
        Tuple of (original, gray, binary, regions)
    """
    print("=" * 60)
    print("DOCUMENT LAYOUT SEGMENTATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Preprocessing
    print("\n[Step 1/4] Preprocessing...")
    original, gray, binary = preprocess_image(image_path)
    
    # Step 2: XY-Cut Segmentation
    print("\n[Step 2/4] XY-Cut Segmentation...")
    regions = segment_page_xycut(binary, min_region, max_depth)
    
    # Step 3: Labeling
    print("\n[Step 3/4] Region Labeling...")
    regions = label_regions(gray, binary, regions)
    
    # Step 4: Merging (optional)
    if do_merge:
        print("\n[Step 4/4] Greedy Merging...")
        
        # Main merge
        regions = greedy_merge(regions, padding=30)
        
        # Merge small regions
        regions = merge_small_regions(regions, max_size=10000, padding=30)
        
        # Remove blank regions
        regions = remove_blank_regions(regions)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal results:")
    print(f"  Total regions: {len(regions)}")
    
    label_counts = {}
    for r in regions:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
    
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")
    
    return original, gray, binary, regions

