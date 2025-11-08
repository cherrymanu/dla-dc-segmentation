"""Test XY-Cut segmentation algorithm."""

import cv2
import numpy as np

from src.preprocessing import preprocess_image
from src.xycut import segment_page_xycut


def visualize_regions(image: np.ndarray, regions, output_path: str):
    """Draw bounding boxes on image to visualize regions."""
    # Convert to color if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Draw each region
    for i, region in enumerate(regions):
        # Random color for each region
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255))
        )
        
        # Draw rectangle
        cv2.rectangle(vis, (region.x, region.y), 
                     (region.x2, region.y2), color, 2)
        
        # Draw region number
        cv2.putText(vis, str(i), 
                   (region.x + 5, region.y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to {output_path}")


def test_xycut():
    """Test XY-Cut segmentation on academic.jpg."""
    
    print("=== Testing XY-Cut Segmentation ===\n")
    
    # Load and preprocess
    print("1. Preprocessing image...")
    original, gray, binary = preprocess_image("inputs/academic.jpg")
    
    # Run XY-Cut
    print("\n2. Running XY-Cut segmentation...")
    regions = segment_page_xycut(binary)  # Use default params: min_region=80, max_depth=15
    
    # Print region statistics
    print(f"\n3. Region statistics:")
    print(f"   Total regions: {len(regions)}")
    
    if regions:
        areas = [r.area for r in regions]
        print(f"   Region areas:")
        print(f"     - Min: {min(areas)} pixels")
        print(f"     - Max: {max(areas)} pixels")
        print(f"     - Mean: {np.mean(areas):.0f} pixels")
        print(f"     - Median: {np.median(areas):.0f} pixels")
    
    # Show first few regions
    print(f"\n4. First 10 regions:")
    for i, region in enumerate(regions[:10]):
        print(f"   Region {i}: {region}")
    
    if len(regions) > 10:
        print(f"   ... and {len(regions) - 10} more")
    
    # Visualize
    print(f"\n5. Creating visualization...")
    visualize_regions(original, regions, "outputs/xycut_result.jpg")
    
    print("\n✓ XY-Cut test completed!")
    print("Check outputs/xycut_result.jpg to see the segmentation")


if __name__ == "__main__":
    test_xycut()

